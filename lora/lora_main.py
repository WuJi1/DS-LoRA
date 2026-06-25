from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from util.generate import generate_fast
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import EarlyStoppingCallback
from .lora_hparams import LoRAHyperParams
PREFIX_CACHE = None

class EpochLoggerAndEarlyStop(TrainerCallback):
    def __init__(self, print_every_epoch=1, min_loss=0.01, patience=100):
        self.epoch_losses = []
        self.print_every_epoch = print_every_epoch
        self.min_loss = min_loss
        self.patience = patience
        self.best_loss = float('inf')
        self.early_stop_counter = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        # 优先找最后一个有 loss 字段的 log
        epoch_loss = None
        for log in reversed(state.log_history):
            if 'loss' in log:
                epoch_loss = log['loss']
                break
        self.epoch_losses.append(epoch_loss)
        current_epoch = int(state.epoch)
        if current_epoch % self.print_every_epoch == 0:
            if epoch_loss is not None:
                print(f"[Epoch {current_epoch}] Loss: {epoch_loss:.4f}")
            else:
                print(f"[Epoch {current_epoch}] Loss: None")
        # early stop 条件
        if epoch_loss is not None and epoch_loss < self.min_loss:
            print(f"Early stopping at epoch {current_epoch}: loss={epoch_loss:.4f} < {self.min_loss}")
            control.should_training_stop = True
        # (可选) patience 早停
        if epoch_loss is not None and epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping (patience) at epoch {current_epoch}")
                control.should_training_stop = True
# ----------------------------
# data utils
# ----------------------------
def preprocess_function(examples, tokenizer: AutoTokenizer, max_length: int = 256):
    input_ids_list, labels_list, attention_mask_list = [], [], []

    for prompt, target in zip(examples["prompt"], examples["target"]):
        # standard causal-lm: prompt + target, mask prompt labels with -100
        full_text = prompt + " " + target

        full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len = len(prompt_enc["input_ids"])

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }

import random
from typing import List

def augment_prompt(model, tok, base_prompt: str) -> List[str]:
    # 定义一个全局缓存变量
    global PREFIX_CACHE

    # 如果缓存为空，则生成新的前缀
    if PREFIX_CACHE is None:
        random.seed(42)  # 设置Python的random种子
        torch.manual_seed(42)  # 设置PyTorch的随机种子
        torch.cuda.manual_seed_all(42)  # 如果使用GPU，确保GPU上的随机种子也被固定

        # 使用generate_fast生成固定的前缀
        PREFIX_CACHE = [
            f.replace("{", " ").replace("}", " ") + "."
            for f in generate_fast(
                model,
                tok,
                ["The", "Therefore", "Because", "I", "You"],
                n_gen_per_prompt=1,  # 每个词生成一个前缀
                max_out_len=10  # 每个前缀的长度
            )
        ]
        print(f"Cached prefixes: {PREFIX_CACHE}")
    
    # 为每个前缀和base_prompt构建最终的提示
    return [f"{prefix} {base_prompt}" for prefix in PREFIX_CACHE]



def make_collate_fn(tokenizer: AutoTokenizer):
    def collate_fn(batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return collate_fn


# ----------------------------
# main: basic LoRA
# ----------------------------
def apply_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: LoRAHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Minimal LoRA training:
    - build Dataset from requests
    - inject PEFT LoRA into specified modules
    - train with HF Trainer
    - merge LoRA into base weights and unload PEFT wrapper
    """

    # if copy:
    #     model = deepcopy(model)

    # 1) Build train_data
    train_data = []
    for req in requests:
        prompt = req["prompt"].format(req["subject"])
        target = req["target_new"]["str"].lstrip()
        base_prompt = req["prompt"].format(req["subject"])
        train_data.append({"prompt": prompt, "target": target})
        # augmented_prompts = augment_prompt(model,tok,base_prompt)  
        # for aug_prompt in augmented_prompts:
        #     train_data.append({"prompt": aug_prompt, "target": target})
    # if len(requests) < 200: 
    #     train_data = train_data * 5    
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tok, "max_length": getattr(hparams, "max_length", 256)},
    )

    # 2) Decide target modules (exact module names)
    # Example: rewrite_module_tmp = "model.layers.{}.mlp.down_proj"
    specific_modules = [
        hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers
    ]

    # 3) Inject LoRA
    lora_config = LoraConfig(
        r=hparams.lora_r,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=specific_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4) Trainer
    training_args = TrainingArguments(
        output_dir=getattr(hparams, "output_dir", "./lora_output"),
        num_train_epochs=hparams.lora_num_epochs,
        per_device_train_batch_size=hparams.lora_batch_size,
        learning_rate=hparams.lora_learning_rate,
        logging_strategy="epoch",
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        optim="adamw_torch",
    )
    
    from torch.utils.data import SequentialSampler
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tok,
    data_collator=make_collate_fn(tok),
    callbacks=[EpochLoggerAndEarlyStop(print_every_epoch=10, min_loss=0.02, patience=99999)],
)

    # 强制 Trainer 使用顺序采样器
    # 这会覆盖 Trainer 默认生成的随机采样器
    # trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
    #     trainer.train_dataset,
    #     batch_size=training_args.train_batch_size,
    #     sampler=SequentialSampler(trainer.train_dataset), # 按索引顺序读取
    #     collate_fn=trainer.data_collator,
    #     drop_last=training_args.dataloader_drop_last,
    #     num_workers=training_args.dataloader_num_workers,
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tok,
        data_collator=make_collate_fn(tok),
        callbacks=[EpochLoggerAndEarlyStop(print_every_epoch=10, min_loss=0.02, patience=99999)],
    )

    print("Start LoRA training...")
    trainer.train()
    print("LoRA training done.")

    # 5) Merge LoRA into base model (important if you want a plain HF model afterwards)
    model.merge_and_unload()

    # (Optional) return_orig_weights kept for interface compatibility; minimal version returns empty dict
    weights_copy: Dict[str, Any] = {}
    return model, weights_copy
