from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams

from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class LoRAHyperParams(HyperParams):
    # ========= 模型与插入位置 =========
    model_name: str = "huggyllama/llama-7b"
    layers: List[int] = field(default_factory=lambda: [20, 21, 22, 23])
    layer_selection: Literal["all", "random"] = "all"
    fact_token: Literal["last", "subject_first", "subject_last", "subject_first_after_last"] = "last"

    # ========= 模块模板 =========
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    mlp_module_tmp: str = "model.layers.{}.mlp"
    attn_module_tmp: str = "model.layers.{}.self_attn"
    ln_f_module: str = "model.norm"
    lm_head_module: str = "lm_head"
    lora_type: str = "lora"
    # ========= 统计与协方差矩阵参数（可选，用于对比等） =========
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 2000
    mom2_dtype: str = "float32"
    mom2_adjustment: bool = False
    mom2_update_weight: float = 1.0

    # ========= LoRA 训练参数 =========
    nullspace_threshold: float = 0.05
    total_layers: int = 27
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    lora_learning_rate: float = 1e-4
    lora_batch_size: int = 16
    lora_num_epochs: int = 40
    pre_cov: bool = True

