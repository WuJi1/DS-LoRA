import os
import gc
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams


# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_seq_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    cache_c=None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas, cache_c = execute_memit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        cache_c=cache_c,
    )

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            w = nethook.get_parameter(model, w_name)

            # 多卡关键：不要 .to("cuda")，而是跟随当前权重所在 GPU
            # 保留原始 MEMIT 的 double matmul 逻辑，最后再转回权重 dtype
            key_mat = key_mat.to(device=w.device, dtype=torch.float64)
            val_mat = val_mat.to(device=w.device, dtype=torch.float64)

            upd_matrix = key_mat @ val_mat.T
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.to(device=w.device, dtype=w.dtype)

            del key_mat, val_mat, upd_matrix
            torch.cuda.empty_cache()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, cache_c


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer.
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> "
            f"[{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model,
            f"{hparams.rewrite_module_tmp.format(layer)}.weight",
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer,
                    hparams.clamp_norm_factor,
                    request["case_id"],
                )
            )
            if cache_template is not None
            else None
        )

        data_loaded = False

        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)

                # 关键：z_star 先放 CPU，避免 20000 edits 下占 GPU
                z_list.append(torch.from_numpy(data["v_star"]).float().cpu())
                data_loaded = True

            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            # 关键：compute_z 返回后立刻搬 CPU
            z_list.append(cur_z.detach().float().cpu())

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")

            del cur_z
            torch.cuda.empty_cache()

    # zs: [hidden_dim, num_edits]，保持 CPU
    zs = torch.stack(z_list, dim=1).float().cpu()
    del z_list
    gc.collect()
    torch.cuda.empty_cache()

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        rewrite_weight = weights[weight_name]
        layer_device = rewrite_weight.device
        layer_dtype = rewrite_weight.dtype

        # Get current model activations
        # compute_ks 可能返回 CPU，也可能返回 GPU；这里统一只把最终 K 搬到当前 layer 所在 GPU
        layer_ks = compute_ks(
            model,
            tok,
            requests,
            hparams,
            layer,
            context_templates,
        ).T

        layer_ks = layer_ks.to(device=layer_device, dtype=layer_dtype)

        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # 关键：这里先在 CPU 上做 residual，避免 zs(cuda) - cur_zs(cpu) 报错
        # 同时避免过早占用 GPU 显存
        cur_zs = cur_zs.detach().float().cpu()
        targets = zs - cur_zs

        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # 只有真正进入 GPU 矩阵计算前，才把 targets 搬到当前 layer 所在 GPU
        targets = targets.to(device=layer_device, dtype=layer_dtype)

        # Load covariance matrix
        force_recompute = False
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            device=layer_device,
            dtype=layer_dtype,
        )

        cache_ci = cache_c[i, :, :].to(device=layer_device, dtype=layer_dtype)

        # Compute update in double precision
        layer_ks = layer_ks.double()
        targets = targets.double()
        cov = cov.double()
        cache_ci = cache_ci.double()

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov
            + cache_ci
            + layer_ks @ layer_ks.T,
            layer_ks,
        )

        resid = targets / (len(hparams.layers) - i)
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = (
                weights_copy[weight_name].to(
                    device=layer_device,
                    dtype=weights[weight_name].dtype,
                )
                + upd_matrix.to(
                    device=layer_device,
                    dtype=weights[weight_name].dtype,
                )
            )

            # 保存到 CPU，避免 deltas 长期占 GPU
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        del layer_ks
        del cur_zs
        del targets
        del cov
        del cache_ci
        del adj_k
        del resid
        del upd_matrix

        gc.collect()
        torch.cuda.empty_cache()

    # Update cache_c on CPU
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(
            model,
            tok,
            requests,
            hparams,
            layer,
            context_templates,
        ).T

        # 这里 cache_c 本来就是 CPU，直接 CPU 上更新，避免再次占 GPU
        layer_ks = layer_ks.detach().float().cpu()
        cache_c[i, :, :] += layer_ks @ layer_ks.T

        del layer_ks
        gc.collect()
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k].to(device=v.device, dtype=v.dtype)

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.

    多卡修改：
        COV_CACHE 始终保存在 CPU；
        需要参与当前 layer 计算时，再搬到当前 layer 所在 GPU。
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")

    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )

        # 关键：缓存只放 CPU
        COV_CACHE[key] = stat.mom2.moment().float().cpu()

    cov = COV_CACHE[key]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = cov.dtype

    cov = cov.to(device=device, dtype=dtype)

    return torch.inverse(cov) if inv else cov


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError.
    """

    if matrix.shape == shape:
        return matrix

    elif matrix.T.shape == shape:
        return matrix.T

    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE