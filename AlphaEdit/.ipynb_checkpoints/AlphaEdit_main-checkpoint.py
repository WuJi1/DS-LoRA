# import os
# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple
# import csv
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from rome.layer_stats import layer_stats
# from util import nethook
# from util.generate import generate_fast
# from util.globals import *

# from .compute_ks import compute_ks
# from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
# from .AlphaEdit_hparams import AlphaEditHyperParams
# # Cache variable(s)
# CONTEXT_TEMPLATES_CACHE = None
# COV_CACHE = {}

# def apply_AlphaEdit_to_model(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: List[Dict],
#     hparams: AlphaEditHyperParams,
#     cache_template: Optional[str] = None,
#     cache_c = None,
#     P = None,
# ) -> Dict[str, Tuple[torch.Tensor]]:
#     """
#     Executes the MEMIT update algorithm for the specified update at the specified layer
#     Invariant: model at beginning of function == model at end of function
#     """

#     # Update target and print info
#     requests = deepcopy(requests)
#     for i, request in enumerate(requests):
#         if request["target_new"]["str"][0] != " ":
#             # Space required for correct tokenization
#             requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
#     for request in requests[:10]:
#         print(
#             f"MEMIT request sample: "
#             f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
#         )

#     # Retrieve weights that user desires to change
#     weights = {
#         f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
#             model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
#         )
#         for layer in hparams.layers
#     }
#     # Compute z for final layer
#     context_templates = get_context_templates(model, tok)
#     z_layer = hparams.layers[-1]
#     z_list = []

#     for request in requests:
#         # Retrieve k/v pair if already stored in cache
#         cache_fname = (
#             Path(
#                 str(cache_template).format(
#                     z_layer, hparams.clamp_norm_factor, request["case_id"]
#                 )
#             )
#             if cache_template is not None
#             else None
#         )
#         data_loaded = False
#         if (
#             cache_fname is not None  # Require cache template
#             and cache_fname.exists()  # Cache file must exist
#         ):
#             try:
#                 data = np.load(cache_fname)
#                 z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
#                 data_loaded = True
#             except Exception as e:
#                 print(f"Error reading cache file due to {e}. Recomputing...")

#         # Compute k/v pair if not loaded from cache
#         if not data_loaded:
#             cur_z = compute_z(
#                 model,
#                 tok,
#                 request,
#                 hparams,
#                 z_layer,
#                 context_templates,
#             )

#             z_list.append(cur_z)

#             if cache_fname is not None:
#                 cache_fname.parent.mkdir(exist_ok=True, parents=True)
#                 np.savez(
#                     cache_fname,
#                     **{
#                         "v_star": cur_z.detach().cpu().numpy(),
#                     },
#                 )
#                 print(f"Cached k/v pair at {cache_fname}")
#     zs = torch.stack(z_list, dim=1)

#     for i, layer in enumerate(hparams.layers):
#         print(f"\n\nLAYER {layer}\n")

#         # Get current model activations
#         layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
#         print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

#         # Compute residual error
#         cur_zs = get_module_input_output_at_words(
#             model,
#             tok,
#             z_layer,
#             context_templates=[request["prompt"] for request in requests],
#             words=[request["subject"] for request in requests],
#             module_template=hparams.layer_module_tmp,
#             fact_token_strategy=hparams.fact_token,
#         )[1].T
#         targets = zs - cur_zs
#         print("z error", torch.linalg.norm(targets, dim=0).mean())

#         repeat_factor = (layer_ks.size(1) // targets.size(1))
#         targets = targets.repeat_interleave(repeat_factor, dim=1)
#         resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
#         upd_matrix = torch.linalg.solve(
#                 P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
#         )
#         # Adjust update matrix shape
#         weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
#         upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
#         print("orig norm", torch.linalg.norm(weights[weight_name]))
#         print("upd norm", torch.linalg.norm(upd_matrix))
#         with torch.no_grad():
#             weights[weight_name][...] = weights[weight_name] + upd_matrix
#         # Clear GPU memory
#         #del U,S,cov
#         for x in [layer_ks, cur_zs, targets, upd_matrix]:
#             x.cpu()
#             del x
#         torch.cuda.empty_cache()
#     for i, layer in enumerate(hparams.layers):
#         layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
#         cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

#     print(f"Deltas successfully computed for {list(weights.keys())}")
#     return model, cache_c


# def get_cov(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     layer_name: str,
#     mom2_dataset: str,
#     mom2_n_samples: str,
#     mom2_dtype: str,
#     inv: bool = False,
#     force_recompute: bool = False,
# ) -> torch.Tensor:
#     """
#     Retrieves covariance statistics, then computes the algebraic inverse.
#     Caches result for future use.
#     """

#     model_name = model.config._name_or_path.replace("/", "_")
#     key = (model_name, layer_name)

#     print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
#     if key not in COV_CACHE or force_recompute:
#         stat = layer_stats(
#             model,
#             tok,
#             layer_name,
#             STATS_DIR,
#             mom2_dataset,
#             to_collect=["mom2"],
#             sample_size=mom2_n_samples,
#             precision=mom2_dtype,
#             force_recompute=force_recompute,
#         )
#         COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

#     return (
#         torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
#     )


# def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
#     """
#     GPT-2 and GPT-J have transposed weight representations.
#     Returns a matrix that matches the desired shape, else raises a ValueError
#     """

#     if matrix.shape == shape:
#         return matrix
#     elif matrix.T.shape == shape:
#         return matrix.T
#     else:
#         raise ValueError(
#             "Update matrix computed by MEMIT does not match original weight shape. "
#             "Check for bugs in the code?"
#         )


# def get_context_templates(model, tok):
#     global CONTEXT_TEMPLATES_CACHE

#     if CONTEXT_TEMPLATES_CACHE is None:
#         CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
#             [
#                 f.replace("{", " ").replace("}", " ") + ". {}"
#                 for f in generate_fast(
#                     model,
#                     tok,
#                     ["The", "Therefore", "Because", "I", "You"],
#                     n_gen_per_prompt=n_gen // 5,
#                     max_out_len=length,
#                 )
#             ]
#             for length, n_gen in [(10, 5)]  # Be careful about changing this.
#         ]
#         print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

#     return CONTEXT_TEMPLATES_CACHE
import os
import gc
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams


# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def _clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
) -> Tuple[AutoModelForCausalLM, Any]:
    """
    Executes AlphaEdit update.

    Multi-GPU and large-edit safe version:
    1. Avoids hard-coded .cuda() / .to("cuda") in the update path.
    2. Keeps large intermediate tensors such as zs/cur_zs on CPU when possible.
    3. Moves only the tensors required for the current layer update to the device
       of that layer's rewrite weight.
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
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

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
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

                # Keep cached z on CPU.
                # This avoids holding [hidden_dim, num_edits] on GPU for 15000/20000 edits.
                z_list.append(torch.from_numpy(data["v_star"]).float().cpu())
                data_loaded = True

            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            # Move cur_z to CPU immediately after computation.
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
            _clear_cuda_cache()

    # zs: [hidden_dim, num_edits], kept on CPU.
    zs = torch.stack(z_list, dim=1).float().cpu()
    del z_list
    _clear_cuda_cache()

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        rewrite_weight = weights[weight_name]
        layer_device = rewrite_weight.device
        layer_dtype = rewrite_weight.dtype

        # Get current model activations.
        # compute_ks may return CPU or GPU tensor depending on your compute_ks/repr_tools implementation.
        # We only move the final K matrix to the current layer's device.
        layer_ks = compute_ks(
            model,
            tok,
            requests,
            hparams,
            layer,
            context_templates,
        ).T

        layer_ks = layer_ks.to(device=layer_device, dtype=torch.float64)

        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute current z.
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # Important:
        # After the OOM fix in rome/repr_tools.py, cur_zs may be a CPU tensor.
        # Therefore compute targets on CPU first to avoid cuda/cpu mismatch and GPU memory pressure.
        cur_zs = cur_zs.detach().float().cpu()
        targets = zs - cur_zs

        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Move targets only when they are needed for GPU matrix operations.
        targets = targets.to(device=layer_device, dtype=torch.float64)
        resid = targets / (len(hparams.layers) - i)

        # P and cache_c may be stored on CPU. Move only the current slice to the current layer device.
        P_i = P[i, :, :].to(device=layer_device, dtype=torch.float64)
        cache_ci = cache_c[i, :, :].to(device=layer_device, dtype=torch.float64)

        eye = torch.eye(
            layer_ks.shape[0],
            dtype=torch.float64,
            device=layer_device,
        )

        lhs = (
            P_i @ (layer_ks @ layer_ks.T + cache_ci)
            + hparams.L2 * eye
        )
        rhs = P_i @ layer_ks @ resid.T

        upd_matrix = torch.linalg.solve(lhs, rhs)

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, rewrite_weight.shape)

        print("orig norm", torch.linalg.norm(rewrite_weight))
        print("upd norm", torch.linalg.norm(upd_matrix))

        with torch.no_grad():
            rewrite_weight[...] = rewrite_weight + upd_matrix.to(
                device=layer_device,
                dtype=layer_dtype,
            )

        # Clear GPU memory aggressively between layers
        del layer_ks
        del cur_zs
        del targets
        del resid
        del P_i
        del cache_ci
        del eye
        del lhs
        del rhs
        del upd_matrix
        _clear_cuda_cache()

    # Update cache_c on CPU.
    # Do not keep layer_ks on GPU here because cache_c is CPU-side accumulated statistics.
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(
            model,
            tok,
            requests,
            hparams,
            layer,
            context_templates,
        ).T

        layer_ks = layer_ks.detach().float().cpu()
        cache_c[i, :, :] += layer_ks @ layer_ks.T

        del layer_ks
        _clear_cuda_cache()

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


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

    Multi-GPU safe behavior:
        Keep covariance cache on CPU.
        Move to the requested device only when used.
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
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
