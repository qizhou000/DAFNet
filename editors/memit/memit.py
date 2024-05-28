from ..editor import BaseEditor, EditorConfig
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import torch
from dataclasses import dataclass
import numpy as np
from typing import List, Literal
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ..utils import nethook
 
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx


@dataclass
class MEMITConfig(EditorConfig):
    edit_model_name: str
    # Method
    edit_layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str


class MEMIT(BaseEditor):

    def __init__(self, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config:MEMITConfig,
        stats_dir: str, 
        comput_stat_data_path = None,
        device = 'cuda',
        verbose = False
    ):
        '''
        stats_dir: The dir of constant matrix C.
        '''
        super().__init__(model, tokenizer, device)
        self.cfg = config
        self.stats_dir = stats_dir
        self.comput_stat_data_path = comput_stat_data_path
        self.verbose = verbose
        # Cache variable(s)
        self.context_templates = get_context_templates(self.model, self.tokenizer)
        self.cov_cache = {}
        # cache original weights
        self.original_ws = {}
        for layer in self.cfg.edit_layers:
            w_name = f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            self.original_ws[w_name] = nethook.get_parameter(self.model, w_name).clone()

    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'memit', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True

    def edit_one_piece(self, request: Dict) -> None:
        """
        request = {'prompt':str, 'subject':str, 'target_new':str}
        """
        self.edit_batch([request])


    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'prompt':str, 'subject':str, 'target_new':str}
            {'prompt':str, 'subject':str, 'target_new':str}, ...
        ]
        '''
        deltas = self.execute_memit(requests)
        with torch.no_grad():
            for w_name, (key_mat, val_mat) in deltas.items():
                key_mat, val_mat = key_mat.to(self.device), val_mat.to(self.device)
                upd_matrix = key_mat @ val_mat.T
                w = nethook.get_parameter(self.model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                w[...] += upd_matrix.float()
        if self.verbose:
            print(f"New weights successfully inserted into {list(deltas.keys())}")


    def restore_to_original_model(self):
        for w_name, ori_w in self.original_ws.items():
            w = nethook.get_parameter(self.model, w_name)
            with torch.no_grad():
                w *= 0
                w += ori_w

    ######################################################################################
    ################################## MEMIT Utilization #################################
    ######################################################################################
    def execute_memit(self, requests: List[Dict]) -> Dict[str, Tuple[torch.Tensor]]:

        deltas = {}

        # Update target and print info
        requests = deepcopy(requests)
        for i, request in enumerate(requests):
            if ('{}' not in request['prompt']):
                assert request['subject'] in request['prompt'] or \
                    print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

                requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
            if request["target_new"][0] != " ":
                request["target_new"] = " " + request["target_new"]
                
        # Retrieve weights that user desires to change
        weights = {
            f"{self.cfg.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                self.model, f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in self.cfg.edit_layers
        }
        # Save old weights for future restoration
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}

        # Compute z for final layer
        # context_templates = get_context_templates(self.model, self.tokenizer)
        z_layer = self.cfg.edit_layers[-1]
        z_list = []

        for request in requests:
            # Compute k/v pair if not loaded from cache
            cur_z = compute_z(
                self.model,
                self.tokenizer,
                request,
                self.cfg, 
                z_layer,
                self.context_templates,
                self.device,
                self.verbose
            )
            z_list.append(cur_z)

        zs = torch.stack(z_list, dim=1)

        # Insert
        for i, layer in enumerate(self.cfg.edit_layers):
            if self.verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            layer_ks = compute_ks(self.model, self.tokenizer, requests, self.cfg, layer, self.context_templates).T
            if self.verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                self.model,
                self.tokenizer,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=self.cfg.layer_module_tmp,
                fact_token_strategy=self.cfg.fact_token,
                track='out'
            ).T
            targets = zs - cur_zs
            if self.verbose:
                print("z error", torch.linalg.norm(targets, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)

            # Load covariance matrix
            cov = self.get_cov(self.cfg.rewrite_module_tmp.format(layer),
                               inv = False, force_recompute = False)

            # Compute update in double precision
            layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )

            adj_k = torch.linalg.solve(
                self.cfg.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                layer_ks,
            )
            resid = targets / (len(self.cfg.edit_layers) - i)  # Distribute residual across layers
            upd_matrix = resid @ adj_k.T

            # Adjust update matrix shape
            weight_name = f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            if self.verbose:
                print("orig norm", torch.linalg.norm(weights[weight_name]))
                print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

        # Restore state of original model
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]

        if self.verbose:
            print(f"Deltas successfully computed for {list(weights.keys())}")

        return deltas


    def get_cov(self, layer_name: str, inv: bool = False, 
                force_recompute: bool = False) -> torch.Tensor:
        """
        Retrieves covariance statistics, then computes the algebraic inverse.
        Caches result for future use.
        """

        model_name = self.cfg.edit_model_name
        key = (model_name, layer_name)
        if self.verbose:
            print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
        if key not in self.cov_cache or force_recompute:
            stat = layer_stats(
                self.model,
                self.tokenizer,
                layer_name,
                self.stats_dir,
                self.cfg.mom2_dataset,
                to_collect=["mom2"],
                model_name=self.cfg.edit_model_name,
                sample_size=self.cfg.mom2_n_samples,
                precision=self.cfg.mom2_dtype,
                device=self.device,
                data_path = self.comput_stat_data_path
            )
            self.cov_cache[key] = stat.mom2.moment().float().to("cpu")

        return (
            torch.inverse(self.cov_cache[key].to(self.device)) if inv else 
            self.cov_cache[key].to(self.device)
        )
    

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
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
    from ..utils.generate import generate_fast
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
