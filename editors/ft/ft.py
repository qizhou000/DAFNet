from ..editor import BaseEditor, EditorConfig
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import torch
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils import nethook


@dataclass
class FTConfig(EditorConfig):
    edit_model_name:str
    
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Defaults
    batch_size: int = 128
    max_length: int = 30



class FT(BaseEditor):

    def __init__(self, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config:FTConfig,
        device = 'cuda',
        verbose = False
    ):
        super().__init__(model, tokenizer, device)
        self.cfg = config
        self.verbose = verbose
        self.original_w = {
            n: p.clone()
            for n, p in self.model.named_parameters()
            for layer in self.cfg.layers
            if self.cfg.rewrite_module_tmp.format(layer) in n
        }

        # for l in self.cfg.layers:
        #     w_name = f"{self.cfg.rewrite_module_tmp.format(l)}"
        #     self.original_w.append(nethook.get_parameter(self.model, w_name).clone())
 
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'ft', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True

    def restore_to_original_model(self):
        # for n, p in self.model.named_parameters():
        #     for layer in self.cfg.layers:
        #         if self.cfg.rewrite_module_tmp.format(layer) in n:
        #             self.model 
        self.model.load_state_dict(self.original_w, strict = False)

        # for i, l in enumerate(self.cfg.layers):
        #     w_name = f"{self.cfg.rewrite_module_tmp.format(l)}"
        #     w = nethook.get_parameter(self.model, w_name)
        #     with torch.no_grad():
        #         w *= 0
        #         w += self.original_w[i]
        #     if self.verbose:
        #         print('return:', w_name)

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
        deltas = self.execute_ft(requests)
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                if self.verbose:
                    print('edit:',w_name)
                w = nethook.get_parameter(self.model, w_name) 
                w[...] += upd_matrix
 

    ######################################################################################
    ############################# Fine-Tune Utilization ##################################
    ######################################################################################
    def execute_ft(self, requests: List[Dict]) -> Dict[str, Tuple[torch.Tensor]]:
        """
        Executes the FT update algorithm for the specified update at the specified layer
        Invariant: model at beginning of function == model at end of function
        """
        device = torch.device(self.device)
        # Update target and print info
        requests = deepcopy(requests)
        for request in requests:
            if request["target_new"] != " ":
                # Space required for correct tokenization
                request["target_new"] = " " + request["target_new"]
            if self.verbose:
                print(
                    f"Executing FT algo for: "
                    f"[{request['prompt']}] -> [{request['target_new']}]"
                )
        
        # Retrieve weights that user desires to change
        weights = {
            n: p
            for n, p in self.model.named_parameters()
            for layer in self.cfg.layers
            if self.cfg.rewrite_module_tmp.format(layer) in n
        }
        
        # Save old weights for future restoration
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}
        if self.verbose:
            print(f"Weights to be updated: {list(weights.keys())}")

        # Define inputs
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]
        
        # Configure optimizer / gradients
        opt = torch.optim.Adam(
            [v for _, v in weights.items()],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        for name, w in self.model.named_parameters():
            w.requires_grad = name in weights

        # Update loop: intervene at layers simultaneously
        loss_meter = AverageMeter()
        for it in range(self.cfg.num_steps):
            if self.verbose:
                print(20 * "=")
                print(f"Epoch: {it}")
                print(20 * "=")
            loss_meter.reset()

            for txt, tgt in zip(
                chunks(texts, self.cfg.batch_size), chunks(targets, self.cfg.batch_size)
            ):
                inputs = self.tokenizer(txt, return_tensors="pt", padding=True).to(device)
                target_ids = self.tokenizer(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                    device
                )
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                loss_mask = target_ids != self.tokenizer.unk_token_id
                opt.zero_grad()
                bs = inputs["input_ids"].shape[0]
                if 't5' in self.cfg.edit_model_name.lower():
                    inputs['labels'] = target_ids
                    logits = self.model(**inputs).logits
                    unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)

                    mask = inputs['labels'] != -100
                    n_tokens = mask.float().sum()
                    avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                    nll = -avg_log_prob
                    loss = nll
                else:
                    probs = torch.nn.functional.log_softmax(
                        self.model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                    )
                    loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                        1
                    ) / loss_mask.sum(1)
                    loss = loss.mean()
                if self.verbose:
                    print(f"Batch loss {loss.item()}")
                loss_meter.update(loss.item(), n=bs)

                if loss.item() >= 1e-2:
                    loss.backward()
                    opt.step()

                if type(self.cfg.norm_constraint) is float:
                    eps = self.cfg.norm_constraint
                    with torch.no_grad():
                        for k, v in weights.items():
                            v[...] = torch.clamp(
                                v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                            )
            if self.verbose:
                print(f"Total loss {loss_meter.avg}")

            if loss_meter.avg < 1e-2:
                break

        deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

        # Restore state of original model
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]

        if self.verbose:
            print(f"Deltas successfully computed for {list(weights.keys())}")

        return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
