from ..editor import BaseEditor, EditorConfig
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import torch
from dataclasses import dataclass
from ..utils import nethook
from . import repr_tools
from copy import deepcopy
from .layer_stats import layer_stats
import numpy as np


@dataclass
class ROMEConfig(EditorConfig):
    # Method
    edit_model_name: str
    edit_layer: int
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]
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
    


class ROME(BaseEditor):

    def __init__(self, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config:ROMEConfig,
        stats_dir: str = 'data/rome-memit-stats', 
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
        self.context_templates = get_context_templates(model, tokenizer, self.cfg.context_template_length_params)
        self.inv_mom2_cache = {}
        w_name = f"{self.cfg.rewrite_module_tmp.format(self.cfg.edit_layer)}.weight"
        self.original_w = nethook.get_parameter(self.model, w_name).clone()
        self.verbose = verbose
        if 'llama' in self.cfg.edit_model_name:
            self.model_hidden_size = model.config.hidden_size
        else:
            self.model_hidden_size = model.config.n_embd

    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'rome', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def edit_one_piece(self, request: Dict) -> None:
        """
        request = {'prompt':str, 'subject':str, 'target_new':str}
        """
        deltas = self.execute_rome(request)
        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(self.model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                w[...] += upd_matrix
        if self.verbose:
            print(f"New weights successfully inserted into {list(deltas.keys())}")


    def edit_batch(self):
        raise 'ROME can not batch edit.'

    def restore_to_original_model(self):
        w_name = f"{self.cfg.rewrite_module_tmp.format(self.cfg.edit_layer)}.weight"
        w = nethook.get_parameter(self.model, w_name)
        with torch.no_grad():
            w *= 0
            w += self.original_w
        
    ######################################################################################
    ################################## ROME Utilization ##################################
    ######################################################################################


    def execute_rome(self, request: Dict) -> None:
        request = deepcopy(request)
        if request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]
        if('{}' not in request['prompt']):
            assert request['subject'] in request['prompt'] or \
                print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
            request['prompt'] = request['prompt'].replace(request['subject'], '{}')
        if self.verbose:
            print(
                f"Executing ROME algorithm for the update: "
                f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
            )
        deltas = {} 
        left_vector: torch.Tensor = self.compute_u(request)
        right_vector: torch.Tensor = self.compute_v(request, left_vector)
        with torch.no_grad():
            weight_name = f"{self.cfg.rewrite_module_tmp.format(self.cfg.edit_layer)}.weight"
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )
        return deltas


    def compute_u(self, request: Dict) -> torch.Tensor:
        # Compute projection token
        word_repr_args = dict(
            model=self.model,
            tok=self.tokenizer,
            layer=self.cfg.edit_layer,
            module_template=self.cfg.rewrite_module_tmp,
            track="in",
        )
        if "subject_" in self.cfg.fact_token and self.cfg.fact_token.index("subject_") == 0:
            word = request["subject"]
            # print(f"Selected u projection object {word}")
            cur_repr = repr_tools.get_reprs_at_word_tokens(
                context_templates=[
                    templ.format(request["prompt"]) for templ in self.context_templates
                ],
                words=[word for _ in range(len(self.context_templates))],
                subtoken=self.cfg.fact_token[len("subject_") :],
                **word_repr_args,
            ).mean(0)
        elif self.cfg.fact_token == "last":
            cur_repr = repr_tools.get_reprs_at_idxs(
                contexts=[
                    templ.format(request["prompt"].format(request["subject"]))
                    for templ in self.context_templates
                ],
                idxs=[[-1] for _ in range(len(self.context_templates))],
                **word_repr_args,
            ).mean(0)
            # print("Selected u projection token with last token")
        else:
            raise ValueError(f"fact_token={self.cfg.fact_token} not recognized")
        # Apply inverse second moment adjustment
        u = cur_repr
        if self.cfg.mom2_adjustment: 
            u = self.get_inv_cov() @ u.unsqueeze(1)
            u = u.squeeze()
        return u / u.norm()

 
    def get_inv_cov(self) -> torch.Tensor:
        """
        Retrieves covariance statistics, then computes the algebraic inverse.
        Caches result for future use.
        """
        layer_name = self.cfg.rewrite_module_tmp.format(self.cfg.edit_layer)
        key = (self.cfg.edit_model_name, layer_name)
        if key not in self.inv_mom2_cache:
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
            self.inv_mom2_cache[key] = torch.inverse(
                stat.mom2.moment().to(self.device)
            ).float()  # Cast back to float32
        return self.inv_mom2_cache[key]



    def compute_v(self, request: Dict, left_vector: torch.Tensor) -> torch.Tensor:
        # Tokenize target into list of int token IDs
        target_ids = self.tokenizer(request["target_new"], 
            return_tensors="pt").to(self.device)["input_ids"][0]

        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + self.tokenizer.decode(target_ids[:-1])
            for context in self.context_templates
        ], ["{} is a"]
        all_prompts = rewriting_prompts + kl_prompts

        input_tok = self.tokenizer(
            [prompt.format(request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device=self.device).repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

        # Compute indices of the tokens where the fact is looked up
        lookup_idxs = [
            self.find_fact_lookup_idx(
                prompt, request["subject"], self.cfg.fact_token, verbose=(i == 0)
            )
            for i, prompt in enumerate(all_prompts)
        ]

        # Finalize rewrite and loss layers
        loss_layer = max(self.cfg.v_loss_layer, self.cfg.edit_layer)
        # print(f"Rewrite layer is {self.cfg.edit_layer}")
        # print(f"Tying optimization objective to {loss_layer}")

        # Set up an optimization over a latent vector that, when output at the
        # rewrite layer, i.e. hypothesized fact lookup location, will induce the
        # target token to be predicted at the final layer.
        delta = torch.zeros((self.model_hidden_size,), requires_grad=True, device=self.device)
        target_init, kl_distr_init = None, None

        # Inserts new "delta" variable at the appropriate part of the computation
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init

            if cur_layer == self.cfg.mlp_module_tmp.format(self.cfg.edit_layer):
                # Store initial value of the vector of interest
                if target_init is None:
                    # print("Recording initial value of v*")
                    # Initial value is recorded for the clean sentence
                    target_init = cur_out[0, lookup_idxs[0]].detach().clone()

                for i, idx in enumerate(lookup_idxs):
                    cur_out[i, idx, :] += delta

            return cur_out

        # Optimizer
        opt = torch.optim.Adam([delta], lr=self.cfg.v_lr)
        nethook.set_requires_grad(False, self.model)

        # Execute optimization
        for it in range(self.cfg.v_num_grad_steps):
            opt.zero_grad()

            # Forward propagation
            with nethook.TraceDict(
                module=self.model,
                layers=[
                    self.cfg.layer_module_tmp.format(loss_layer),
                    self.cfg.mlp_module_tmp.format(self.cfg.edit_layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = self.model(**input_tok).logits

                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = self.cfg.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = self.cfg.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + kl_loss + weight_decay
            if self.verbose:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                    f"avg prob of [{request['target_new']}] "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )

            if loss < 5e-2:
                break

            if it == self.cfg.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = self.cfg.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()
        target = target_init + delta

        # Retrieve cur_input, the current input to the 2nd MLP layer, and
        # cur_output, the original output of the 2nd MLP layer.
        cur_input, cur_output = self.get_module_input_output_at_word(
            context_template=request["prompt"],
            word=request["subject"],
            module_template=self.cfg.rewrite_module_tmp,
            fact_token_strategy=self.cfg.fact_token,
        )

        # Solving the linear system to compute the right vector
        right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
        # print(f"Delta norm: {(target - cur_output).norm().item()}")
        # print(
        #     f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
        # )
        # print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
        # print(f"Right vector norm: {right_vector.norm()}")

        return right_vector


    def get_module_input_output_at_word(self, context_template: str, 
        word: str, module_template: str, fact_token_strategy: str) -> Tuple[torch.Tensor]:
        """
        Retrieves detached representations for a word at the input and
        output of a particular layer module.
        """
        word_repr_args = dict(
            model=self.model,
            tok=self.tokenizer,
            layer=self.cfg.edit_layer,
            module_template=module_template,
        )
        if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
            subtoken = fact_token_strategy[len("subject_") :]
            l_input, l_output = repr_tools.get_reprs_at_word_tokens(
                track="both",
                subtoken=subtoken,
                context_templates=[context_template],
                words=[word],
                **word_repr_args,
            )
        elif fact_token_strategy == "last":
            l_input, l_output = repr_tools.get_reprs_at_idxs(
                track="both",
                contexts=[context_template.format(word)],
                idxs=[[-1]],
                **word_repr_args,
            )
        else:
            raise ValueError(f"fact_token={fact_token_strategy} not recognized")

        l_input, l_output = l_input[0], l_output[0]
        return l_input.detach(), l_output.detach()


    def find_fact_lookup_idx(self, prompt: str, subject: str,
        fact_token_strategy: str, verbose=True) -> int:
        """
        Computes hypothesized fact lookup index given a sentence and subject.
        """
        ret = None
        if fact_token_strategy == "last":
            ret = -1
        elif (
            "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
        ):
            ret = repr_tools.get_words_idxs_in_templates(
                tok=self.tokenizer,
                context_templates=[prompt],
                words=[subject],
                subtoken=fact_token_strategy[len("subject_") :],
            )[0][0]
        else:
            raise ValueError(f"fact_token={fact_token_strategy} not recognized")

        sentence = prompt.format(subject)
        if verbose:
            pass
            # print(
            #     f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            #     self.tokenizer.decode(self.tokenizer(sentence)["input_ids"][ret]),
            # )

        return ret




def get_context_templates(model, tok, context_template_length):
    from ..utils.generate import generate_fast
    context_templates = ["{}"]
    for ctl in context_template_length:
        length, n_gen  = ctl 
        context_templates.extend([
            f.replace("{", " ").replace("}", " ") + ". {}"
            for f in generate_fast(
                model,
                tok,
                ["The", "Therefore", "Because", "I", "You"],
                n_gen_per_prompt=n_gen // 5,
                max_out_len=length)
        ])
    print(f"Cached context templates {context_templates}")
    return context_templates


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
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )
