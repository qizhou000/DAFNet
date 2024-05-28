import torch
import torch.nn as nn
import transformers
from ..editor import BaseEditor, EditorConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Union
from .aux_model import GradientTransformer, MultiGPUGradientTransformer
import json, yaml
from utils.data import prompts_target_to_x_y_mask
from torch.optim import Adam
from utils.data import ParallelDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from transformers.pytorch_utils import Conv1D
import numpy as np
from copy import deepcopy

@dataclass
class DAFNetConfig(EditorConfig):
    @dataclass
    class AuxModelConfig():
        layer_n: int 
        inner_att_dim: int
        ffn_dim: int
        outer_att_dim: int
        outer_att_head_n: int
        outer_score_att_dim: int
        outer_score_att_head_n: int
        norm_layer: str
    @dataclass
    class TrainingConfig():
        aux_model_lr: float
        init_edit_lr: float
        edit_lr_lr: float
        relia_lambda: float
        gen_lambda: float
        loc_lambda: float
        max_seq_modeling: int
        increase_seq_modeling_i: int
        increase_seq_modeling_pace: float
        ema_loss_lambda: float
        grad_clip: bool
        accum_n: int
        loss_batch_size: int
        loss_sample_max_count: int

    edit_model_name: str
    edit_modules: List[str]
    if_edit_bias: bool
    aux_model: AuxModelConfig
    training: TrainingConfig
    gradient_signal_batch_size: int

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['aux_model'] = self.AuxModelConfig(**data['aux_model'])
        data['training'] = self.TrainingConfig(**data['training'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise


class DAFNet(BaseEditor):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
        config: DAFNetConfig, device = 'cuda:0', device_gradient_signal = 'cuda:1', 
        devices_aux_models = ['cuda:0', 'cuda:1'], ckpt_path = None):
        '''
        device: the device of model that needs editing.
        device_gradient_signal: the device that gets gradient signal.
        devices_aux_models: the devices of auxiliary models.
        '''

        self.device_gradient_signal = device_gradient_signal
        self.model_gradient_signal = self.init_gradient_signal_model(model.to('cpu'), device_gradient_signal)
        super().__init__(model, tokenizer, device)

        self.cfg = config
        # initialize model modules that gets gradient signal
        self.modules_gradient_signal = self.init_get_modules_by_names(self.model_gradient_signal, self.cfg.edit_modules)
        self.gradient_signal_hooks = self.init_register_gradient_signal_hooks(self.modules_gradient_signal) 
        self.autograd_params = self.init_autograd_params(self.modules_gradient_signal) 
        # initialize modules that needs editing
        self.edit_modules = self.init_get_modules_by_names(self.model, self.cfg.edit_modules)
        self.edit_hooks = self.init_register_edit_hooks(self.edit_modules) 
        # initialize auxiliary models 
        self.aux_models, self.shape_to_em_ids, self.em_id2_aux_model_device = self.init_aux_models(self.edit_modules, self.cfg.aux_model, self.cfg.if_edit_bias, devices_aux_models)
        self.edit_reps = self.init_edit_representations(self.edit_modules)
        # initialize weights update learning rate 
        self.edit_lrs = self.init_edit_lrs(self.edit_modules, config.training.init_edit_lr, self.device)
        # cache original weights
        self.original_weights, self.original_biases = self.init_cache_edit_module_original_weights(self.edit_modules)
        # initialize if not train
        self.log_writer = None
        self.set_train(False)
        self.aux_module_parallel = False
        # load checkpoint
        if ckpt_path != None:
            self.load_ckpt(ckpt_path, False, True)

    ############################################################################
    #                     DAFNet initialization functions                         #
    ############################################################################
    def init_gradient_signal_model(self, original_model:nn.Module, device_gradient_signal):
        model_gradient_signal = deepcopy(original_model).to(device_gradient_signal)
        return model_gradient_signal

    def init_get_modules_by_names(self, model:nn.Module, edit_modules_names:List[str]) ->\
          List[Union[nn.Linear, Conv1D]]:
        edit_modules = []
        for module_name in edit_modules_names:
            module = find_module(model, module_name.split('.')) 
            assert isinstance(module, Conv1D) or \
                   isinstance(module, nn.Linear) # Only support Linear layer
            edit_modules.append(module) 
        return edit_modules

    def init_register_gradient_signal_hooks(self, modules_gradient_signal:List[Union[nn.Linear, Conv1D]]):
        def forward_x_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out] 
            if self.__get_gradient_signal_stage__:
                module.__x__.append(args[0].detach())
        def backward_delta_hook(module, grad_input, grad_output):
            # grad_input: tuple(input grad) [batch_size, max_length, dim_in]
            # grad_output: tuple(output grad) [batch_size, max_length, dim_out]
            if self.__get_gradient_signal_stage__:
                module.__delta__.append(grad_output[0].detach())
        self.__get_gradient_signal_stage__ = False
        hooks = []
        for mgs in modules_gradient_signal:
            x_handle = mgs.register_forward_hook(forward_x_hook)
            delta_handle = mgs.register_full_backward_hook(backward_delta_hook)
            mgs.__x__ = []
            mgs.__delta__ = []
            hooks.append((x_handle, delta_handle))
        return hooks

    def init_register_edit_hooks(self, edit_modules:List[Union[nn.Linear, Conv1D]]):
        def forward_edit_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            if self.__use_delta_weight__:
                if module.__delta_weight__ != None:
                    # Wx + b + W'x = (W + W')x + b
                    output = output + args[0] @ module.__delta_weight__ 
                if module.__delta_bias__ != None:
                    # Wx + b + W'x + b'
                    output = output + module.__delta_bias__ 
                return output
        self.__use_delta_weight__ = True
        hooks = []
        for em in edit_modules:
            edit_handle = em.register_forward_hook(forward_edit_hook)
            em.__delta_weight__ = None
            em.__delta_bias__ = None
            hooks.append(edit_handle)
        return hooks

    def init_aux_models(self, edit_modules:List[Union[nn.Linear, Conv1D]],
                        aux_model_config:DAFNetConfig.AuxModelConfig, 
                        if_edit_bias, devices) -> \
                        Tuple[nn.ModuleDict, Dict[str, List[int]]]:
        same_shape_em_ids = defaultdict(list)
        for i, em in enumerate(edit_modules):
            if isinstance(em, Conv1D):
                in_dim, out_dim = em.weight.shape
            elif isinstance(em, nn.Linear):
                out_dim, in_dim = em.weight.shape
            else:
                raise 'Only support linear layer.'
            shape = (in_dim, out_dim)
            same_shape_em_ids[shape].append(i)
        aux_models = nn.ModuleDict()
        shape_to_em_ids = {}
        em_id2_aux_model_device = [None]*len(edit_modules)
        for aux_i, (shape, em_ids) in enumerate(same_shape_em_ids.items()):
            aux_models[str(shape)] = GradientTransformer(shape[0], shape[1], 
                aux_model_config, len(em_ids), if_edit_bias).to(devices[aux_i%len(devices)])
            shape_to_em_ids[str(shape)] = em_ids
            for em_i in em_ids:
                em_id2_aux_model_device[em_i] = devices[aux_i%len(devices)]
        return aux_models, shape_to_em_ids, em_id2_aux_model_device
 
    def init_edit_representations(self, edit_modules:List[Union[nn.Linear, Conv1D]])->List[Tuple[List[torch.Tensor], torch.Tensor]]:
        edit_reps = [None for em in edit_modules]
        return edit_reps

    def init_edit_lrs(self, edit_modules:List[Union[nn.Linear, Conv1D]], 
                      init_lr:float, device):
        edit_lrs = [torch.tensor(init_lr) for _ in edit_modules]
        edit_lrs = nn.ParameterList(edit_lrs).to(device)
        return edit_lrs

    def init_autograd_params(self, modules_gradient_signal:List[Union[nn.Linear, Conv1D]]):
        autograd_params = nn.ParameterList()
        for mgs in modules_gradient_signal:
            # assert mgs.bias != None 
            autograd_params.append(mgs.weight)
        return autograd_params

    def init_cache_edit_module_original_weights(self, edit_modules:List[Union[nn.Linear, Conv1D]]):
        original_weights = [em.weight.clone().detach() for em in edit_modules]
        if self.cfg.if_edit_bias:
            original_biases = [em.bias.clone().detach() for em in edit_modules]
        else:
            original_biases = None
        return original_weights, original_biases

    ############################################################################
    #                          DAFNet Functions                                   #
    ############################################################################
    def clear_delta_weights(self):
        for em in self.edit_modules:
            em.__delta_weight__ = None
            em.__delta_bias__ = None
    
    def hold_delta_weights(self):
        '''
        Add delta weights onto model weights and clear delta weights.
        '''
        for em in self.edit_modules:
            if em.__delta_weight__.shape == em.weight.shape:
                em.weight += em.__delta_weight__.detach()
            else:
                em.weight += em.__delta_weight__.permute(1, 0).detach()
            if self.cfg.if_edit_bias:
                em.bias += em.__delta_bias__.flatten().detach()
        self.clear_delta_weights()

    def restore_to_original_weights(self):
        if self.cfg.if_edit_bias:
            for em, ow, ob in zip(self.edit_modules, self.original_weights, self.original_biases):
                em.load_state_dict({'weight': ow,'bias': ob})
        else:
            for em, ow in zip(self.edit_modules, self.original_weights):
                em.load_state_dict({'weight': ow}, strict=False)

    def clear_past_reps(self):
        self.edit_reps = self.init_edit_representations(self.edit_modules)

    def set_delta_weight_usable(self, if_use):
        self.__use_delta_weight__ = if_use
    
    def __get_model_gradient_signal__(self, input_ids:torch.Tensor, label_ids:torch.Tensor, masks:torch.Tensor):
        def preprocess_gradient_signals(U:List[torch.Tensor], V:List[torch.Tensor]):
            cat_x = []
            edit_lens = []
            for u, v in zip(U, V):
                batch_size, max_len, _ = u.shape
                bs = batch_size * max_len
                # get input x
                u = u.to(torch.float32).reshape(bs, -1) # [batch_size * max_len, x_dim]
                v = v.to(torch.float32).reshape(bs, -1) # [batch_size * max_len, delta_dim]
                g_mask = (u != 0).any(-1) * (v != 0).any(-1) # [batch_size * max_len]
                edit_lens.append(g_mask.reshape(batch_size, max_len).sum(1)) # [batch_size]
                u = u[g_mask] # [now_edit_len, x_dim]
                v = v[g_mask] # [now_edit_len, delta_dim]
                cat_x.append(torch.cat([u, v], -1))
            x = torch.cat(cat_x, 0)
            edit_lens = torch.cat(edit_lens, 0)
            return x, edit_lens
        
        # get gradient signals, input_ids/label_ids/masks: [batch, max_len]
        self.autograd_params.requires_grad_(True) 
        self.__get_gradient_signal_stage__ = True
        input_ids = input_ids.to(self.device_gradient_signal).split(self.cfg.gradient_signal_batch_size, 0)
        label_ids = label_ids.to(self.device_gradient_signal).split(self.cfg.gradient_signal_batch_size, 0)
        masks = masks.to(self.device_gradient_signal).split(self.cfg.gradient_signal_batch_size, 0)
        for iids, lids, msks in zip(input_ids, label_ids, masks):        
            edit_loss = label_loss(self.model_gradient_signal, iids, lids, msks, False)  
            # hooked `__x__` and `__weight__` in hook functions 
            torch.autograd.grad(edit_loss, self.autograd_params) 
        # self.autograd_params.zero_grad()
        self.__get_gradient_signal_stage__ = False
        self.autograd_params.requires_grad_(False)
        # preprocess gradient signals
        gradient_signals = []
        edit_lens = []
        for i, mgs in enumerate(self.modules_gradient_signal):
            gs, nel = preprocess_gradient_signals(mgs.__x__, mgs.__delta__)
            gradient_signals.append(gs.to(self.em_id2_aux_model_device[i]))
            edit_lens.append(nel.to(self.em_id2_aux_model_device[i]))
            mgs.__x__ = []
            mgs.__delta__ = []
        return gradient_signals, edit_lens

    def __update_delta_weight__(self, gradient_signals:List[torch.Tensor], edit_lens:List[torch.Tensor], detach_past = True):
        # Compute weight updates `__delta_weight__` and `__delta_bias__` (if available)
        # using gradient signals.
        def update_delta_weight(em_i, delta_weight, delta_bias, past_update_scale):
            em = self.edit_modules[em_i]
            lr = self.edit_lrs[em_i]
            if em.__delta_weight__ == None:
                em.__delta_weight__ = lr * delta_weight.to(self.device)
            else:
                if detach_past:
                    em.__delta_weight__ = em.__delta_weight__.detach() 
                em.__delta_weight__ = past_update_scale * em.__delta_weight__ + lr * delta_weight.to(self.device)
            # update delta bias
            if self.cfg.if_edit_bias:
                if em.__delta_bias__ == None:
                    em.__delta_bias__ = lr * delta_bias.to(self.device) # [dim_out]
                else:
                    if detach_past:
                        em.__delta_bias__ = em.__delta_bias__.detach()
                    em.__delta_bias__ = past_update_scale * em.__delta_bias__ + lr * delta_bias.to(self.device)
            
        for shape, em_ids in self.shape_to_em_ids.items():
            aux_model = self.aux_models[shape]
            if self.aux_module_parallel:
                inputs = []
                # get auxiliary model inputs
                for i, em_i in enumerate(em_ids):
                    # get new delta_weight & delta_bias and update representations
                    inputs.append((gradient_signals[em_i], edit_lens[em_i], i, self.edit_reps[em_i]))
                outpts = aux_model(*inputs)
                for em_i, outp in zip(em_ids, outpts):
                    delta_weight, delta_bias, past_update_scale, reps = outp
                    self.edit_reps[em_i] = reps
                    update_delta_weight(em_i, delta_weight, delta_bias, past_update_scale)
            else:
                for i, em_i in enumerate(em_ids):
                    # get new delta_weight & delta_bias and update representations
                    delta_weight, delta_bias, past_update_scale, reps = aux_model(
                        gradient_signals[em_i], edit_lens[em_i], i, self.edit_reps[em_i]) 
                    self.edit_reps[em_i] = reps
                    # update delta weights
                    update_delta_weight(em_i, delta_weight, delta_bias, past_update_scale)

    ############################################################################
    #         Implementation Virtual Functions of Base Class                   #
    ############################################################################
    def name_of_editor_and_model(self):
        return 'dafnet', self.cfg.edit_model_name

    def if_can_batch_edit(self)->bool:
        return True

    def restore_to_original_model(self):
        self.clear_delta_weights()
        self.clear_past_reps() 
        self.restore_to_original_weights()

    def edit_one_piece(self, request: Dict):
        '''request = {'prompt': str, 'target_new': str}'''
        self.edit_batch([request])

    def edit_batch(self, requests: List[Dict]): 
        '''
        Only for inference, and assume self.model a auto-regressive model.
        requests = [
          {'prompt': str, 'target_new': str},
          {'prompt': str, 'target_new': str},
        ]
        '''
        prompts, new_targets = [], []
        for r in requests: 
            prompts.append(r['prompt'])
            new_targets.append(r['target_new'])
        input_ids, label_ids, masks = prompts_target_to_x_y_mask(self.tokenizer, prompts, new_targets, self.device_gradient_signal)
        gradient_signals, edit_lens = self.__get_model_gradient_signal__(input_ids, label_ids, masks)
        self.__update_delta_weight__(gradient_signals, edit_lens, True) 
    
    ############################################################################
    #                      Training Functions                                  #
    ############################################################################
    def __prepare_loc_sample_before_edit__(self, loc_xm):
        '''
        loc_xm: {
            loss_name_1: (input_ids, masks)
            loss_name_2: (input_ids, masks), ...
        } -> loc_xym: {
            loss_name_1: (input_ids, pre_logits, masks)
            loss_name_2: (input_ids, pre_logits, masks), ...
        }
        '''
        with torch.no_grad():
            loc_xym = {}
            for loss_name, sp in loc_xm.items():
                input_ids, masks = sp
                pre_logits = torch.cat([self.model_gradient_signal(ids).logits
                    for ids in input_ids.to(self.device_gradient_signal). 
                    split(self.cfg.training.loss_batch_size, 0)], 0).to(self.device) 
                loc_xym[loss_name] = (input_ids, pre_logits, masks)
        return loc_xym

    def train_init(self, sample_count, get_data_by_ids, records_dir:str = 'train_records', 
        train_name_prefix = None, train_name:str = None, load_ckpt_path:str = None, 
        save_ckpt_per_i = 3000, log_per_i = 10, random_seed = None, 
        aux_module_parallel = False, aux_models_devices_master = [0, 1],
        aux_models_devices_slave = [2, 3, 4, 5]):  
        '''
        Used to initialize `ParallelDataset`:
            sample_count: count of data in dataset.
            get_data_by_ids: function getting data by ids, assume data structure: (
                edit_xym: (input_ids, label_ids, masks),
                gen_xym: {
                    loss_name_1: (input_ids, label_ids, masks),
                    loss_name_2: (input_ids, label_ids, masks), ...
                },
                loc_xm: {
                    loss_name_1: (input_ids, masks)
                    loss_name_2: (input_ids, masks), ...
                }  
            ), where `edit_xym` is the edit samples with count of `len(edit_xym)`.
        '''
        self.set_train(True) 
        # initialize data yielder
        self.random_rng = np.random.default_rng(random_seed)
        def get_data_and_gradient_signal_by_ids(ids):
            def select_loss_samples(*args):
                lsmc = min(len(args[0]), self.cfg.training.loss_sample_max_count)
                idx = self.random_rng.choice(len(args[0]), lsmc, replace = False)
                return [a[idx] for a in args]
            edit_xym, gen_xym, loc_xm = get_data_by_ids(ids)
            gradient_signals, edit_lens = self.__get_model_gradient_signal__(*edit_xym) 
            # select loss samples by max count
            edit_xym = select_loss_samples(*edit_xym)
            for loss_name, l in gen_xym.items():
                gen_xym[loss_name] = select_loss_samples(*l)
            for loss_name, l in loc_xm.items():
                loc_xm[loss_name] = select_loss_samples(*l)
            # get locality samples
            loc_xym = self.__prepare_loc_sample_before_edit__(loc_xm) 
            return gradient_signals, edit_lens, edit_xym, gen_xym, loc_xym
        self.now_seq_modeling = 1
        self.data_generator = ParallelDataset(sample_count, get_data_and_gradient_signal_by_ids, 
            list(range(1, self.now_seq_modeling + 1)), True, 4, random_seed = random_seed)
        self.ema_loss = 1
        self.min_ema_loss = 10
        self.min_ema_loss_i = 0
        # initialize training loss accumulating related
        self.accm_now_n = 0
        self.accm_edit_loss = 0
        self.accm_rel_loss = 0
        self.accm_gen_losses = defaultdict(float)
        self.accm_loc_losses = defaultdict(float)
        # initialize log directory and writer
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        train_name = (train_name_prefix + '-' if train_name_prefix else "") + \
            (train_name if train_name else t)
        records_dir = os.path.join(records_dir, *self.name_of_editor_and_model(), train_name)
        self.save_ckpt_dir = os.path.join(records_dir, 'checkpoints')
        if not os.path.exists(self.save_ckpt_dir):
            os.makedirs(self.save_ckpt_dir)
        logs_path = os.path.join(records_dir, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        with open(os.path.join(records_dir, 'config.yaml'), 'w') as f:
            yaml.dump(asdict(self.cfg), f)
        self.log_writer = SummaryWriter(logs_path)
        self.save_ckpt_per_i = save_ckpt_per_i
        self.log_per_i = log_per_i
        # auxiliary model training parallel
        self.aux_module_parallel = aux_module_parallel
        if aux_module_parallel:
            now_n = 0
            for i, shape in enumerate(self.aux_models.keys()):
                devices = [aux_models_devices_master[i]]
                for j, em_id in enumerate(self.shape_to_em_ids[shape]):
                    if j != 0:
                        devices.append(aux_models_devices_slave[now_n % len(aux_models_devices_slave)])
                        now_n += 1
                    self.em_id2_aux_model_device[em_id] = devices[j] 
                print(shape, 'Auxiliary model devices: ',devices)
                self.aux_models[shape].to(devices[0])
                self.aux_models[shape] = MultiGPUGradientTransformer(self.aux_models[shape], devices, self.device)
        # initialize optimizer and load checkpoints
        self.aux_models_opt = Adam(self.aux_models.parameters(), self.cfg.training.aux_model_lr)
        self.edit_lrs_opt = Adam(self.edit_lrs.parameters(), self.cfg.training.edit_lr_lr)
        if load_ckpt_path and os.path.isfile(load_ckpt_path):
            self.load_ckpt(load_ckpt_path, True)  
        else:
            self.train_i, self.train_epoch = 1, 1

    def set_train(self, if_train = False):
        self.model.train(False)
        self.model.requires_grad_(False)
        self.autograd_params.requires_grad_(False)
        self.autograd_params.train(False)
        self.model_gradient_signal.train(False)
        self.model_gradient_signal.requires_grad_(False)
        self.aux_models.requires_grad_(if_train)
        self.aux_models.train(if_train)
        self.edit_lrs.requires_grad_(if_train)
        self.edit_lrs.train(if_train)

    def train(self, epochs):
        if self.log_writer == None:
            raise "Call `self.train_init()` to initialize training first!"
        print('Checkpoints dir: ', self.save_ckpt_dir)
        start_epoch = self.train_epoch
        for self.train_epoch in range(start_epoch, epochs + 1): 
            progress_bar = tqdm(total = self.data_generator.sample_count, 
                position = 0, leave = True, desc = "Epoch %d"%self.train_epoch, dynamic_ncols = True)
            for gradient_signals, edit_lens, edit_xym, gen_xym, loc_xym in self.data_generator:
                # edit 
                self.__update_delta_weight__(gradient_signals, edit_lens, True)
                # train after edit
                log_dict = self.__train_a_batch__(edit_xym, gen_xym, loc_xym) 
                # log
                edit_n = len(edit_lens[0])
                log_dict["Edit-Samples-Count"] =  edit_n
                log_dict['Epoch'] = self.train_epoch
                if self.train_i % self.log_per_i == 0:
                    self.write_logs(self.train_i, log_dict)
                if self.train_i % self.save_ckpt_per_i == 0:
                    self.save_ckpt(self.train_i, self.train_epoch, log_dict['Edit-Loss'])
                self.train_i += 1 
                progress_bar.update(edit_n)
            progress_bar.close() 
        self.set_train(False)


    def __train_a_batch__(self, edit_xym:Tuple, gen_xym:Dict[str, Tuple], loc_xym:Dict[str, Tuple]):
        # edit loss & backward 
        gen_lambda = self.cfg.training.gen_lambda
        loc_lambda = self.cfg.training.loc_lambda
        # for n in gen_xym.keys():
        #     if 'cf' in n:
        #         gen_lambda = self.cfg.training.gen_lambda*5
        #         loc_lambda = self.cfg.training.loc_lambda*3
        #         break
        edit_loss, relia_loss, gen_losses, loc_losses = self.__edit_losses_and_backward__(
            self.model, edit_xym, gen_xym, loc_xym, self.cfg.training.relia_lambda, 
            gen_lambda, loc_lambda) 
        self.accm_edit_loss += edit_loss
        self.accm_rel_loss += relia_loss
        for loss_name, l in gen_losses.items():
            self.accm_gen_losses[loss_name] += l
        for loss_name, l in loc_losses.items():
            self.accm_loc_losses[loss_name] += l
        self.accm_now_n += 1
        # increase sequential modeling 
        self.ema_loss += (edit_loss - self.ema_loss) * self.cfg.training.ema_loss_lambda
        if self.ema_loss < self.min_ema_loss:
            self.min_ema_loss = self.ema_loss.item()
            self.min_ema_loss_i = self.train_i
        if self.train_i - self.min_ema_loss_i > self.cfg.training.increase_seq_modeling_i / max(1, np.log10(self.now_seq_modeling)):
            if self.now_seq_modeling < self.cfg.training.max_seq_modeling:
                self.now_seq_modeling += int(max(self.now_seq_modeling * self.cfg.training.increase_seq_modeling_pace, 10))
                self.now_seq_modeling = min(self.cfg.training.max_seq_modeling, self.now_seq_modeling)
                self.__reset_data_generator_batch_size__()  
                self.ema_loss += 0.3
                self.min_ema_loss = 10 
                self.min_ema_loss_i = self.train_i
        # log dict  
        log_dict = {
            "Edit-Loss": edit_loss,
            "Edit-Reliability-Loss": relia_loss,
            "Edit-Generality-Loss": dict(gen_losses),
            "Edit-Locality-Loss": dict(loc_losses),
            "Edit-EMA-Loss": self.ema_loss,
            "Edit-EMA-Min-Loss": self.min_ema_loss,
            "Now-Sequential-Modeling": self.now_seq_modeling
        }
        # reset edited model
        self.clear_delta_weights()
        self.clear_past_reps()
        # update auxiliary model
        if self.accm_now_n >= self.cfg.training.accum_n:
            log_dict.update({
                "Accumulate-Edit-Loss": self.accm_edit_loss,
                "Accumulate-Edit-Reliability-Loss": self.accm_rel_loss,
                "Accumulate-Edit-Generality-Loss": dict(self.accm_gen_losses),
                "Accumulate-Edit-Locality-Loss": dict(self.accm_loc_losses),
            })
            if self.cfg.training.grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.aux_models.parameters(), 
                                               100., error_if_nonfinite=True)
                log_dict['Grad-Norm'] = grad_norm 
            self.aux_models_opt.step()
            self.edit_lrs_opt.step()
            self.aux_models_opt.zero_grad()
            self.edit_lrs_opt.zero_grad()
            # reset accumulation
            self.accm_now_n = 0
            self.accm_edit_loss = 0
            self.accm_rel_loss = 0
            self.accm_gen_losses = defaultdict(float)
            self.accm_loc_losses = defaultdict(float)
        return log_dict

    def __reset_data_generator_batch_size__(self):
        # if self.now_seq_modeling >= self.cfg.training.max_seq_modeling:
        #     btzs = [1,10,100,1000]
        #     btzs = [i for i in btzs if i <= self.cfg.training.max_seq_modeling]
        #     if self.cfg.training.max_seq_modeling not in btzs:
        #         btzs.append(self.cfg.training.max_seq_modeling)
        #     print('batch size:', btzs)
        #     self.data_generator.set_batch_size(btzs) 
        # else:
            self.data_generator.set_batch_size(list(range(1, self.now_seq_modeling + 1))) 
        


    def __edit_losses_and_backward__(self, model, edit_xym:Tuple[torch.Tensor], 
                        gen_xym:Dict[str, Tuple], loc_xym:Dict[str, Tuple], 
                        relia_lambda, gen_lambda, loc_lambda):
        def split_and_scale(ids1, ids2, masks):
            average_scale = masks.sum() * self.cfg.training.accum_n
            ids1 = ids1.split(self.cfg.training.loss_batch_size, 0)
            ids2 = ids2.split(self.cfg.training.loss_batch_size, 0)
            masks = masks.split(self.cfg.training.loss_batch_size, 0)
            return ids1, ids2, masks, average_scale
        edit_loss = 0
        # reliability loss
        input_ids, label_ids, masks, average_scale = split_and_scale(*edit_xym)
        relia_coef = relia_lambda / average_scale
        relia_loss = 0
        for iids, lids, msks in zip(input_ids, label_ids, 
                tqdm(masks, leave = False, desc = "Computing reliability loss")):
            rl = relia_coef * label_loss(model, iids, lids, msks, False) 
            rl.backward(retain_graph=True)
            rl = rl.detach()
            relia_loss += rl
        edit_loss += relia_loss
        # generality loss
        gen_losses = defaultdict(float)
        for loss_name, sp in gen_xym.items():
            input_ids, label_ids, masks, average_scale = split_and_scale(*sp)
            gen_coef = gen_lambda / average_scale
            for iids, lids, msks in zip(input_ids, label_ids, 
                    tqdm(masks, leave = False, desc = "Computing generality loss: %s"%loss_name)):
                gl = gen_coef * label_loss(model, iids, lids, msks, False) 
                gl.backward(retain_graph=True)
                gl = gl.detach()
                gen_losses[loss_name] += gl
            edit_loss += gen_losses[loss_name] 
        # locality loss
        loc_losses = defaultdict(float)
        for loss_name, sp in loc_xym.items(): 
            input_ids, pre_logits, masks, average_scale = split_and_scale(*sp)
            loc_coef = loc_lambda / average_scale
            for iids, pls, msks in zip(input_ids, pre_logits, 
                    tqdm(masks, leave = False, desc = "Computing locality loss: %s"%loss_name)):
                ll = loc_coef * logit_KL_loss(pls, model(iids).logits, msks, False) 
                ll.backward(retain_graph=True)
                ll = ll.detach()
                loc_losses[loss_name] += ll
            edit_loss += loc_losses[loss_name]
        return edit_loss, relia_loss, gen_losses, loc_losses


    def write_logs(self, i, logs:dict):
        for log_name, log in logs.items():
            if type(log) == dict:
                logs1 = {}
                for n, l in log.items():
                    logs1[log_name + '-' + n] = l
                self.write_logs(i, logs1)
            else:   
                self.log_writer.add_scalar(log_name, log, i)


    def save_ckpt(self, i:int, epoch:int, loss:float):
        ckpt_name = 'epoch-%d-i-%d-loss-%.4f'%(epoch, i, loss)
        ckpt_path = os.path.join(self.save_ckpt_dir, ckpt_name)
        ckpt = {
            'i': i,
            'epoch': epoch,
            'loss': loss,
            'edit_lrs':self.edit_lrs.state_dict(),
            'aux_models_opt': self.aux_models_opt.state_dict(),
            'edit_lrs_opt': self.edit_lrs_opt.state_dict(),
            'now_seq_modeling': self.now_seq_modeling, 
            'ema_loss': self.ema_loss,
            'min_ema_loss': self.min_ema_loss,
            'min_ema_loss_i': self.min_ema_loss_i
        }
        if self.aux_module_parallel:
            ckpt['aux_models'] = {k:self.aux_models[k].module.state_dict() for k in self.aux_models},
        else:   
            ckpt['aux_models'] = {k:self.aux_models[k].state_dict() for k in self.aux_models},

        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path, is_train = False, restrict = True):
        ckpt = torch.load(ckpt_path, 'cpu')
        if self.aux_module_parallel:
            for k in self.aux_models:
                self.aux_models[k].module.load_state_dict(ckpt['aux_models'][0][k], restrict)
        else:
            for k in self.aux_models:
                self.aux_models[k].load_state_dict(ckpt['aux_models'][0][k], restrict)
        self.edit_lrs.load_state_dict(ckpt['edit_lrs'], restrict)
        if is_train:
            if self.log_writer == None:
                raise "Call `self.train_init()` to initialize training first!"
            self.aux_models_opt.load_state_dict(ckpt['aux_models_opt'])
            self.edit_lrs_opt.load_state_dict(ckpt['edit_lrs_opt'])
            if 'now_seq_modeling' in ckpt: 
                self.now_seq_modeling = ckpt['now_seq_modeling'] 
                self.__reset_data_generator_batch_size__()
                self.ema_loss = float(ckpt['ema_loss'])
                self.min_ema_loss = ckpt['min_ema_loss']
                self.min_ema_loss_i = ckpt['min_ema_loss_i']
                self.train_i = ckpt['i']
                self.train_epoch = ckpt['epoch']
        print('Load DAFNet checkpoints from', ckpt_path)
        return ckpt['loss']


def find_module(module:nn.Module, module_path:List[str]):
    for comp in module_path:
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module
 
def label_loss(model, input_ids:torch.Tensor, label_ids:torch.Tensor, masks:torch.Tensor, average = True):
    # input_ids/label_ids/masks: [batch, max_len]
    logits = model(input_ids).logits
    log_pre_p = torch.log_softmax(logits, 2) # [batch, max_len, voc_size]
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, max_len]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

def logit_KL_loss(logits1:torch.Tensor, logits2:torch.Tensor, masks:torch.Tensor, average = True):
    # logits1/logits2: [batch, max_len, voc_size], masks: [batch, max_len]
    log_p1 = torch.log_softmax(logits1, 2)
    log_p2 = torch.log_softmax(logits2, 2)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

