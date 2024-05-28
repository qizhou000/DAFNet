#%%
import torch
from torch import nn
from typing import Any, List
from .aux_model_modules import ExactBatchNorm1d, BatchNorm1d, DAFNetModel

class GradientTransformer(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg,#:DAFNetConfig.AuxModelConfig, 
                 n_modes, edit_bias):
        super().__init__()
        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.model_dim = x_dim + delta_dim
        if cfg.norm_layer == 'EBN':
            self.norm_layer = ExactBatchNorm1d(self.model_dim)
        elif cfg.norm_layer == 'BN':
            self.norm_layer = BatchNorm1d(self.model_dim)
        elif cfg.norm_layer == 'LN':
            self.norm_layer = nn.LayerNorm(self.model_dim)
        else:
            raise
        self.dafnet_model = DAFNetModel(cfg.layer_n, self.model_dim,
            cfg.inner_att_dim, cfg.ffn_dim, cfg.outer_att_dim, cfg.outer_att_head_n, 
            cfg.outer_score_att_dim, cfg.outer_score_att_head_n, n_modes)
        self.layer_n = cfg.layer_n
        self.edit_bias = edit_bias
    
    def set_past_reps(self, device):
        past_reps = [torch.zeros([1, 0, self.model_dim], device = device)
                        for i in range(self.layer_n)]
        return past_reps

    def forward(self, x:torch.Tensor, edit_lens:torch.Tensor, mode:int, past_reps:List[torch.Tensor]):
        '''
        x: [now_edit_len, x_dim + delta_dim]
        edit_lens: [now_edit_n]
        past_reps: [[1, past_edit_n, x_dim + delta_dim] * cfg.t_layer_n]
        ''' 
        # x, edit_lens = self.__get_inputs__(U, V)
        if past_reps == None:
            past_reps = self.set_past_reps(x.device)
        # model process
        x = self.norm_layer(x) # [now_edit_len, x_dim + delta_dim]
        mode = torch.tensor(mode).to(x.device)
        x, new_reps, past_update_scale, update_scale = self.dafnet_model(x, 
                                        past_reps, edit_lens, mode)
        # rescale update weight
        update_scale = update_scale.repeat_interleave(edit_lens)
        update_scale = update_scale / edit_lens.repeat_interleave(edit_lens)
        x1, x2 = torch.split(x, [self.x_dim, self.delta_dim], -1) 
        x2 = x2 * update_scale.unsqueeze(1)

        delta_weight = x1.permute(1,0) @ x2
        delta_bias = None
        if self.edit_bias:
            delta_bias = x2.sum(0) 
            
        return delta_weight, delta_bias, past_update_scale.item(), new_reps


class MultiGPUGradientTransformer(nn.DataParallel):  
    def __init__(self, module, device_ids, output_device):
        super().__init__(module, device_ids, output_device)
        self.kwargs = [{}]*len(self.device_ids)
        self.out_device = output_device

    def forward(self, *inputs):
        # from itertools import chain
        '''
        element of inputs must at specific GPU!
        '''
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return [self.module(*i) for i in inputs]
            if len(inputs) != len(self.device_ids):
                raise 
            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **self.kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, self.kwargs)
            outputs = [(dw.to(self.out_device), db.to(self.out_device) if db != None else db, pus, nr) for dw, db, pus, nr in outputs]
            return outputs
   