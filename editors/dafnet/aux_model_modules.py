
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from typing import Any, List

class ExactBatchNorm1d(nn.Module):
    def __init__(self, dim, eps = 1e-8) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("k", torch.tensor(0))
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x):
        # x: [batch_size, dim]
        bcs, _ = x.shape
        if self.training:
            if self.k != 0 or bcs != 1:
                old_k, new_k = self.k, self.k + bcs
                new_mean = old_k / new_k * self.mean  + x.sum(0) / new_k
                new_var_1 = (old_k - 1) / (new_k - 1) * self.var
                new_var_2 =  old_k / (new_k - 1) * (new_mean - self.mean) ** 2 
                new_var_3 =  ((new_mean - x) ** 2).sum(0) / (new_k - 1)
                new_var = new_var_1 + new_var_2 + new_var_3
                self.mean = new_mean
                self.var = new_var
                self.std = new_var ** 0.5
                self.k = new_k
            else: # self.k == 0 and bcs == 1:
                self.k += 1
                self.mean = x[0]
                return x
        x = (x - self.mean) / (self.std + self.eps)
        return x


class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-8, momentum=0.1):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("eps", torch.tensor(eps))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            current_mean = x.mean(0)
            current_var = x.var(0, unbiased=False)
            self.mean = (1 - self.momentum) * self.mean + self.momentum * current_mean
            self.var = (1 - self.momentum) * self.var + self.momentum * current_var
            self.std = self.var.sqrt()
        x = (x - self.mean) / (self.std + self.eps)
        return x

class FeedForward(nn.Module):
    def __init__(self, input_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EditOuterAttention(nn.Module):
    def __init__(self, input_dim, head_n, qkv_dim, return_attention_score):
        super().__init__()
        self.lq = nn.Linear(input_dim, head_n*qkv_dim)
        self.lk = nn.Linear(input_dim, head_n*qkv_dim)
        if not return_attention_score:
            self.lv = nn.Linear(input_dim, head_n*qkv_dim)
            self.lo = nn.Linear(head_n*qkv_dim, input_dim)
        self.qkv_dim = qkv_dim
        self.head_n = head_n
        self.norm = qkv_dim**0.5
        self.return_attention_score = return_attention_score
        
    def forward(self, x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor):
        # x: [batch_size, len_x, dim]
        # y: [batch_size, len_y, dim] 
        # mask: [batch_size, 1, len_x, len_y] ~ (0 & -9999999)
        bs, len_x, _ = x.shape
        bs, len_y, _ = y.shape
        q = self.lq(x)
        k = self.lk(y)
        # [batch_size, head_n, len_x, qkv_dim]
        q = q.reshape(bs, len_x, self.head_n, self.qkv_dim).permute(0, 2, 1, 3) 
        # [batch_size, head_n, qkv_dim, len_y]
        k = k.reshape(bs, len_y, self.head_n, self.qkv_dim).permute(0, 2, 3, 1) 
        # [batch_size, head_n, len_x, len_y]
        attention_scores = torch.matmul(q, k) / self.norm + mask
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if self.return_attention_score:
            return attention_scores.mean(1) # [batch_size, len_x, len_y]
        v = self.lv(y)
        # [batch_size, head_n, len_y, qkv_dim]
        v = v.reshape(bs, len_y, self.head_n, self.qkv_dim).permute(0, 2, 1, 3)
        # [batch_size, head_n, len_x, qkv_dim]
        output = torch.matmul(attention_scores, v)
        # [batch_size, len_x, head_n * qkv_dim]
        output = output.permute(0, 2, 1, 3).reshape(bs, len_x, -1) 
        output = self.lo(output)
        return output

    def __call__(self, x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor)->torch.Tensor:
        return self._call_impl(x, y, mask)
    

class EditInnerAttention(nn.Module):
    def __init__(self, inpt_dim, attn_dim):
        super().__init__()
        self.trans1 = nn.Sequential(
            nn.Linear(inpt_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, inpt_dim)
        )
        self.trans2 = nn.Sequential(
            nn.Linear(inpt_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 1),
        )
        
    def forward(self, x:torch.Tensor, edit_lens:torch.Tensor):
        # x: [now_edit_len, d]
        att_x = self.trans1(x) # [now_edit_len, d]
        att_s = self.trans2(att_x) # [now_edit_len, 1]

        att_x = torch.split(att_x, edit_lens.tolist(), 0) 
        att_x = pad_sequence(att_x, True, 0) # [edit_n, max_l, d]
        att_s = torch.split(att_s, edit_lens.tolist(), 0)
        att_s = pad_sequence(att_s, True, -999999)  
        att_s = torch.softmax(att_s, 1) # [edit_n, max_l, 1]

        att_x = att_x * att_s # [edit_n, max_l, d]
        sample_rep = att_x.sum(1) # [edit_n, d]
        
        mask = torch.ones(edit_lens.sum(0), dtype = torch.bool, device = x.device)
        mask = torch.split(mask, edit_lens.tolist(), 0)
        mask = pad_sequence(mask, True, False)  # [edit_n, max_l]
        att_x = att_x[mask]  # [now_edit_len, d]
        
        return att_x, sample_rep
    
    def __call__(self, x:torch.Tensor, edit_lens:torch.Tensor)->torch.Tensor:
        return self._call_impl(x, edit_lens)
    

class DAFNetBaseLayer(nn.Module):
    def __init__(self, input_dim, inner_att_dim, ffn_dim, outer_att_dim, 
                 outer_att_head_n, outer_score_att_dim, outer_score_att_head_n, n_modes):
        super().__init__()
        self.inner_att = EditInnerAttention(input_dim, inner_att_dim)
        self.outer_att = EditOuterAttention(input_dim, outer_att_head_n, outer_att_dim, False)
        self.outer_att_score = EditOuterAttention(input_dim, outer_score_att_head_n, outer_score_att_dim, True)
        self.feedforward = FeedForward(input_dim, ffn_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)

        self.mode_shift = nn.Embedding(n_modes, input_dim)
        self.mode_shift.weight.data.zero_()
        self.mode_scale = nn.Embedding(n_modes, input_dim)
        self.mode_scale.weight.data.fill_(1)
 
    def forward(self, x, past_rep:torch.Tensor, edit_lens:torch.Tensor, mode:torch.Tensor):
        # x: [now_edit_len, d]
        # past_rep: [1, past_edit_n, d] 
        
        # edit inner attention
        att_x, sample_rep = self.inner_att(x, edit_lens) # x: [now_edit_len, d], sample_rep: [edit_n, d] 
        x = x + att_x
        x = self.layer_norm1(x)
        sample_rep = sample_rep.unsqueeze(0) # [1, now_edit_n, d] 

        # edit outer attention
        past_edit_n, now_edit_n = past_rep.shape[1], sample_rep.shape[1]
        new_rep = torch.cat([past_rep, sample_rep], 1) # [1, past_edit_n + now_edit_n, d] 

        mask = torch.ones((now_edit_n, past_edit_n + now_edit_n), device = x.device)
        mask = torch.triu(mask, past_edit_n + 1)
        mask[mask == 1] = -999999
        att_x = self.outer_att(sample_rep, new_rep, mask)[0] # [now_edit_n, d] 
        att_x = att_x.repeat_interleave(edit_lens, 0) # [now_edit_len, d] 
        x = x + att_x # [now_edit_len, d] 
        x = self.layer_norm2(x)
        
        # compute final scales
        update_scale = self.outer_att_score(sample_rep, new_rep, mask)[0] # [now_edit_n, past_edit_n + now_edit_n] 
        update_scale = update_scale.diag(past_edit_n) # [now_edit_n]
        update_scale_p = torch.cumprod((1 - update_scale).flip(0), 0).flip(0)
        update_scale[:-1] *= update_scale_p[1:]
        past_update_scale = update_scale_p[0]

        # extra transformation
        ffn_x = self.feedforward(x)
        x = self.layer_norm3(x + ffn_x)
        
        # mode transformation
        scale, shift = self.mode_scale(mode), self.mode_shift(mode)
        pre_act = x * scale + shift
        # need clamp instead of relu so gradient at 0 isn't 0
        x = pre_act.clamp(min=0) + x
        return x, past_update_scale, update_scale, new_rep
    
    def __call__(self, x, past_rep:torch.Tensor, edit_lens:torch.Tensor, mode:torch.Tensor):
        return self._call_impl(x, past_rep, edit_lens, mode)
    


class DAFNetModel(nn.Module):
    def __init__(self, layer_n, hidden_dim, inner_att_dim, ffn_dim, outer_att_dim, 
            outer_att_head_n, outer_score_att_dim, outer_score_att_head_n, n_modes):
        super().__init__()
        self.layers = nn.ModuleList([DAFNetBaseLayer(hidden_dim, inner_att_dim, 
            ffn_dim, outer_att_dim, outer_att_head_n, outer_score_att_dim, 
            outer_score_att_head_n, n_modes) for _ in range(layer_n)])
    def forward(self, x, past_reps:torch.Tensor, edit_lens:torch.Tensor, mode:torch.Tensor):
        # x: [now_edit_len, x_dim + delta_dim]
        # past_reps: List[[1, past_edit_n, d], ...]
        # edit_lens: [now_edit_n]
        # mode: torch.tensor(int)
        new_reps = []
        past_update_scale = []
        update_scale = []
        for layer, pr in zip(self.layers, past_reps): 
            x, pus, us, nr = layer(x, pr, edit_lens, mode)
            new_reps.append(nr.detach()) 
            past_update_scale.append(pus.detach()) 
            update_scale.append(us.detach()) 
        past_update_scale = torch.stack(past_update_scale, 0).mean(0) #[1]
        update_scale = torch.stack(update_scale, 0).mean(0) # [edit_n]
        return x, new_reps, past_update_scale, update_scale

    def __call__(self, x, past_reps:torch.Tensor, edit_lens:torch.Tensor, mode:torch.Tensor):
        return self._call_impl(x, past_reps, edit_lens, mode)
    
