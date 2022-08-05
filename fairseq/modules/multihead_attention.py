# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        simple_attention=False,
        cosformer_attn_enable=False,
        cosformer_expt_attn_enable=False,
        combin_attn_enable=False,
        combin_expt_attn_enable=False,
        enable_norm_stretch_factor=True,
        max_src_len_step_size=128,
        shortened_expt_simil=False,
        linear_pos_enable=False,
        linear_pos_layer_num=2,
        #linear_simul_attn_chkpts=False,
        #simul_attn_chkpts = Optional[Dict[str, Dict[str, Optional[Tensor]]]
    ):
        super().__init__()
        
        # VA, quick debugging statement
        torch.autograd.set_detect_anomaly(True)
        
        # hard-coded ratio for now, currently set to speech to text version
        #self.tgt_len_mod = 1.5
        self.tgt_len_mod = 0.6

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.simple_attention = simple_attention
        print(f"Simple attention: {self.simple_attention}", flush=True)	

        self.cosformer_attn_enable = cosformer_attn_enable
        print(f"cosFormer attention: {self.cosformer_attn_enable}", flush=True)	
        
        self.cosformer_expt_attn_enable = cosformer_expt_attn_enable
        print(f"cosFormer expt attention: {self.cosformer_expt_attn_enable}", flush=True)	
        
        self.combin_attn_enable = combin_attn_enable
        print(f"combin attention: {self.combin_attn_enable}", flush=True)	
        
        self.combin_expt_attn_enable = combin_expt_attn_enable
        print(f"combin_expt attention: {self.combin_expt_attn_enable}", flush=True)	
       
        self.enable_norm_stretch_factor = enable_norm_stretch_factor

        self.max_src_len_step_size = max_src_len_step_size
        
        self.shortened_expt_simil = shortened_expt_simil

        # generation of additional linear layers, not helpful for non-linear behavior however
        self.linear_pos_enable = linear_pos_enable
        self.linear_pos_layers = nn.ModuleList()
        #if self.linear_pos_enable:
        #    for i in range(linear_pos_layer_num):
        #        self.linear_pos_layers.append(nn.Linear(
    
        # hardcoded for quick iteration
        self.linearized_train = False

        # implemented for quick testing, will add some functionality later
        # self.linear_simul_attn_chkpts = linear_simul_attn_chkpts
        self.load_simul_attn_chkpts = {}

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.skip_embed_dim_check = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _get_reserve_head_index(self, num_heads_to_keep: int):
        k_proj_heads_norm = []
        q_proj_heads_norm = []
        v_proj_heads_norm = []

        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.k_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.k_proj.bias[start_idx:end_idx])).tolist()
            )
            q_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.q_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.q_proj.bias[start_idx:end_idx])).tolist()
            )
            v_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.v_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.v_proj.bias[start_idx:end_idx])).tolist()
            )

        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(
                k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i]
            )

        sorted_head_index = sorted(
            range(self.num_heads), key=lambda k: heads_norm[k], reverse=True
        )
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            start = sorted_head_index[i] * self.head_dim
            end = (sorted_head_index[i] + 1) * self.head_dim
            reserve_head_index.append((start, end))
        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index: List[Tuple[int, int]]):
        new_q_weight = []
        new_q_bias = []
        new_k_weight = []
        new_k_bias = []
        new_v_weight = []
        new_v_bias = []
        new_out_proj_weight = []

        for ele in reserve_head_index:
            start_idx, end_idx = ele
            new_q_weight.append(
                self.q_proj.weight[
                    start_idx:end_idx,
                ]
            )
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])

            new_k_weight.append(
                self.k_proj.weight[
                    start_idx:end_idx,
                ]
            )

            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])

            new_v_weight.append(
                self.v_proj.weight[
                    start_idx:end_idx,
                ]
            )
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])

            new_out_proj_weight.append(self.out_proj.weight[:, start_idx:end_idx])

        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True

        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True

        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True

        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True

        self.q_proj.weight = torch.nn.Parameter(new_q_weight)
        self.q_proj.bias = torch.nn.Parameter(new_q_bias)

        self.k_proj.weight = torch.nn.Parameter(new_k_weight)
        self.k_proj.bias = torch.nn.Parameter(new_k_bias)

        self.v_proj.weight = torch.nn.Parameter(new_v_weight)
        self.v_proj.bias = torch.nn.Parameter(new_v_bias)

        self.out_proj.weight = torch.nn.Parameter(new_out_proj_weight)

        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def cosformer_attn_baseline_train(
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx = None,
        is_tpu = False,
    ):

        src_len = k.size(1)
        tgt_len = q.size(1)
        bsz = int(k.size(0) / self.num_heads)

        # implementation differs from typical key_padding_mask application, but this is useful later and should be fine
        if key_padding_mask is not None:
            key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
            k = k.view(bsz, self.num_heads, src_len, list(k.shape)[2])          
            k = k.masked_fill(key_pad_mask_unsqueeze, 0)
            k = k.view(bsz*self.num_heads, src_len, list(k.shape)[3])          
           
        # begin setup and transformations   
        max_len = max(src_len, tgt_len)
        q_sin_init = q
        q_cos_init = q
        k_sin_init = k
        k_cos_init = k
        q_sin = torch.zeros(q.shape, device=q.device)
        q_cos = torch.zeros(q.shape, device=q.device)
        k_sin = torch.zeros(k.shape, device=k.device)
        k_cos = torch.zeros(k.shape, device=k.device)
        idx = torch.arange(1, max_len + 1, device=k.device)
        norm_sin = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        norm_cos = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        
        # transform tensors
        sin_tr_q = torch.sin((math.pi / 2) * (idx[:tgt_len] / tgt_len))
        cos_tr_q = torch.cos((math.pi / 2) * (idx[:tgt_len] / tgt_len))
        
        sin_tr_q = sin_tr_q.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cos_tr_q = cos_tr_q.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        sin_tr_k = torch.sin((math.pi / 2) * (idx[:src_len] / src_len))
        cos_tr_k = torch.cos((math.pi / 2) * (idx[:src_len] / src_len))
        
        sin_tr_k = sin_tr_k.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cos_tr_k = cos_tr_k.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # query transforms
        q_sin = torch.matmul(q_sin_init.unsqueeze(-1), sin_tr_q)
        q_cos = torch.matmul(q_cos_init.unsqueeze(-1), cos_tr_q)
        q_sin = q_sin.squeeze(-1)
        q_cos = q_sin.squeeze(-1)
        
        # key transforms
        k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr_k).squeeze(-1)
        k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr_k).squeeze(-1)
       

        if self.linearized_train:
            # einsum based approach should be much faster, larger space complexity however
            # the below expression stores the d x d resulting matrices instead of adding them together, outer product notation
            kTv_sin_steps = torch.einsum('nld,nlm->nldm', k_sin, v)
            kTv_cos_steps = torch.einsum('nld,nlm->nldm', k_cos, v)

            kTv_sin_cum = torch.cumsum(kTv_sin_steps, dim=1)
            kTv_cos_cum = torch.cumsum(kTv_cos_steps, dim=1)

            attn_weights_sin = torch.einsum('nld,nldm->nlm', q_sin, kTv_sin_cum)
            attn_weights_cos = torch.einsum('nld,nldm->nlm', q_cos, kTv_cos_cum)
            attn_weights = attn_weights_sin + attn_weights_cos

            # building out normalization
            norm_sin = torch.cumsum(k_sin, dim=1)
            norm_cos = torch.cumsum(k_cos, dim=1)
            
            prob_norm_sin = torch.bmm(q_sin, norm_sin.transpose(1, 2))
            prob_norm_cos = torch.bmm(q_cos, norm_cos.transpose(1, 2))
            prob_norm = prob_norm_sin + prob_norm_cos

            prob_norm = torch.diagonal(prob_norm, dim1=1, dim2=2).unsqueeze(-1)
            prob_norm = torch.clamp_min(prob_norm, 0.1)
            
            attn = attn_weights / prob_norm

        # quadratic doesn't experience a memory bottleneck
        else:
            attn_weights_sin = torch.bmm(q_sin, k_sin.transpose(1, 2))
            attn_weights_cos = torch.bmm(q_cos, k_cos.transpose(1, 2))
            attn_weights = attn_weights_sin + attn_weights_cos

            if attn_mask is not None:
                attn_mask_bool = attn_mask.to(torch.bool)
                attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
           
            attn_weights = self.dropout_module(attn_weights)
            attn_probs = torch.bmm(attn_weights, v)

            # section to try and replicate casual relationship in normalization
            if attn_mask is not None:
                norm_sin = torch.cumsum(k_sin, dim=1).transpose(1, 2)
                norm_cos = torch.cumsum(k_cos, dim=1).transpose(1, 2)
            else:
                norm_sin = torch.sum(k_sin, dim=1).unsqueeze(-1)
                norm_cos = torch.sum(k_cos, dim=1).unsqueeze(-1)
            
            prob_norm_sin = torch.bmm(q_sin, norm_sin)
            prob_norm_cos = torch.bmm(q_cos, norm_cos)
            prob_norm = prob_norm_sin + prob_norm_cos

            if attn_mask is not None:
                prob_norm = torch.diagonal(prob_norm, dim1=1, dim2=2).unsqueeze(-1)

            prob_norm = torch.clamp_min(prob_norm, 0.1)

            attn = attn_probs / prob_norm
            
            #print(torch.isnan(attn_weights).any())
            #print(torch.isnan(attn).any())
            #print(torch.isnan(prob_norm).any())
            #print(attn.shape)
            #print(attn_weights.shape)
            #print(k.shape)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        return attn, attn_weights

    def cosformer_attn_baseline_infer(
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        tgt_len_mod = None,
    ):
        
        src_idx = incremental_state["steps"]["src"]
        tgt_idx = incremental_state["steps"]["tgt"]
       
        src_len = k.size(1)
        tgt_len = q.size(1)

        tgt_len_p = src_idx * self.tgt_len_mod

        idx = torch.arange(1, src_len + 1, device = k.device)
        
        # transform tensors
        sin_tr = torch.sin((math.pi / 2) * (idx / tgt_len_p))
        cos_tr = torch.cos((math.pi / 2) * (idx / tgt_len_p))
        
        sin_tr = sin_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cos_tr = cos_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # query transforms
        q_sin = torch.mul(q, math.sin((math.pi / 2) * (tgt_idx / tgt_len_p)))
        q_cos = torch.mul(q, math.cos((math.pi / 2) * (tgt_idx / tgt_len_p)))
        
        # key transforms
        k_sin = torch.matmul(k.unsqueeze(-1), sin_tr).squeeze(-1)
        k_cos = torch.matmul(k.unsqueeze(-1), cos_tr).squeeze(-1)

        # construct d x d intermediate matrices and normalization tensors
        kTv_sin = torch.bmm(k_sin.transpose(1, 2), v)
        kTv_cos = torch.bmm(k_cos.transpose(1, 2), v)

        norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
        norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)

        # final attn calculations
        attn_weights_sin = torch.bmm(q_sin, kTv_sin)
        attn_weights_cos = torch.bmm(q_cos, kTv_cos)
        attn_weights = attn_weights_sin + attn_weights_cos

        prob_norm_sin = torch.bmm(q_sin, norm_sin)
        prob_norm_cos = torch.bmm(q_cos, norm_cos)
        prob_norm = prob_norm_sin + prob_norm_cos

        prob_norm = torch.clamp_min(prob_norm, 0.1)
        attn_probs = attn_weights / prob_norm

        attn = attn_probs
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, 1, self.embed_dim)
        attn = self.out_proj(attn)

        return attn, attn_weights

    def cosformer_attn_cache_infer( 
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx = None,
        tgt_len_mod = None,
    ):
        
        # beginning of attention calculations
        src_idx = incremental_state["steps"]["src"]
        tgt_idx = incremental_state["steps"]["tgt"]
       
        src_len = k.size(1)
        tgt_len = q.size(1)
        
        tgt_len_p = src_idx * self.tgt_len_mod

        old_src_idx = simul_attn_chkpts["old_indices"]["src"]
        old_tgt_idx = simul_attn_chkpts["old_indices"]["tgt"]
       
        q_sin = torch.mul(q, math.sin((math.pi / 2) * (tgt_idx / tgt_len_p)))
        q_cos = torch.mul(q, math.cos((math.pi / 2) * (tgt_idx / tgt_len_p)))

        k_sin = torch.mul(k, math.sin((math.pi / 2) * (tgt_idx / tgt_len_p)))
        k_cos = torch.mul(k, math.cos((math.pi / 2) * (tgt_idx / tgt_len_p)))

        #idx = torch.arange(1, src_len + 1)
        #sin_tr = torch.sin((math.pi / 2) * (idx / tgt_len_p))
        #cos_tr = torch.cos((math.pi / 2) * (idx / tgt_len_p))
        #sin_tr = simul_attn_chkpts["sin_tr"]
        #cos_tr = simul_attn_chkpts["cos_tr"]

        #k_sin_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_sin"]
        #k_cos_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_cos"]
        norm_sin_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_sin"]
        norm_cos_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_cos"]

        # build key transforms if necessary
#        if k_sin_old is not None and k_cos_old is not None:
#            old_src = list(k_sin_old.shape)[1]
#            if old_src == tgt_idx:
#                k_sin = k_sin_old
#                k_cos = k_cos_old
#            else:
#                sin_tr = sin_tr[old_src:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                cos_tr = cos_tr[old_src:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                k_sin = torch.cat((k_sin_old, torch.matmul(k[:, old_src:, :].unsqueeze(-1), sin_tr).squeeze(-1)), dim=1)
#                k_cos = torch.cat((k_cos_old, torch.matmul(k[:, old_src:, :].unsqueeze(-1), cos_tr).squeeze(-1)), dim=1)
#        else:
#            sin_tr = sin_tr[:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#            cos_tr = cos_tr[:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#            k_sin = torch.matmul(k.unsqueeze(-1), sin_tr).squeeze(-1)
#            k_cos = torch.matmul(k.unsqueeze(-1), cos_tr).squeeze(-1)
#
#        # build normalization vectors if necessary
#        if norm_sin_old is not None and norm_cos_old is not None:
#            if old_src == tgt_idx:
#                norm_sin = norm_sin_old
#                norm_cos = norm_cos_old
#            else:
#                norm_sin = norm_sin_old + torch.sum(k_sin.unsqueeze(-1)[:, old_src:, :], dim=1)
#                norm_cos = norm_cos_old + torch.sum(k_cos.unsqueeze(-1)[:, old_src:, :], dim=1)
#        else:
#            norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
#            norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)

        # build normalization vectors
        if norm_sin_old is not None and norm_cos_old is not None:
            norm_sin = norm_sin_old + k_sin.transpose(1, 2)
            norm_cos = norm_cos_old + k_cos.transpose(1, 2)
        else:
            norm_sin = k_sin.transpose(1, 2)
            norm_cos = k_cos.transpose(1, 2)


        # build out d x d intermediate matrix
        old_attn_weights_v_sin = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_sin"]
        old_attn_weights_v_cos = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_cos"]

#        if old_attn_weights_v_sin is not None and old_attn_weights_v_cos is not None:
#            if old_src == tgt_idx:
#                attn_weights_v_sin = old_attn_weights_v_sin
#                attn_weights_v_cos = old_attn_weights_v_cos
#            else:
#                attn_weights_v_sin = old_attn_weights_v_sin + torch.bmm(k_sin[:, old_src:, :].transpose(1, 2), v[:, old_src:, :]) 
#                attn_weights_v_cos = old_attn_weights_v_cos + torch.bmm(k_cos[:, old_src:, :].transpose(1, 2), v[:, old_src:, :])
#        else:
#            attn_weights_v_sin = torch.bmm(k_sin.transpose(1, 2), v)
#            attn_weights_v_cos = torch.bmm(k_cos.transpose(1, 2), v)

        # build out d x d intermediate matrix
        if old_attn_weights_v_sin is not None and old_attn_weights_v_cos is not None:
            attn_weights_v_sin = old_attn_weights_v_sin + torch.bmm(k_sin.transpose(1, 2), v)
            attn_weights_v_cos = old_attn_weights_v_cos + torch.bmm(k_cos.transpose(1, 2), v)
        else:
            attn_weights_v_sin = torch.bmm(k_sin.transpose(1, 2), v)
            attn_weights_v_cos = torch.bmm(k_cos.transpose(1, 2), v)

        attn_weights_sin = torch.bmm(q_sin, attn_weights_v_sin)
        attn_weights_cos = torch.bmm(q_cos, attn_weights_v_cos)
        attn_weights = attn_weights_sin + attn_weights_cos

        prob_norm_sin = torch.bmm(q_sin, norm_sin)
        prob_norm_cos = torch.bmm(q_cos, norm_cos)
        prob_norm = prob_norm_sin + prob_norm_cos

        #prob_norm.expand(list(prob_norm.shape)[0], list(prob_norm.shape)[1], list(attn_weights.shape)[2])
        prob_norm = torch.clamp_min(prob_norm, 0.1)

        attn_probs = attn_weights / prob_norm

        attn = attn_probs

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, 1, self.embed_dim)
        attn = self.out_proj(attn)

        simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_sin"] = norm_sin
        simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_cos"] = norm_cos
        #simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_sin"] = k_sin
        #simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_cos"] = k_cos
        simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_sin"] = attn_weights_v_sin
        simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_cos"] = attn_weights_v_cos

        return attn, attn_weights
    
    def simple_attn_baseline_infer(
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx = None,
        tgt_len_mod = None,
    ):
       
        src_len = k.size(1)
        tgt_len = q.size(1)

        # construct d x d intermediate matrices and normalization tensors
        kTv = torch.bmm(k.transpose(1, 2), v)

        norm = torch.sum(k.unsqueeze(-1), dim=1)

        # final attn calculations
        attn_weights = torch.bmm(q, kTv)

        # normalization calculation
        prob_norm = torch.bmm(q, norm)
        prob_norm = torch.clamp_min(prob_norm, 0.1)

        attn_probs = attn_weights / prob_norm

        attn = attn_probs
        attn = attn.transpose(0, 1).contiguous().view(1, 1, self.embed_dim)
        attn = self.out_proj(attn)
        
        return attn, attn_weights
    
    def simple_attn_cache_infer( 
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx = None,
        tgt_len_mod = None,
    ):
        old_src_idx = simul_attn_chkpts["old_indices"]["src"]
        old_tgt_idx = simul_attn_chkpts["old_indices"]["tgt"]

        tgt_idx = incremental_state["steps"]["tgt"]

        norm_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_cos"]

        old_src = tgt_idx - 1

        # build normalization vectors if necessary
        if norm_old is not None:
            norm = norm_old + k.transpose(1, 2)
        else:
            norm = k.transpose(1, 2)

        # build out d x d intermediate matrix
        old_kTv = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_cos"]

        if old_kTv is not None:
            kTv = torch.bmm(k.transpose(1, 2), v)
            kTv = old_kTv + kTv
        else:
            kTv = torch.bmm(k.transpose(1, 2), v)

        # final attention calculation
        attn_weights = torch.bmm(q, kTv)

        # normalization calculation
        prob_norm = torch.bmm(q, norm)
        prob_norm = torch.clamp_min(prob_norm, 0.1)

        attn_probs = attn_weights / prob_norm

        attn = attn_probs

        attn = attn.transpose(0, 1).contiguous().view(1, 1, self.embed_dim)
        attn = self.out_proj(attn)

        simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_cos"] = norm
        simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_cos"] = kTv

        return attn, attn_weights

    def cosformer_attn_train_and_infer(
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        src_len = None,
        bsz = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx = None,
        is_tpu = False,
    ):

        assert v is not None
        assert q is not None
        assert k is not None
        
        if key_padding_mask is not None:
            key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
            k = k.view(bsz, self.num_heads, src_len, list(k.shape)[2])          
            k = k.masked_fill(key_pad_mask_unsqueeze, 0)
            k = k.view(bsz*self.num_heads, src_len, list(k.shape)[3])          

        q_sin_init = q
        q_cos_init = q
        k_sin_init = k
        k_cos_init = k
        q_sin = torch.zeros(q.shape, device=q.device)
        q_cos = torch.zeros(q.shape, device=q.device)
        k_sin = torch.zeros(k.shape, device=k.device)
        k_cos = torch.zeros(k.shape, device=k.device)
        idx = torch.arange(1, src_len + 1, device = k.device)
        norm_sin = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        norm_cos = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        old_tgt = 0
        
        tgt_len = list(q.shape)[1]

        # use this for src length thresholding, 10% of max src_len step size is used to avoid bias towards early end of sentence characters
        src_len_p = math.ceil((src_len + self.max_src_len_step_size / 10) / self.max_src_len_step_size)

        if incremental_state is not None:
            src_idx = incremental_state["steps"]["src"]
            tgt_idx = incremental_state["steps"]["tgt"]

        # should strongly consider changing 2*src_len to 768 or some other constant
        if simul_attn_chkpts is not None:
            src_idx = incremental_state["steps"]["src"]
            tgt_idx = incremental_state["steps"]["tgt"]
            old_src_idx = simul_attn_chkpts["old_indices"]["src"]
            old_tgt_idx = simul_attn_chkpts["old_indices"]["tgt"]
           
            if self.cosformer_attn_enable:
                q_sin = torch.mul(q_sin_init, math.sin((math.pi*tgt_idx)/(2* src_len_p * self.max_src_len_step_size)))
                q_cos = torch.mul(q_cos_init, math.cos((math.pi*tgt_idx)/(2* src_len_p * self.max_src_len_step_size)))

            elif self.cosformer_expt_attn_enable:
                q_sin = torch.mul(q_sin_init, math.sin((math.pi*(1 - math.exp(-1 * tgt_idx)))/2))
                q_cos = torch.mul(q_cos_init, math.cos((math.pi*(1 - math.exp(-1 * tgt_idx)))/2))
            
            sin_tr = simul_attn_chkpts["sin_tr"]
            cos_tr = simul_attn_chkpts["cos_tr"]
            
            k_sin_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_sin"]
            k_cos_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_cos"]
            norm_sin_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_sin"]
            norm_cos_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_cos"]

            # build key transforms if necessary
            if k_sin_old is not None and k_cos_old is not None:
                old_tgt = list(k_sin_old.shape)[1]
                if old_tgt == tgt_idx:
                    k_sin = k_sin_old
                    k_cos = k_cos_old
                else:
                    sin_tr = sin_tr[old_tgt:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    cos_tr = cos_tr[old_tgt:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    k_sin = torch.cat((k_sin_old, torch.matmul(k_sin_init[:, old_tgt:, :].unsqueeze(-1), sin_tr).squeeze(-1)), dim=1)
                    k_cos = torch.cat((k_cos_old, torch.matmul(k_cos_init[:, old_tgt:, :].unsqueeze(-1), cos_tr).squeeze(-1)), dim=1)
            else:
                sin_tr = sin_tr[:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                cos_tr = cos_tr[:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr).squeeze(-1)
                k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr).squeeze(-1)

            # build normalization vectors if necessary
            if norm_sin_old is not None and norm_cos_old is not None:
                if old_tgt == tgt_idx:
                    norm_sin = norm_sin_old
                    norm_cos = norm_cos_old
                else:
                    norm_sin = norm_sin_old + torch.sum(k_sin.unsqueeze(-1)[:, old_tgt:, :], dim=1)
                    norm_cos = norm_cos_old + torch.sum(k_cos.unsqueeze(-1)[:, old_tgt:, :], dim=1)
            else:
                norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
                norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)

        else:
            if self.cosformer_attn_enable:
                loop_limit = math.ceil((src_len + self.max_src_len_step_size / 10) / self.max_src_len_step_size)
                
                step = self.max_src_len_step_size
                temp_idx = torch.zeros(src_len, device=k.device)
                bound_l = 0
                for i in range(loop_limit):
                    bound_h = min(math.ceil((i + 1) * step - step/10), src_len)
                    temp_idx[bound_l:bound_h] = idx[bound_l:bound_h] / ((i + 1) * step)
                    bound_l = bound_h
                
                sin_tr = torch.sin((math.pi / 2) * temp_idx)
                cos_tr = torch.cos((math.pi / 2) * temp_idx)
            elif self.cosformer_expt_attn_enable:
                sin_tr = torch.sin((math.pi*(1 - torch.exp(-1 * idx)))/2)
                cos_tr = torch.cos((math.pi*(1 - torch.exp(-1 * idx)))/2)
            
            sin_tr = sin_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            cos_tr = cos_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            
            if incremental_state is not None:
                q_sin = torch.mul(q_sin_init.unsqueeze(-1), sin_tr[0, tgt_idx - 1, 0, 0]).squeeze(-1)
                q_cos = torch.mul(q_cos_init.unsqueeze(-1), cos_tr[0, tgt_idx - 1, 0, 0]).squeeze(-1)
            else:    
                q_sin = torch.matmul(q_sin_init.unsqueeze(-1), sin_tr).squeeze(-1)
                q_cos = torch.matmul(q_cos_init.unsqueeze(-1), cos_tr).squeeze(-1)
            
            k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr).squeeze(-1)
            k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr).squeeze(-1)
            
            if incremental_state is not None:
                norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
                norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)
            else:
                norm_sin = torch.cumsum(k_sin, dim=1).transpose(1, 2)
                norm_cos = torch.cumsum(k_cos, dim=1).transpose(1, 2)

        # build out d x d intermediate matrix
        if simul_attn_chkpts is not None:
            old_attn_weights_v_sin = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_sin"]
            old_attn_weights_v_cos = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_cos"]

            if old_attn_weights_v_sin is not None and old_attn_weights_v_cos is not None:
                if old_tgt == tgt_idx:
                    attn_weights_v_sin = old_attn_weights_v_sin
                    attn_weights_v_cos = old_attn_weights_v_cos
                else:
                    attn_weights_v_sin = old_attn_weights_v_sin + torch.bmm(k_sin[:, old_tgt:, :].transpose(1, 2), v[:, old_tgt:, :]) 
                    attn_weights_v_cos = old_attn_weights_v_cos + torch.bmm(k_cos[:, old_tgt:, :].transpose(1, 2), v[:, old_tgt:, :])
            else:
                attn_weights_v_sin = torch.bmm(k_sin.transpose(1, 2), v)
                attn_weights_v_cos = torch.bmm(k_cos.transpose(1, 2), v)

        else:
            # used for cosformer quadratic attention
            if incremental_state is not None:
                attn_weights_v_sin = torch.bmm(k_sin.transpose(1, 2), v)
                attn_weights_v_cos = torch.bmm(k_cos.transpose(1, 2), v)
            
            # training only behavior, non-linearized with respect to number of samples
            else:
                attn_weights_v_sin = torch.bmm(q_sin, k_sin.transpose(1, 2))
                attn_weights_v_cos = torch.bmm(q_cos, k_cos.transpose(1, 2))
       
        # remaining computation
        if incremental_state is not None:
            attn_weights_sin = torch.bmm(q_sin, attn_weights_v_sin)
            attn_weights_cos = torch.bmm(q_cos, attn_weights_v_cos)
            attn_weights = attn_weights_sin + attn_weights_cos

            # expanding normalizing vector to 768, accounting for size
            if self.enable_norm_stretch_factor:
                norm_stretch_factor = src_len_p * self.max_src_len_step_size / list(k.shape)[1]
            else:
                norm_stretch_factor = 1

            prob_norm_sin = torch.bmm(q_sin, norm_stretch_factor * norm_sin)
            prob_norm_cos = torch.bmm(q_cos, norm_stretch_factor * norm_cos)
            prob_norm = prob_norm_sin + prob_norm_cos

            prob_norm.expand(list(prob_norm.shape)[0], list(prob_norm.shape)[1], list(attn_weights.shape)[2])
            prob_norm = torch.clamp_min(prob_norm, 0.1)

            attn_probs = attn_weights / prob_norm

            attn = attn_probs

            attn = attn.transpose(0, 1).contiguous().view(1, bsz, self.embed_dim)
            attn = self.out_proj(attn)

            if simul_attn_chkpts is not None:
                simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_sin"] = norm_sin
                simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_cos"] = norm_cos
                simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_sin"] = k_sin
                simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_cos"] = k_cos
                simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_sin"] = attn_weights_v_sin
                simul_attn_chkpts["layers"][layer_idx]["self_attn"]["kTv_cos"] = attn_weights_v_cos

            return attn, attn_weights

        # training enabled for n x n masking
        else:
            attn_weights = attn_weights_v_sin + attn_weights_v_cos
           
            if attn_mask is not None:
                attn_mask_bool = attn_mask.to(torch.bool)
                attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
           
            attn_weights = self.dropout_module(attn_weights)
            attn_probs = torch.bmm(attn_weights, v)

            # expanding normalizing vector to 768, accounting for size
            if self.enable_norm_stretch_factor:
                norm_stretch_factor = src_len_p * self.max_src_len_step_size / list(k.shape)[1]
            else:
                norm_stretch_factor = 1

            # section to try and replicate casual relationship in normalization
            
            prob_norm_sin = torch.bmm(q_sin, norm_stretch_factor * norm_sin)
            prob_norm_cos = torch.bmm(q_cos, norm_stretch_factor * norm_cos)
            prob_norm = prob_norm_sin + prob_norm_cos

            prob_norm = torch.diagonal(prob_norm, dim1=1, dim2=2).unsqueeze(-1)
            
            #prob_norm.expand(list(prob_norm.shape)[0], list(prob_norm.shape)[1], list(attn_weights.shape)[2])
            prob_norm = torch.clamp_min(prob_norm, 0.1)
            
            attn = attn_weights / prob_norm

            #attn_weights_float = attn_weights.type(torch.float32)
            #denom = torch.clamp_min(norm_stretch_factor * attn_weights_float.sum(dim=-1, keepdim=True), 0.1)
            #attn_weights_float = attn_weights_float / denom
            
            #attn = torch.bmm(attn_probs, v)

            attn = attn.transpose(0, 1).contiguous().view(src_len, bsz, self.embed_dim)
            attn = self.out_proj(attn)

            return attn, attn_weights
   
    # structured slightly differently, will go back later and make it step by step instead of grouped
    # into training and inference
    def combin_attn_train_and_infer(
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        src_len = None,
        bsz = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx = None,
        is_tpue = False,
    ):
        
        assert v is not None
        assert q is not None
        assert k is not None
        
        
        if key_padding_mask is not None:
            key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
            k = k.view(bsz, self.num_heads, src_len, list(k.shape)[2])          
            k = k.masked_fill(key_pad_mask_unsqueeze, 0)
            k = k.view(bsz*self.num_heads, src_len, list(k.shape)[3])          

        q_t = q
        q_tt = q
        k_t = k
        k_tt = k
        idx = torch.arange(1, src_len + 1, device = k.device)
        norm = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        norm_t = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        norm_tt = torch.zeros(list(k.shape)[0], list(k.shape)[2], 1, device=k.device)
        
        tgt_len = list(q.shape)[1]

        src_len_p = math.ceil((src_len + self.max_src_len_step_size / 10) / self.max_src_len_step_size)

        # similarity function is f(x) = 1 - (i - j)^2 or exponential alternative
        # QK^TV*f(x) = QK^TV - Q''K^TV + 2*Q'K'^TV - QK''^TV
        # activation is still relu
        
        if incremental_state is not None:
            src_idx = incremental_state["steps"]["src"]
            tgt_idx = incremental_state["steps"]["tgt"]
        
        if simul_attn_chkpts is not None:
            src_idx = incremental_state["steps"]["src"]
            tgt_idx = incremental_state["steps"]["tgt"]
            old_src_idx = simul_attn_chkpts["old_indices"]["src"]
            old_tgt_idx = simul_attn_chkpts["old_indices"]["tgt"]
            
            if self.combin_attn_enable:
                temp_idx = (tgt_idx + 0.1) / (src_len_p * self.max_src_len_step_size)
                q_t = torch.mul(q, temp_idx)
                q_tt = torch.mul(q, pow(temp_idx, 2))
            elif self.combin_expt_attn_enable:
                q_t = torch.mul(q, math.exp(-1 * tgt_idx))
                q_tt = torch.mul(q, math.exp(-2 * tgt_idx))

            k_t_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_t"]
            k_tt_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_tt"]
            norm_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm"]
            norm_t_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_t"]
            norm_tt_old = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_tt"]
            j_tr = simul_attn_chkpts["j_tr"]
            j_ttr = simul_attn_chkpts["j_ttr"]

            # build key transforms if necessary
            if k_t_old is not None and k_tt_old is not None:
                assert k_t_old.shape == k_tt_old.shape
                old_tgt = list(k_t_old.shape)[1]
                if old_tgt == tgt_idx:
                    k_t = k_t_old
                    k_tt = k_tt_old
                else:
                    j_tr = j_tr[old_tgt:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    j_ttr = j_ttr[old_tgt:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    k_t = torch.cat((k_t_old, torch.matmul(k_t[:, old_tgt:, :].unsqueeze(-1), j_tr).squeeze(-1)), dim=1)
                    k_tt = torch.cat((k_tt_old, torch.matmul(k_tt[:, old_tgt:, :].unsqueeze(-1), j_ttr).squeeze(-1)), dim=1)
            else:
                j_tr = j_tr[:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                j_ttr = j_ttr[:tgt_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                k_t = torch.matmul(k_t.unsqueeze(-1), j_tr).squeeze(-1)
                k_tt = torch.matmul(k_tt.unsqueeze(-1), j_ttr).squeeze(-1)

            # build normalization vectors if necessary
            if norm_old is not None and norm_t_old is not None and norm_tt_old is not None:
                assert norm_old.shape == norm_t_old.shape
                assert norm_old.shape == norm_tt_old.shape
                if old_tgt == tgt_idx:
                    norm = norm_old
                    norm_t = norm_t_old
                    norm_tt = norm_tt_old
                else:
                    norm = norm + torch.sum(k.unsqueeze(-1)[:, old_tgt:, :], dim=1)
                    norm_t = norm_t_old + torch.sum(k_t.unsqueeze(-1)[:, old_tgt:, :], dim=1)
                    norm_tt = norm_tt_old + torch.sum(k_tt.unsqueeze(-1)[:, old_tgt:, :], dim=1)
            else:
                norm = torch.sum(k.unsqueeze(-1), dim=1)
                norm_t = torch.sum(k_t.unsqueeze(-1), dim=1)
                norm_tt = torch.sum(k_tt.unsqueeze(-1), dim=1)
            

            # build out d x d intermediate matrix
            old_attn_weights = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["attn_weights"]
            old_attn_weights_kt = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["attn_weights_kt"]
            old_attn_weights_ktt = simul_attn_chkpts["layers"][layer_idx]["self_attn"]["attn_weights_ktt"]

            if old_attn_weights is not None and old_attn_weights_kt is not None and old_attn_weights_ktt is not None:
                assert old_attn_weights.shape == old_attn_weights_kt.shape
                assert old_attn_weights.shape == old_attn_weights_ktt.shape
                if old_tgt == tgt_idx:
                    attn_weights = old_attn_weights
                    attn_weights_kt = old_attn_weights_kt
                    attn_weights_ktt = old_attn_weights_ktt
                else:
                    attn_weights = old_attn_weights + torch.bmm(k[:, old_tgt:, :].transpose(1, 2), v[:, old_tgt:, :])
                    attn_weights_kt = old_attn_weights_kt + torch.bmm(k_t[:, old_tgt:, :].transpose(1, 2), v[:, old_tgt:, :])
                    attn_weights_ktt = old_attn_weights_ktt + torch.bmm(k_tt[:, old_tgt:, :].transpose(1, 2), v[:, old_tgt:, :])
            else:
                attn_weights = torch.bmm(k.transpose(1, 2), v)
                attn_weights_kt = torch.bmm(k_t.transpose(1, 2), v)
                attn_weights_ktt = torch.bmm(k_tt.transpose(1, 2), v)
            
            attn_weights_f = torch.bmm(q, attn_weights)
            attn_weights_f_t = torch.bmm(q_t, attn_weights_kt)
            attn_weights_f_qtt = torch.bmm(q_tt, attn_weights)
            attn_weights_f_ktt = torch.bmm(q, attn_weights_ktt)
            attn_weights_f_sum = attn_weights_f - attn_weights_f_qtt + 2 * attn_weights_f_t - attn_weights_f_ktt

            # expanding normalizing vector to 768, accounting for size
            if self.enable_norm_stretch_factor:
                norm_stretch_factor = src_len_p * self.max_src_len_step_size / list(k.shape)[1]
            else:
                norm_stretch_factor = 1

            prob_norm_f = torch.bmm(q, norm)
            prob_norm_f_t = torch.bmm(q_t, norm_t)
            prob_norm_f_qtt = torch.bmm(q_tt, norm)
            prob_norm_f_ktt = torch.bmm(q, norm_tt)

            prob_norm = norm_stretch_factor * (prob_norm_f - prob_norm_f_qtt + 2 * prob_norm_f_t - prob_norm_f_ktt)
            prob_norm = torch.clamp_min(prob_norm, 0.1)

            attn = attn_weights_f_sum / prob_norm
          
            attn = attn.transpose(0, 1).contiguous().view(1, bsz, self.embed_dim)
            attn = self.out_proj(attn)

            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_t"] = k_t
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["k_tt"] = k_tt
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm"] = norm
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_t"] = norm_t
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["norm_tt"] = norm_tt
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["attn_weights"] = attn_weights
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["attn_weights_kt"] = attn_weights_kt
            simul_attn_chkpts["layers"][layer_idx]["self_attn"]["attn_weights_ktt"] = attn_weights_ktt

            return attn, attn_weights_f_sum

        else:

            if self.combin_attn_enable:
                loop_limit = math.ceil((src_len + self.max_src_len_step_size / 10) / self.max_src_len_step_size)
                
                step = self.max_src_len_step_size
                temp_idx = torch.zeros(src_len) 
                bound_l = 0
                for i in range(loop_limit):
                    bound_h = min(math.ceil((i + 1) * step - step/10), src_len)
                    temp_idx[bound_l:bound_h] = (idx[bound_l:bound_h] + 0.1) / ((i + 1) * step)
                    bound_l = bound_h
                
                i_tr = temp_idx
                i_ttr = torch.square(temp_idx)
                j_tr = temp_idx
                j_ttr = torch.square(temp_idx)
            
            elif self.combin_expt_attn_enable:
                i_tr = torch.exp(-1 * idx)
                i_ttr = torch.exp(-2 * idx)
                j_tr = torch.exp(-1 * idx)
                j_ttr = torch.exp(-2 * idx)

            i_tr = i_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            i_ttr = i_ttr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            j_tr = j_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            j_ttr = j_ttr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
           
            if incremental_state is not None:
                q_t = torch.mul(q_t.unsqueeze(-1), i_tr[0, tgt_idx - 1, 0, 0]).squeeze(-1)
                q_tt = torch.mul(q_tt.unsqueeze(-1), i_ttr[0, tgt_idx - 1, 0, 0]).squeeze(-1)
            else:
                q_t = torch.matmul(q_t.unsqueeze(-1), i_tr).squeeze(-1)
                q_tt = torch.matmul(q_tt.unsqueeze(-1), i_ttr).squeeze(-1)
            
            k_t = torch.matmul(k_t.unsqueeze(-1), j_tr).squeeze(-1)
            k_tt = torch.matmul(k_tt.unsqueeze(-1), j_ttr).squeeze(-1)
            
            if incremental_state is not None:
                norm = torch.sum(k.unsqueeze(-1), dim=1)
                norm_t = torch.sum(k_t.unsqueeze(-1), dim=1)
                norm_tt = torch.sum(k_tt.unsqueeze(-1), dim=1)
            else:
                norm = torch.cumsum(k, dim=1).transpose(1, 2)
                norm_t = torch.cumsum(k_t, dim=1).transpose(1, 2)
                norm_tt = torch.cumsum(k_tt, dim=1).transpose(1, 2)

            if incremental_state is not None:
                attn_weights = torch.bmm(k.transpose(1, 2), v)
                attn_weights_qtt = torch.bmm(k.transpose(1, 2), v)
                attn_weights_t = torch.bmm(k_t.transpose(1, 2), v)
                attn_weights_ktt = torch.bmm(k_tt.transpose(1, 2), v)

                attn_weights = torch.bmm(q, attn_weights)
                attn_weights_qtt = torch.bmm(q_tt, attn_weights_qtt)
                attn_weights_t = torch.bmm(q_t, attn_weights_t)
                attn_weights_ktt = torch.bmm(q, attn_weights_ktt)

                attn_weights_sum = attn_weights - attn_weights_qtt + 2*attn_weights_t - attn_weights_ktt
                attn_weights = attn_weights_sum

            # non-linearized during training, compatible with attn mask
            else:
                attn_weights = torch.bmm(q, k.transpose(1, 2))
                attn_weights_qtt = torch.bmm(q_tt, k.transpose(1, 2))
                attn_weights_t = torch.bmm(q_t, k_t.transpose(1, 2))
                attn_weights_ktt = torch.bmm(q, k_tt.transpose(1, 2))
                attn_weights_sum = attn_weights - attn_weights_qtt + 2*attn_weights_t - attn_weights_ktt
                
                attn_weights = attn_weights_sum
                if attn_mask is not None:
                    attn_mask_bool = attn_mask.to(torch.bool)
                    attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
                
                attn_weights = self.dropout_module(attn_weights)
            
                attn_weights = torch.bmm(attn_weights, v)
            
            # expanding normalizing vector to 768, accounting for size
            if self.enable_norm_stretch_factor:
                norm_stretch_factor = src_len_p * self.max_src_len_step_size / list(k.shape)[1]
            else:
                norm_stretch_factor = 1

            prob_norm_f = torch.bmm(q, norm)
            prob_norm_f_t = torch.bmm(q_t, norm_t)
            prob_norm_f_qtt = torch.bmm(q_tt, norm)
            prob_norm_f_ktt = torch.bmm(q, norm_tt)
            
            if incremental_state is None:
                prob_norm_f = torch.diagonal(prob_norm_f, dim1=1, dim2=2).unsqueeze(-1)
                prob_norm_f_t = torch.diagonal(prob_norm_f_t, dim1=1, dim2=2).unsqueeze(-1)
                prob_norm_f_qtt = torch.diagonal(prob_norm_f_qtt, dim1=1, dim2=2).unsqueeze(-1)
                prob_norm_f_ktt = torch.diagonal(prob_norm_f_ktt, dim1=1, dim2=2).unsqueeze(-1)

            prob_norm = norm_stretch_factor * (prob_norm_f - prob_norm_f_qtt + 2 * prob_norm_f_t - prob_norm_f_ktt)
            prob_norm = torch.clamp_min(prob_norm, 0.1)

            attn = attn_weights / prob_norm

            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
            attn = self.out_proj(attn)

            return attn, attn_weights_sum
   
    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        layer_idx: int = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        tgt_len_mod = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """


        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        
        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            and not self.skip_embed_dim_check
            and not self.simple_attention
            and not self.cosformer_attn_enable
            and not self.combin_attn_enable
            and not self.combin_expt_attn_enable
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        
        if self.simple_attention or self.cosformer_attn_enable or self.cosformer_expt_attn_enable or self.combin_attn_enable or self.combin_expt_attn_enable and self.self_attention:
            #q *= tgt_len**-0.5
            q = F.relu(q)
            k = F.relu(k)
        else:
            q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        #print(key_padding_mask is None)
        #print(k.size())
        if self.cosformer_attn_enable and self.self_attention:
            if incremental_state is not None:
                if simul_attn_chkpts is not None:
                    return self.cosformer_attn_cache_infer(q, k, v, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx, tgt_len_mod)
        
        if self.simple_attention and self.self_attention:
            if incremental_state is not None:
                if simul_attn_chkpts is not None:
                    return self.simple_attn_cache_infer(q, k, v, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx, tgt_len_mod)

    
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        if self.cosformer_attn_enable and self.self_attention:
            if incremental_state is not None:
                #if simul_attn_chkpts is not None:
                #    return self.cosformer_attn_cache_infer(q, k, v, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx, tgt_len_mod)
                #else:
                return self.cosformer_attn_baseline_infer(q, k, v, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, tgt_len_mod)
            else:
                return self.cosformer_attn_baseline_train(q, k, v, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts)

        if self.simple_attention and self.self_attention:
            if incremental_state is not None:
                return self.simple_attn_baseline_infer(q, k, v, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, tgt_len_mod)


        # quick test for small similarity function addition
        if self.shortened_expt_simil and self.self_attention:
            idx = torch.arange(1, src_len + 1, device=k.device)
            if incremental_state is not None:
                tgt_idx = src_len - 1
            else:
                tgt_idx = 0
            i_tr = torch.exp(-1/128 * idx[tgt_idx:src_len])
            j_tr = torch.exp(1/128 * idx[:src_len])
            
            i_tr = i_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            j_tr = j_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            q = torch.matmul(q.unsqueeze(-1), i_tr).squeeze(-1)
            k = torch.matmul(k.unsqueeze(-1), j_tr).squeeze(-1)


        # cosFormer implementation alongside some alternative decomposable similarity functions 
        #if self.cosformer_attn_enable or self.cosformer_expt_attn_enable and self.self_attention:
        #    return self.cosformer_attn_train_and_infer(q, k, v, src_len, bsz, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx, is_tpu)
        #elif self.combin_attn_enable or self.combin_expt_attn_enable and self.self_attention:
        #    return self.combin_attn_train_and_infer(q, k, v, src_len, bsz, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx, is_tpu)

        #if not self.self_attention:
        #    print(f"Sizing information of interest: bsz {q.size(0)}, target {q.size(1)}, source {k.size(1)}")
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)

            if self.simple_attention and self.self_attention:
                attn_mask_bool = attn_mask.to(torch.bool)
                attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
            else:
                attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
                if self.simple_attention and self.self_attention:
                    attn_weights = attn_weights.masked_fill(key_pad_mask_unsqueeze, 0)
                else:
                    attn_weights = attn_weights.masked_fill(key_pad_mask_unsqueeze, float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                if self.simple_attention and self.self_attention:
                    attn_weights = attn_weights.masked_fill(key_padding_mask, 0)
                else:
                    attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        if self.simple_attention and self.self_attention:
            attn_weights_float = attn_weights.type(torch.float32)
            denom = torch.clamp_min(attn_weights_float.sum(dim=-1, keepdim=True), 0.1)
            #print(f"Denom: {denom}", flush=True)
            attn_weights_float = attn_weights_float / denom
        else:
            attn_weights_float = utils.softmax(
                attn_weights, dim=-1, onnx_trace=self.onnx_trace
            )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]
            state_dict[key] = value

