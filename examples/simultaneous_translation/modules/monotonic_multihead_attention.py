# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from examples.simultaneous_translation.utils.p_choose_strategy import (
    learnable_p_choose,
    waitk_p_choose
)

from examples.simultaneous_translation.utils.monotonic_attention import (
    expected_alignment_from_p_choose,
    expected_soft_attention,
    mass_preservation,
)
from fairseq.modules import MultiheadAttention

from . import register_monotonic_attention
from typing import Dict, Optional


@register_monotonic_attention("hard_aligned")
class MonotonicAttention(MultiheadAttention):
    """
    Abstract class of monotonic attentions
    """
    k_in_proj: Dict[str, nn.Linear]
    q_in_proj: Dict[str, nn.Linear]

    def __init__(self, args):
        super().__init__(
            embed_dim=args.decoder_embed_dim,
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            #cosformer_attn_enable=args.cosformer_attn_enable,
            simple_attention=args.simple_attention,
        )

        self.soft_attention = False

        self.simple_cross_attn = args.simple_cross_attn
        self.vanilla_cross = args.vanilla_cross
       
        #print(f"Quick configs of interest, simple_cross {self.simple_cross_attn}, vanilla_cross {self.vanilla_cross}")

        #if self.simple_cross_attn is None:
        #    self.simple_cross_attn = False
        #if self.vanilla_cross is None:
        #    self.vanilla_cross = False
        if self.cosformer_attn_enable is None:
            self.cosformer_attn_enable = False
        
        self.eps = getattr(args, "attention_eps", 1e-6)
        self.mass_preservation = getattr(args, "mass_preservation", True)

        self.noise_type = args.noise_type
        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var

        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = (
            nn.Parameter(self.energy_bias_init * torch.ones([1]))
            if args.energy_bias is True
            else 0
        )

        self.k_in_proj = {"monotonic": self.k_proj}
        self.q_in_proj = {"monotonic": self.q_proj}
        self.chunk_size = None

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--no-mass-preservation', action="store_false",
                            dest="mass_preservation",
                            help='Do not stay on the last token when decoding')
        parser.add_argument('--mass-preservation', action="store_true",
                            dest="mass_preservation",
                            help='Stay on the last token when decoding')
        parser.set_defaults(mass_preservation=True)
        parser.add_argument('--noise-var', type=float, default=1.0,
                            help='Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0,
                            help='Mean of discretness noise')
        parser.add_argument('--noise-type', type=str, default="flat",
                            help='Type of discretness noise')
        parser.add_argument('--energy-bias', action="store_true",
                            default=False,
                            help='Bias for energy')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0,
                            help='Initial value of the bias for energy')
        parser.add_argument('--attention-eps', type=float, default=1e-6,
                            help='Epsilon when calculating expected attention')
        parser.add_argument('--vanilla-cross', action="store_true", default=False,
                            help='Remove linearizable behavior for cross-attention.')

    def energy_from_qk(
        self,
        query: Tensor,
        key: Tensor,
        energy_type: str,
        key_padding_mask: Optional[Tensor] = None,
        bias: int = 0
    ):
        """
        Compute energy from query and key
        q_func_value is a tuple looks like
        (q_proj_func, q_tensor)
        q_tensor size: bsz, tgt_len, emb_dim
        k_tensor size: bsz, src_len, emb_dim
        key_padding_mask size: bsz, src_len
        attn_mask: bsz, src_len
        """

        length, bsz, _ = query.size()
        q = self.q_in_proj[energy_type].forward(query)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        q = q * self.scaling
        length, bsz, _ = key.size()
        k = self.k_in_proj[energy_type].forward(key)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        energy = torch.bmm(q, k.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            energy = energy.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                -1e4 if energy.dtype == torch.float16 else -1e8
            )

        return energy

    def p_choose_from_qk(self, query, key, key_padding_mask, incremental_state=None):
        monotonic_energy = self.energy_from_qk(
            query,
            key,
            "monotonic",
            key_padding_mask=key_padding_mask,
            bias=self.energy_bias,
        )

        p_choose = learnable_p_choose(
            monotonic_energy,
            self.noise_mean,
            self.noise_var,
            self.training
        )
        return p_choose

    def p_choose(self, query, key, key_padding_mask, incremental_state=None):
        return self.p_choose_from_qk(query, key, key_padding_mask, incremental_state=incremental_state)

    def monotonic_attention_process_infer(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ):
        """
        Monotonic attention at inference time
        Notice that this function is designed for simuleval not sequence_generator
        """
        assert query is not None
        assert key is not None

        if query.size(1) != 1:
            raise RuntimeError(
                "Simultaneous translation models don't support batch decoding."
            )
        # 1. compute stepwise probability
        p_choose = self.p_choose(
            query, key, None, incremental_state
        ).squeeze(1)

        # 2. Compute the alpha
        src_len = key.size(0)
        # Maximum steps allows in this iteration
        max_steps = src_len - 1 if self.mass_preservation else src_len
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        # Step for each head
        monotonic_step = monotonic_cache.get(
            'head_step',
            p_choose.new_zeros(self.num_heads, 1).long()
        )
        assert monotonic_step is not None
        finish_read = monotonic_step.eq(max_steps)
        p_choose_i = torch.tensor(1)

        while finish_read.sum().item() < self.num_heads:
            # p_choose: self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: self.num_heads, 1
            p_choose_i = (
                p_choose.gather(
                    1,
                    monotonic_step
                    .clamp(0, src_len - 1),
                )
            )

            read_one_step = (
                (p_choose_i < 0.5)
                .type_as(monotonic_step)
                .masked_fill(finish_read, 0)
            )
            # self.num_heads x 1
            # sample actions on unfinished seq
            # 0 means stay, finish reading
            # 1 means leave, continue reading

            monotonic_step += read_one_step

            finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

        # p_choose at last steps
        p_choose_i = (
            p_choose.gather(
                1,
                monotonic_step
                .clamp(0, src_len - 1),
            )
        )

        monotonic_cache["head_step"] = monotonic_step
        # Whether a head is looking for new input
        monotonic_cache["head_read"] = (
            monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
        )
        self._set_monotonic_buffer(incremental_state, monotonic_cache)

        # 2. Update alpha
        alpha = (
            p_choose
            .new_zeros([self.num_heads, src_len])
            .scatter(
                1,
                (monotonic_step)
                .view(self.num_heads, 1).clamp(0, src_len - 1),
                1
            )
        )

        if not self.mass_preservation:
            alpha = alpha.masked_fill(
                (monotonic_step == max_steps)
                .view(self.num_heads, 1),
                0
            )

        # 4. Compute Beta
        if self.soft_attention:
            # monotonic_step = monotonic_step.t()
            beta_mask = torch.arange(src_len, device=alpha.device).expand_as(alpha).gt(monotonic_step).unsqueeze(1)
            #print(beta_mask)
            #print(beta_mask.shape)
            # If it's soft attention just do softmax on current context
            soft_energy = self.energy_from_qk(
                query,
                key,
                "soft"
            )
            beta = torch.nn.functional.softmax(
                soft_energy.masked_fill(
                    beta_mask,
                    -1e4 if soft_energy.dtype == torch.float16 else -1e8
                ), dim=-1
            )
            # It could happen that a head doesn't move at all
            beta = beta.masked_fill(monotonic_step.eq(0).unsqueeze(1), 0)
        else:
            # If it's hard attention just select the last state
            beta = alpha.view(self.num_heads, 1, src_len)  # bsz * head, tgt, src

        return p_choose, alpha, beta

    def monotonic_attention_process_train(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Calculating monotonic attention process for training
        Including:
            stepwise probability: p_choose
            expected hard alignment: alpha
            expected soft attention: beta
        """
        assert query is not None
        assert key is not None

        # 1. compute stepwise probability
        p_choose = self.p_choose(query, key, key_padding_mask)

        # 2. compute expected_alignment
        alpha = expected_alignment_from_p_choose(
            p_choose.float(),  # prevents latency loss from nan
            key_padding_mask,
            eps=self.eps,
        )

        if self.mass_preservation:
            alpha = mass_preservation(
                alpha, key_padding_mask
            )

        # 3. compute expected soft attention (soft aligned model only)
        if self.soft_attention:
            soft_energy = self.energy_from_qk(
                query,
                key,
                "soft",
                key_padding_mask=None,
            )

            beta = expected_soft_attention(
                alpha,
                soft_energy,
                padding_mask=key_padding_mask,
                chunk_size=self.chunk_size,
                eps=self.eps,
            )
        else:
            beta = alpha
            soft_energy = alpha

        return p_choose, alpha, beta, soft_energy

    def forward(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True, static_kv: bool = False, need_head_weights: bool = False,
        tgt_len_mod = None,
        layer_idx = None,
    ):
        """
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        """

        assert attn_mask is None
        assert query is not None
        assert key is not None
        assert value is not None

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)

        #print(f"Information of interest. bsz {bsz}, src_len {src_len}, tgt_len {tgt_len}")

        if key_padding_mask is not None:
            assert not key_padding_mask[:, 0].any(), (
                "Only right padding is supported."
            )
            key_padding_mask = (
                key_padding_mask
                .unsqueeze(1)
                .expand([bsz, self.num_heads, src_len])
                .contiguous()
                .view(-1, src_len)
            )
        
        tgt_len_mod = 0.6
        #if self.cosformer_attn_enable:
        if self.cosformer_attn_enable or self.simple_cross_attn and not self.vanilla_cross:
            if incremental_state is not None:
                if simul_attn_chkpts is not None:
                    return self.cosformer_attn_cache_infer(query, key, value, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, tgt_len_mod, energy_type="soft", layer_idx=layer_idx)
                else:
                    return self.cosformer_attn_baseline_infer(query, key, value, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, tgt_len_mod, energy_type="soft")
            else:
                return self.cosformer_attn_baseline_train(query, key, value, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, energy_type="soft")
        #if self.cosformer_attn_enable or self.cosformer_expt_attn_enable:
        #    return cosformer_attn_train_and_infer(query, key, value, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx)
        #if self.combin_attn_enable or self.combin_expt_attn_enable:
        #    return combin_attn_train_and_infer(query, key, value, key_padding_mask, attn_mask, incremental_state, simul_attn_chkpts, layer_idx)
       
        if incremental_state is not None:
            # Inference
            (
                p_choose, alpha, beta
            ) = self.monotonic_attention_process_infer(
                query, key, incremental_state
            )
            soft_energy = beta
        else:
            # Train
            (
                p_choose, alpha, beta, soft_energy
            ) = self.monotonic_attention_process_train(
                query, key, key_padding_mask
            )

        v = self.v_proj(value)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn = torch.bmm(beta.type_as(v), v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)
        
        #print(f"Some shapes of interest. k: {key.shape}, q: {query.shape}, v: {value.shape}, beta: {beta.shape}, attn: {attn.shape}")

        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, {
            "p_choose": p_choose,
            "alpha": alpha,
            "beta": beta,
        }

    def _get_monotonic_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        maybe_incremental_state = self.get_incremental_state(
            incremental_state,
            'monotonic',
        )
        if maybe_incremental_state is None:
            typed_empty_dict: Dict[str, Optional[Tensor]] = {}
            return typed_empty_dict
        else:
            return maybe_incremental_state

    def _set_monotonic_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], buffer: Dict[str, Optional[Tensor]]):
        self.set_incremental_state(
            incremental_state,
            'monotonic',
            buffer,
        )

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
        energy_type = None,
    ):
        
        # start of general behavior
        assert q is not None
        assert k is not None
        assert v is not None
        
        # 1. compute stepwise probability
        p_choose = self.p_choose(q, k, key_padding_mask)

        # 2. compute expected_alignment
        alpha = expected_alignment_from_p_choose(
            p_choose.float(),  # prevents latency loss from nan
            key_padding_mask,
            eps=self.eps,
        )

        if self.mass_preservation:
            alpha = mass_preservation(
                alpha, key_padding_mask
            )

        # prepping input tensors
        length, bsz, _ = q.size()
        q = self.q_in_proj[energy_type].forward(q)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        length, bsz, _ = k.size()
        k = self.k_in_proj[energy_type].forward(k)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        v = self.v_proj(v)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        src_len = k.size(1)
        tgt_len = q.size(1)
        
        q = F.relu(q)
        k = F.relu(k)
        # ending input tensor prep
        
        # implementation differs from typical key_padding_mask application, but this is useful later and should be fine
        if key_padding_mask is not None:
            key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(-1).to(torch.bool)
            k = k.masked_fill(key_pad_mask_unsqueeze, 0)
           
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
        idx = torch.arange(1, max_len + 1, device = k.device)
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

        #if self.cosformer_attn_enable:
        if not self.simple_cross_attn:
            print("made it to here")
            # query transforms
            q_sin = torch.matmul(q_sin_init.unsqueeze(-1), sin_tr_q).squeeze(-1)
            q_cos = torch.matmul(q_cos_init.unsqueeze(-1), cos_tr_q).squeeze(-1)
            
            # key transforms
            k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr_k).squeeze(-1)
            k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr_k).squeeze(-1)
        else:
            # creates equivalent ReLu based option that is linearized during training
            q_sin = 0.5 * q
            q_cos = 0.5 * q
            k_sin = k
            k_cos = k
        
        # limit is based on predecision ratio and wait-k lagging value
        start = self.waitk_lagging * self.pre_decision_ratio
        p_limit = min(start, src_len)
        old_limit = 0
        #old_kTv_sin = torch.zeros(bsz*self.num_heads, k.size(2), v.size(2), device=k.device)
        #old_kTv_cos = torch.zeros(bsz*self.num_heads, k.size(2), v.size(2), device=k.device)
        #old_norm_sin = torch.zeros(bsz*self.num_heads, 1, k.size(2), device=k.device)
        #old_norm_cos = torch.zeros(bsz*self.num_heads, 1, k.size(2), device=k.device)
        #tot_attn = torch.empty(bsz*self.num_heads, 0, k.size(2), device=k.device)

        # einsum based approach should be much faster, larger space complexity however
        # the below expression stores the d x d resulting matrices instead of adding them together, outer product notation
        kTv_sin_steps = torch.einsum('nld,nlm->nldm', k_sin, v)
        kTv_cos_steps = torch.einsum('nld,nlm->nldm', k_cos, v)

        # can't use simple torch.cumsum, must use a more dynamic solution
        #kTv_sin_cum = torch.cumsum(kTv_sin_steps, dim=1)
        #kTv_cos_cum = torch.cumsum(kTv_cos_steps, dim=1)

        kTv_sin_cum = torch.zeros(k.size(0), k.size(1), k.size(2), v.size(2), device=k.device)
        kTv_cos_cum = torch.zeros(k.size(0), k.size(1), k.size(2), v.size(2), device=k.device)
        norm_sin = torch.zeros(k.shape, device=k.device)
        norm_cos = torch.zeros(k.shape, device=k.device)
        loop_limit = math.ceil((src_len - start) / self.pre_decision_ratio)
        if loop_limit > 0:
            for i in range(loop_limit):
                kTv_sin_cum[:, old_limit:p_limit, :, :] = torch.sum(kTv_sin_steps[:, :p_limit, :, :], dim=1, keepdim=True)
                kTv_cos_cum[:, old_limit:p_limit, :, :] = torch.sum(kTv_cos_steps[:, :p_limit, :, :], dim=1, keepdim=True)
                norm_sin[:, old_limit:p_limit, :] = torch.sum(k_sin[:, :p_limit, :], dim=1, keepdim=True)
                norm_cos[:, old_limit:p_limit, :] = torch.sum(k_cos[:, :p_limit, :], dim=1, keepdim=True)
                old_limit = p_limit
                p_limit = min(p_limit + self.pre_decision_ratio, src_len)
        else:
            kTv_sin_cum = torch.sum(kTv_sin_steps, dim=1, keepdim=True)
            kTv_cos_cum = torch.sum(kTv_cos_steps, dim=1, keepdim=True)
            norm_sin = torch.sum(k_sin, dim=1, keepdim=True)
            norm_cos = torch.sum(k_cos, dim=1, keepdim=True)

        print(q_sin.shape, kTv_sin_cum.shape)

        attn_weights_sin = torch.einsum('ntd,nldm->ntm', q_sin, kTv_sin_cum)
        attn_weights_cos = torch.einsum('ntd,nldm->ntm', q_cos, kTv_cos_cum)
        attn_weights = attn_weights_sin + attn_weights_cos

        #print(q_sin.shape)
        #print(norm_sin.shape)
        prob_norm_sin = torch.bmm(q_sin, norm_sin.transpose(1, 2))
        prob_norm_cos = torch.bmm(q_cos, norm_cos.transpose(1, 2))
        prob_norm = prob_norm_sin + prob_norm_cos

        # necessary step to ensure proper normalization for most samples with non-trivial waitk
        if loop_limit > 0:
            start = self.waitk_lagging * self.pre_decision_ratio
            p_limit = min(start, src_len)
            old_limit = 0
            prob_norm_temp = torch.zeros(q.size(0), q.size(1), 1, device=k.device)
            for i in range(loop_limit):
                #print(prob_norm_temp[:, i, 0].shape, prob_norm[:, i, p_limit].shape)
                prob_norm_temp[:, i, 0] = prob_norm[:, i, p_limit]
                old_limit = p_limit
                p_limit = min(p_limit + self.pre_decision_ratio, src_len)
            if loop_limit < tgt_len:
                #print(prob_norm_temp[:, loop_limit - 1:tgt_len, 0].shape, prob_norm[:, loop_limit - 1:tgt_len, src_len - 1].shape)
                prob_norm_temp[:, loop_limit - 1:tgt_len, 0] = prob_norm[:, loop_limit - 1:tgt_len, src_len - 1]
            prob_norm = prob_norm_temp
        
        #prob_norm.expand(list(prob_norm.shape)[0], list(prob_norm.shape)[1], list(attn_weights.shape)[2])
        prob_norm = torch.clamp_min(prob_norm, 0.1)
       
        print(prob_norm_sin.shape, norm_sin.shape)
        print(attn_weights.shape, prob_norm.shape)
        print(src_len, tgt_len)

        attn = attn_weights / prob_norm
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        return attn, {
            "p_choose": p_choose,
            "alpha": alpha,
            "beta": None,
        }

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
        energy_type = None,
        layer_idx = None,
    ):
        
        """
        Monotonic attention at inference time
        Notice that this function is designed for simuleval not sequence_generator
        """

        if q.size(1) != 1:
            raise RuntimeError(
                "Simultaneous translation models don't support batch decoding."
            )
        # 1. compute stepwise probability
        p_choose = self.p_choose(
            q, k, None, incremental_state
        ).squeeze(1)

        # 2. Compute the alpha
        src_len = k.size(0)
        # Maximum steps allows in this iteration
        max_steps = src_len - 1 if self.mass_preservation else src_len
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        # Step for each head
        monotonic_step = monotonic_cache.get(
            'head_step',
            p_choose.new_zeros(self.num_heads, 1).long()
        )
        assert monotonic_step is not None
        finish_read = monotonic_step.eq(max_steps)
        p_choose_i = torch.tensor(1)

        while finish_read.sum().item() < self.num_heads:
            # p_choose: self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: self.num_heads, 1
            p_choose_i = (
                p_choose.gather(
                    1,
                    monotonic_step
                    .clamp(0, src_len - 1),
                )
            )

            read_one_step = (
                (p_choose_i < 0.5)
                .type_as(monotonic_step)
                .masked_fill(finish_read, 0)
            )
            # self.num_heads x 1
            # sample actions on unfinished seq
            # 0 means stay, finish reading
            # 1 means leave, continue reading

            monotonic_step += read_one_step

            finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

        # p_choose at last steps
        p_choose_i = (
            p_choose.gather(
                1,
                monotonic_step
                .clamp(0, src_len - 1),
            )
        )
        
#        # 2. Update alpha
#        alpha = (
#            p_choose
#            .new_zeros([self.num_heads, src_len])
#            .scatter(
#                1,
#                (monotonic_step)
#                .view(self.num_heads, 1).clamp(0, src_len - 1),
#                1
#            )
#        )
#
#        if not self.mass_preservation:
#            alpha = alpha.masked_fill(
#                (monotonic_step == max_steps)
#                .view(self.num_heads, 1),
#                0
#            )

        monotonic_cache["head_step"] = monotonic_step
        # Whether a head is looking for new input
        monotonic_cache["head_read"] = (
            monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
        )
        self._set_monotonic_buffer(incremental_state, monotonic_cache)
        # end monotonic inference behavior, onto more typical functionality
        
        #print(monotonic_step)
        #print(monotonic_step.shape)

        # start of general behavior
        assert q is not None
        assert k is not None
        assert v is not None

        # prepping input tensors
        src_len = k.size(0)

        length, bsz, _ = q.size()
        q = self.q_in_proj[energy_type].forward(q)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        length, bsz, _ = k.size()
        k = self.k_in_proj[energy_type].forward(k)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        v = self.v_proj(v)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        src_len = k.size(1)
        tgt_len = q.size(1)
        
        q = F.relu(q)
        k = F.relu(k)
        # ending input tensor prep
        
        # beginning of attention calculations
        src_idx = incremental_state["steps"]["src"]
        tgt_idx = incremental_state["steps"]["tgt"]
       
        src_len = k.size(1)
        tgt_len = q.size(1)

        q_sin_init = q
        q_cos_init = q
        k_sin_init = k
        k_cos_init = k
        q_sin = torch.zeros(q.shape, device=q.device)
        q_cos = torch.zeros(q.shape, device=q.device)
        k_sin = torch.zeros(k.shape, device=k.device)
        k_cos = torch.zeros(k.shape, device=k.device)
        
        idx = torch.arange(1, src_len + 1, device = k.device)
       
        tgt_len_p = src_len * self.tgt_len_mod

        # transform tensors
        #sin_tr_q = torch.sin((math.pi / 2) * tgt_idx / tgt_len_p))
        #cos_tr_q = torch.cos((math.pi / 2) * tgt_idx / tgt_len_p))
        
        #sin_tr_q = sin_tr_q.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #cos_tr_q = cos_tr_q.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        sin_tr = torch.sin((math.pi / 2) * (idx / src_len))
        cos_tr = torch.cos((math.pi / 2) * (idx / src_len))
        
        sin_tr = sin_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cos_tr = cos_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # fix later
        #if not self.simple_cross_attn:
        if self.cosformer_attn_enable:
            # query transforms
            q_sin = torch.mul(q_sin_init, math.sin((math.pi / 2) * (tgt_idx / tgt_len_p)))
            q_cos = torch.mul(q_cos_init, math.cos((math.pi / 2) * (tgt_idx / tgt_len_p)))
            
            # key transforms
            k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr).squeeze(-1)
            k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr).squeeze(-1)
        else:
            # creates equivalent ReLu based option that is linearized during training
            q_sin = 0.5 * q
            q_cos = 0.5 * q
            k_sin = k
            k_cos = k

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

        prob_norm.expand(list(prob_norm.shape)[0], list(prob_norm.shape)[1], list(attn_weights.shape)[2])
        prob_norm = torch.clamp_min(prob_norm, 0.1)

        attn_probs = attn_weights / prob_norm

        attn = attn_probs

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        return attn, {
            "p_choose": p_choose,
            "alpha": None,
            "beta": attn_weights,
        }

    def cosformer_attn_cache_infer(
        self,
        q,
        k: Optional[Tensor],
        v: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        tgt_len_mod = None,
        energy_type = None,
        layer_idx = None,
    ):

        """
        Monotonic attention at inference time
        Notice that this function is designed for simuleval not sequence_generator
        """

        if q.size(1) != 1:
            raise RuntimeError(
                "Simultaneous translation models don't support batch decoding."
            )
        # 1. compute stepwise probability
        p_choose = self.p_choose(
            q, k, None, incremental_state
        ).squeeze(1)

        # 2. Compute the alpha
        src_len = k.size(0)
        # Maximum steps allows in this iteration
        max_steps = src_len - 1 if self.mass_preservation else src_len
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        # Step for each head
        monotonic_step = monotonic_cache.get(
            'head_step',
            p_choose.new_zeros(self.num_heads, 1).long()
        )
        assert monotonic_step is not None
        finish_read = monotonic_step.eq(max_steps)
        p_choose_i = torch.tensor(1)

        while finish_read.sum().item() < self.num_heads:
            # p_choose: self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: self.num_heads, 1
            p_choose_i = (
                p_choose.gather(
                    1,
                    monotonic_step
                    .clamp(0, src_len - 1),
                )
            )

            read_one_step = (
                (p_choose_i < 0.5)
                .type_as(monotonic_step)
                .masked_fill(finish_read, 0)
            )
            # self.num_heads x 1
            # sample actions on unfinished seq
            # 0 means stay, finish reading
            # 1 means leave, continue reading

            monotonic_step += read_one_step

            finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

#        # p_choose at last steps
        p_choose_i = (
            p_choose.gather(
                1,
                monotonic_step
                .clamp(0, src_len - 1),
            )
        )
#        
#        # 2. Update alpha
#        alpha = (
#            p_choose
#            .new_zeros([self.num_heads, src_len])
#            .scatter(
#                1,
#                (monotonic_step)
#                .view(self.num_heads, 1).clamp(0, src_len - 1),
#                1
#            )
#        )
#
#        if not self.mass_preservation:
#            alpha = alpha.masked_fill(
#                (monotonic_step == max_steps)
#                .view(self.num_heads, 1),
#                0
#            )
#

        monotonic_cache["head_step"] = monotonic_step
        # Whether a head is looking for new input
        monotonic_cache["head_read"] = (
            monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
        )
        self._set_monotonic_buffer(incremental_state, monotonic_cache)
#        # end monotonic inference behavior, onto more typical functionality
        
        # start of general behavior
        assert q is not None
        assert k is not None
        assert v is not None

        # prepping input tensors
        src_len = k.size(0)
        
        length, bsz, _ = q.size()
        q = self.q_in_proj[energy_type].forward(q)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        length, bsz, _ = k.size()
        if not simul_attn_chkpts["save_k_v_cross"]:
            k = self.k_in_proj[energy_type].forward(k)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        length, bsz, _ = v.size()
        if not simul_attn_chkpts["save_k_v_cross"]:
            v = self.v_proj(v)
        else:
            v = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["v"]
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        q = F.relu(q)
        if not simul_attn_chkpts["save_k_v_cross"]:
            k = F.relu(k)
        # ending input tensor prep
        
        # beginning of attention calculations
        src_idx = incremental_state["steps"]["src"]
        tgt_idx = incremental_state["steps"]["tgt"]
      
        tgt_len = q.size(1)

        tgt_len_p = src_len * tgt_len_mod

        q_sin_init = q
        q_cos_init = q
        k_sin_init = k
        k_cos_init = k
        q_sin = torch.zeros(q.shape, device=q.device)
        q_cos = torch.zeros(q.shape, device=q.device)
        k_sin = torch.zeros(k.shape, device=k.device)
        k_cos = torch.zeros(k.shape, device=k.device)
        
        src_idx = incremental_state["steps"]["src"]
        tgt_idx = incremental_state["steps"]["tgt"]
        old_src_idx = simul_attn_chkpts["old_indices"]["src"]
        old_tgt_idx = simul_attn_chkpts["old_indices"]["tgt"]
       
        q_sin = torch.mul(q_sin_init, math.sin((math.pi / 2) * (tgt_idx / tgt_len_p)))
        q_cos = torch.mul(q_cos_init, math.cos((math.pi / 2) * (tgt_idx / tgt_len_p)))

        idx = torch.arange(1, src_len + 1)
        sin_tr = torch.sin((math.pi / 2) * (idx / src_len))
        cos_tr = torch.cos((math.pi / 2) * (idx / src_len))
        #sin_tr = simul_attn_chkpts["sin_tr"]
        #cos_tr = simul_attn_chkpts["cos_tr"]
        
        k_sin_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_sin"]
        k_cos_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_cos"]
        norm_sin_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_sin"]
        norm_cos_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_cos"]

        # build key transforms if necessary
        if k_sin_old is not None and k_cos_old is not None:
            old_src = list(k_sin_old.shape)[1]
            if old_src == src_idx:
                k_sin = k_sin_old
                k_cos = k_cos_old
            else:
                sin_tr = sin_tr[old_src:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                cos_tr = cos_tr[old_src:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                k_sin = torch.cat((k_sin_old, torch.matmul(k_sin_init[:, old_src:, :].unsqueeze(-1), sin_tr).squeeze(-1)), dim=1)
                k_cos = torch.cat((k_cos_old, torch.matmul(k_cos_init[:, old_src:, :].unsqueeze(-1), cos_tr).squeeze(-1)), dim=1)
        else:
            sin_tr = sin_tr[:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            cos_tr = cos_tr[:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr).squeeze(-1)
            k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr).squeeze(-1)

        # build normalization vectors if necessary
        if norm_sin_old is not None and norm_cos_old is not None:
            if old_src == src_idx:
                norm_sin = norm_sin_old
                norm_cos = norm_cos_old
            else:
                norm_sin = norm_sin_old + torch.sum(k_sin.unsqueeze(-1)[:, old_src:, :], dim=1)
                norm_cos = norm_cos_old + torch.sum(k_cos.unsqueeze(-1)[:, old_src:, :], dim=1)
        else:
            norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
            norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)
    
        # build out d x d intermediate matrix
        old_attn_weights_v_sin = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_sin"]
        old_attn_weights_v_cos = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_cos"]

        if old_attn_weights_v_sin is not None and old_attn_weights_v_cos is not None:
            if old_src == src_idx:
                attn_weights_v_sin = old_attn_weights_v_sin
                attn_weights_v_cos = old_attn_weights_v_cos
            else:
                attn_weights_v_sin = old_attn_weights_v_sin + torch.bmm(k_sin[:, old_src:, :].transpose(1, 2), v[:, old_src:, :]) 
                attn_weights_v_cos = old_attn_weights_v_cos + torch.bmm(k_cos[:, old_src:, :].transpose(1, 2), v[:, old_src:, :])
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

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_sin"] = norm_sin
        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_cos"] = norm_cos
        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_sin"] = k_sin
        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_cos"] = k_cos
        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["v"] = v
        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_sin"] = attn_weights_v_sin
        simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_cos"] = attn_weights_v_cos

        return attn, {
                "p_choose" : p_choose,
                "alpha"    : None,
                "beta"     : attn_weights,
        }

    def cosformer_attn_train_and_infer(
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
        
        if incremental_state is not None:
            """
            Monotonic attention at inference time
            Notice that this function is designed for simuleval not sequence_generator
            """

            if q.size(1) != 1:
                raise RuntimeError(
                    "Simultaneous translation models don't support batch decoding."
                )
            # 1. compute stepwise probability
            p_choose = self.p_choose(
                q, k, None, incremental_state
            ).squeeze(1)

            # 2. Compute the alpha
            src_len = k.size(0)
            # Maximum steps allows in this iteration
            max_steps = src_len - 1 if self.mass_preservation else src_len
            monotonic_cache = self._get_monotonic_buffer(incremental_state)
            # Step for each head
            monotonic_step = monotonic_cache.get(
                'head_step',
                p_choose.new_zeros(self.num_heads, 1).long()
            )
            assert monotonic_step is not None
            finish_read = monotonic_step.eq(max_steps)
            p_choose_i = torch.tensor(1)

            while finish_read.sum().item() < self.num_heads:
                # p_choose: self.num_heads, src_len
                # only choose the p at monotonic steps
                # p_choose_i: self.num_heads, 1
                p_choose_i = (
                    p_choose.gather(
                        1,
                        monotonic_step
                        .clamp(0, src_len - 1),
                    )
                )

                read_one_step = (
                    (p_choose_i < 0.5)
                    .type_as(monotonic_step)
                    .masked_fill(finish_read, 0)
                )
                # self.num_heads x 1
                # sample actions on unfinished seq
                # 0 means stay, finish reading
                # 1 means leave, continue reading

                monotonic_step += read_one_step

                finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

            # p_choose at last steps
            p_choose_i = (
                p_choose.gather(
                    1,
                    monotonic_step
                    .clamp(0, src_len - 1),
                )
            )

            monotonic_cache["head_step"] = monotonic_step
            # Whether a head is looking for new input
            monotonic_cache["head_read"] = (
                monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
            )
            self._set_monotonic_buffer(incremental_state, monotonic_cache)
            # end monotonic inference behavior, onto more typical functionality
      
        # training only behavior
        else:
            # 1. compute stepwise probability
            p_choose = self.p_choose(query, key, key_padding_mask)

            # 2. compute expected_alignment
            alpha = expected_alignment_from_p_choose(
                p_choose.float(),  # prevents latency loss from nan
                key_padding_mask,
                eps=self.eps,
            )

            if self.mass_preservation:
                alpha = mass_preservation(
                    alpha, key_padding_mask
                )

        # start of general behavior
        assert q is not None
        assert k is not None
        assert v is not None

        # prepping input tensors
        length, bsz, _ = query.size()
        q = self.q_in_proj[energy_type].forward(query)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        length, bsz, _ = key.size()
        k = self.k_in_proj[energy_type].forward(key)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        v = self.v_proj(value)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        src_len = k.size(1)
        tgt_len = q.size(1)
        
        q = F.relu(q)
        k = F.relu(k)
        # ending input tensor prep

        # implementation differs from typical key_padding_mask application, but this is useful later and should be fine
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
        #idx = torch.arange(1, src_len + 1, device = k.device)
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
            
            k_sin_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_sin"]
            k_cos_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_cos"]
            norm_sin_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_sin"]
            norm_cos_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_cos"]

            # build key transforms if necessary
            if k_sin_old is not None and k_cos_old is not None:
                old_src = list(k_sin_old.shape)[1]
                if old_src == tgt_idx:
                    k_sin = k_sin_old
                    k_cos = k_cos_old
                else:
                    sin_tr = sin_tr[old_src:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    cos_tr = cos_tr[old_src:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    k_sin = torch.cat((k_sin_old, torch.matmul(k_sin_init[:, old_src:, :].unsqueeze(-1), sin_tr).squeeze(-1)), dim=1)
                    k_cos = torch.cat((k_cos_old, torch.matmul(k_cos_init[:, old_src:, :].unsqueeze(-1), cos_tr).squeeze(-1)), dim=1)
            else:
                sin_tr = sin_tr[:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                cos_tr = cos_tr[:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr).squeeze(-1)
                k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr).squeeze(-1)

            # build normalization vectors if necessary
            if norm_sin_old is not None and norm_cos_old is not None:
                if old_src == tgt_idx:
                    norm_sin = norm_sin_old
                    norm_cos = norm_cos_old
                else:
                    norm_sin = norm_sin_old + torch.sum(k_sin.unsqueeze(-1)[:, old_src:, :], dim=1)
                    norm_cos = norm_cos_old + torch.sum(k_cos.unsqueeze(-1)[:, old_src:, :], dim=1)
            else:
                norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
                norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)

        else:
            max_len = max(src_len, tgt_len)
            if self.cosformer_attn_enable:
                loop_limit = math.ceil((max_len + self.max_src_len_step_size / 10) / self.max_src_len_step_size)
                
                step = self.max_src_len_step_size
                temp_idx = torch.arange(1, max_len + 1, device=k.device)
                bound_l = 0
                for i in range(loop_limit):
                    bound_h = min(math.ceil((i + 1) * step - step/10), max_len)
                    temp_idx[bound_l:bound_h] = temp_idx[bound_l:bound_h] / ((i + 1) * step)
                    bound_l = bound_h
                
                sin_tr = torch.sin((math.pi / 2) * temp_idx)
                cos_tr = torch.cos((math.pi / 2) * temp_idx)
            elif self.cosformer_expt_attn_enable:
                temp_idx = torch.arange(1, max_len + 1, device=k.device)
                sin_tr = torch.sin((math.pi*(1 - torch.exp(-1 * temp_idx)))/2)
                cos_tr = torch.cos((math.pi*(1 - torch.exp(-1 * temp_idx)))/2)
            
            sin_tr = sin_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            cos_tr = cos_tr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            
            if incremental_state is not None:
                q_sin = torch.mul(q_sin_init.unsqueeze(-1), sin_tr[0, tgt_idx - 1, 0, 0]).squeeze(-1)
                q_cos = torch.mul(q_cos_init.unsqueeze(-1), cos_tr[0, tgt_idx - 1, 0, 0]).squeeze(-1)
            else:    
                q_sin = torch.matmul(q_sin_init.unsqueeze(-1), sin_tr[0, :tgt_len, 0, 0]).squeeze(-1)
                q_cos = torch.matmul(q_cos_init.unsqueeze(-1), cos_tr[0, :tgt_len, 0, 0]).squeeze(-1)
            
            k_sin = torch.matmul(k_sin_init.unsqueeze(-1), sin_tr[0, :src_len, 0, 0]).squeeze(-1)
            k_cos = torch.matmul(k_cos_init.unsqueeze(-1), cos_tr[0, :src_len, 0, 0]).squeeze(-1)
            
            if incremental_state is not None:
                norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
                norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)
            else:
                norm_sin = torch.cumsum(k_sin, dim=1).transpose(1, 2)
                norm_cos = torch.cumsum(k_cos, dim=1).transpose(1, 2)

        # build out d x d intermediate matrix
        if simul_attn_chkpts is not None:
            old_attn_weights_v_sin = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_sin"]
            old_attn_weights_v_cos = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_cos"]

            if old_attn_weights_v_sin is not None and old_attn_weights_v_cos is not None:
                if old_src == src_idx:
                    attn_weights_v_sin = old_attn_weights_v_sin
                    attn_weights_v_cos = old_attn_weights_v_cos
                else:
                    attn_weights_v_sin = old_attn_weights_v_sin + torch.bmm(k_sin[:, old_src:, :].transpose(1, 2), v[:, old_src:, :]) 
                    attn_weights_v_cos = old_attn_weights_v_cos + torch.bmm(k_cos[:, old_src:, :].transpose(1, 2), v[:, old_src:, :])
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
                simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_sin"] = norm_sin
                simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_cos"] = norm_cos
                simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_sin"] = k_sin
                simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_cos"] = k_cos
                simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_sin"] = attn_weights_v_sin
                simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["kTv_cos"] = attn_weights_v_cos

            return attn, {
                    "p_choose" : p_choose,
                    "alpha"    : alpha,
                    "beta"     : attn_weights,
            }

        # training enabled for n x n masking
        else:
            attn_weights = attn_weights_v_sin + attn_weights_v_cos
           
            #if attn_mask is not None:
            #    attn_mask_bool = attn_mask.to(torch.bool)
            #    attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
           
            alpha = torch.cumsum(alpha.flip(dims=[2])).flip(dims=[2])
            attn_weights = attn_weights.masked_fill(~alpha, 0)

            torch.set_printoptions(threshold=10_000)
            print(attn_weights)

            attn_weights = torch.bmm(attn_weights, v)

            attn_probs = self.dropout_module(attn_weights)

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

            return attn, {
                    "p_choose" : p_choose,
                    "alpha"    : alpha,
                    "beta"     : attn_weights,
            }
   
    # structured slightly differently, will go back later and make it step by step instead of grouped
    # into training and inference
    def combin_attn_train_and_infer(
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
        
        assert q is not None
        assert k is not None
        assert v is not None
        
        if incremental_state is not None:
            """
            Monotonic attention at inference time
            Notice that this function is designed for simuleval not sequence_generator
            """

            if q.size(1) != 1:
                raise RuntimeError(
                    "Simultaneous translation models don't support batch decoding."
                )
            # 1. compute stepwise probability
            p_choose = self.p_choose(
                q, k, None, incremental_state
            ).squeeze(1)

            # 2. Compute the alpha
            src_len = k.size(0)
            # Maximum steps allows in this iteration
            max_steps = src_len - 1 if self.mass_preservation else src_len
            monotonic_cache = self._get_monotonic_buffer(incremental_state)
            # Step for each head
            monotonic_step = monotonic_cache.get(
                'head_step',
                p_choose.new_zeros(self.num_heads, 1).long()
            )
            assert monotonic_step is not None
            finish_read = monotonic_step.eq(max_steps)
            p_choose_i = torch.tensor(1)

            while finish_read.sum().item() < self.num_heads:
                # p_choose: self.num_heads, src_len
                # only choose the p at monotonic steps
                # p_choose_i: self.num_heads, 1
                p_choose_i = (
                    p_choose.gather(
                        1,
                        monotonic_step
                        .clamp(0, src_len - 1),
                    )
                )

                read_one_step = (
                    (p_choose_i < 0.5)
                    .type_as(monotonic_step)
                    .masked_fill(finish_read, 0)
                )
                # self.num_heads x 1
                # sample actions on unfinished seq
                # 0 means stay, finish reading
                # 1 means leave, continue reading

                monotonic_step += read_one_step

                finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

            # p_choose at last steps
            p_choose_i = (
                p_choose.gather(
                    1,
                    monotonic_step
                    .clamp(0, src_len - 1),
                )
            )

            monotonic_cache["head_step"] = monotonic_step
            # Whether a head is looking for new input
            monotonic_cache["head_read"] = (
                monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
            )
            self._set_monotonic_buffer(incremental_state, monotonic_cache)
            # end monotonic inference behavior, onto more typical functionality

        # training only behavior
        else:
            # 1. compute stepwise probability
            p_choose = self.p_choose(query, key, key_padding_mask)

            # 2. compute expected_alignment
            alpha = expected_alignment_from_p_choose(
                p_choose.float(),  # prevents latency loss from nan
                key_padding_mask,
                eps=self.eps,
            )

            if self.mass_preservation:
                alpha = mass_preservation(
                    alpha, key_padding_mask
                )

        # start of general behavior
        # prepping input tensors
        length, bsz, _ = query.size()
        q = self.q_in_proj[energy_type].forward(query)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        length, bsz, _ = key.size()
        k = self.k_in_proj[energy_type].forward(key)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        v = self.v_proj(value)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        src_len = k.size(1)
        tgt_len = q.size(1)
        
        q = F.relu(q)
        k = F.relu(k)
        # ending input tensor prep
        
        
        if key_padding_mask is not None:
            key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
            k = k.view(bsz, self.num_heads, src_len, list(k.shape)[2])          
            k = k.masked_fill(key_pad_mask_unsqueeze, 0)
            k = k.view(bsz*self.num_heads, src_len, list(k.shape)[3])          

        q_t = q
        q_tt = q
        k_t = k
        k_tt = k
        #idx = torch.arange(1, src_len + 1, device = k.device)
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

            k_t_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_t"]
            k_tt_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_tt"]
            norm_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm"]
            norm_t_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_t"]
            norm_tt_old = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_tt"]
            j_tr = simul_attn_chkpts["j_tr"]
            j_ttr = simul_attn_chkpts["j_ttr"]

            # build key transforms if necessary
            if k_t_old is not None and k_tt_old is not None:
                assert k_t_old.shape == k_tt_old.shape
                old_src = list(k_t_old.shape)[1]
                if old_src == src_idx:
                    k_t = k_t_old
                    k_tt = k_tt_old
                else:
                    j_tr = j_tr[old_src:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    j_ttr = j_ttr[old_src:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    k_t = torch.cat((k_t_old, torch.matmul(k_t[:, old_src:, :].unsqueeze(-1), j_tr).squeeze(-1)), dim=1)
                    k_tt = torch.cat((k_tt_old, torch.matmul(k_tt[:, old_src:, :].unsqueeze(-1), j_ttr).squeeze(-1)), dim=1)
            else:
                j_tr = j_tr[:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                j_ttr = j_ttr[:src_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                k_t = torch.matmul(k_t.unsqueeze(-1), j_tr).squeeze(-1)
                k_tt = torch.matmul(k_tt.unsqueeze(-1), j_ttr).squeeze(-1)

            # build normalization vectors if necessary
            if norm_old is not None and norm_t_old is not None and norm_tt_old is not None:
                assert norm_old.shape == norm_t_old.shape
                assert norm_old.shape == norm_tt_old.shape
                if old_src == src_idx:
                    norm = norm_old
                    norm_t = norm_t_old
                    norm_tt = norm_tt_old
                else:
                    norm = norm + torch.sum(k.unsqueeze(-1)[:, old_src:, :], dim=1)
                    norm_t = norm_t_old + torch.sum(k_t.unsqueeze(-1)[:, old_src:, :], dim=1)
                    norm_tt = norm_tt_old + torch.sum(k_tt.unsqueeze(-1)[:, old_src:, :], dim=1)
            else:
                norm = torch.sum(k.unsqueeze(-1), dim=1)
                norm_t = torch.sum(k_t.unsqueeze(-1), dim=1)
                norm_tt = torch.sum(k_tt.unsqueeze(-1), dim=1)
            

            # build out d x d intermediate matrix
            old_attn_weights = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["attn_weights"]
            old_attn_weights_kt = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["attn_weights_kt"]
            old_attn_weights_ktt = simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["attn_weights_ktt"]

            if old_attn_weights is not None and old_attn_weights_kt is not None and old_attn_weights_ktt is not None:
                assert old_attn_weights.shape == old_attn_weights_kt.shape
                assert old_attn_weights.shape == old_attn_weights_ktt.shape
                if old_src == src_idx:
                    attn_weights = old_attn_weights
                    attn_weights_kt = old_attn_weights_kt
                    attn_weights_ktt = old_attn_weights_ktt
                else:
                    attn_weights = old_attn_weights + torch.bmm(k[:, old_src:, :].transpose(1, 2), v[:, old_src:, :])
                    attn_weights_kt = old_attn_weights_kt + torch.bmm(k_t[:, old_src:, :].transpose(1, 2), v[:, old_src:, :])
                    attn_weights_ktt = old_attn_weights_ktt + torch.bmm(k_tt[:, old_src:, :].transpose(1, 2), v[:, old_src:, :])
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

            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_t"] = k_t
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["k_tt"] = k_tt
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm"] = norm
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_t"] = norm_t
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["norm_tt"] = norm_tt
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["attn_weights"] = attn_weights
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["attn_weights_kt"] = attn_weights_kt
            simul_attn_chkpts["layers"][layer_idx]["cross_attn"]["attn_weights_ktt"] = attn_weights_ktt

            return attn, {
                    "p_choose" : p_choose,
                    "alpha"    : alpha,
                    "beta"     : attn_weights_f_sum,
            }

        else:
            max_len = max(src_len, tgt_len)
            if self.combin_attn_enable:
                loop_limit = math.ceil((max_len + self.max_src_len_step_size / 10) / self.max_src_len_step_size)
                
                step = self.max_src_len_step_size
                temp_idx = torch.arange(1, src_len + 1, device=k.device) 
                bound_l = 0
                for i in range(loop_limit):
                    bound_h = min(math.ceil((i + 1) * step - step/10), max_len)
                    temp_idx[bound_l:bound_h] = (temp_idx[bound_l:bound_h] + 0.1) / ((i + 1) * step)
                    bound_l = bound_h
                
                i_tr = temp_idx
                i_ttr = torch.square(temp_idx)
                j_tr = temp_idx
                j_ttr = torch.square(temp_idx)
            
            elif self.combin_expt_attn_enable:
                temp_idx = torch.arange(1, src_len + 1, device=k.device) 
                i_tr = torch.exp(-1 * temp_idx)
                i_ttr = torch.exp(-2 * temp_idx)
                j_tr = torch.exp(-1 * temp_idx)
                j_ttr = torch.exp(-2 * temp_idx)

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
                
                #if attn_mask is not None:
                #    attn_mask_bool = attn_mask.to(torch.bool)
                #    attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
                
                alpha = torch.cumsum(alpha.flip(dims=[2])).flip(dims=[2])
                attn_weights = attn_weights.masked_fill(~alpha, 0)

                
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

            return attn, {
                    "p_choose" : p_choose,
                    "alpha"    : alpha,
                    "beta"     : attn_weights_sum,
            }

@register_monotonic_attention("infinite_lookback")
class MonotonicInfiniteLookbackAttention(
    MonotonicAttention
):
    def __init__(self, args):
        super().__init__(args)
        self.soft_attention = True
        self.init_soft_attention()

    def init_soft_attention(self):
        self.k_proj_soft = nn.Linear(self.kdim, self.embed_dim, bias=True)
        self.q_proj_soft = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_in_proj["soft"] = self.k_proj_soft
        self.q_in_proj["soft"] = self.q_proj_soft

        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
            nn.init.xavier_uniform_(
                self.q_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
        else:
            nn.init.xavier_uniform_(self.k_in_proj["soft"].weight)
            nn.init.xavier_uniform_(self.q_in_proj["soft"].weight)


@register_monotonic_attention("waitk")
class WaitKAttention(
    MonotonicInfiniteLookbackAttention
):
    """
    STACL: Simultaneous Translation with Implicit Anticipation and
    Controllable Latency using Prefix-to-Prefix Framework
    https://www.aclweb.org/anthology/P19-1289/
    """
    def __init__(self, args):
        super().__init__(args)
        self.q_in_proj["soft"] = self.q_in_proj["monotonic"]
        self.k_in_proj["soft"] = self.k_in_proj["monotonic"]

        self.waitk_lagging = args.waitk_lagging
        assert self.waitk_lagging > 0, (
            f"Lagging has to been larger than 0, get {self.waitk_lagging}."
        )

    @staticmethod
    def add_args(parser):
        super(
            MonotonicInfiniteLookbackAttention,
            MonotonicInfiniteLookbackAttention
        ).add_args(parser)

        parser.add_argument(
            "--waitk-lagging", type=int, required=True, help="Wait K lagging"
        )

    def p_choose_from_qk(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        assert query is not None
        assert key is not None

        p_choose = waitk_p_choose(
            tgt_len=query.size(0),
            src_len=key.size(0),
            bsz=query.size(1) * self.num_heads,
            waitk_lagging=self.waitk_lagging,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
        )

        return p_choose.to(query)


@register_monotonic_attention("chunkwise")
class ChunkwiseAttention(
    MonotonicInfiniteLookbackAttention
):
    def __init__(self, args):
        super().__init__(args)
        self.chunk_size = args.mocha_chunk_size
        assert self.chunk_size > 1

    @staticmethod
    def add_args(parser):
        super(
            MonotonicInfiniteLookbackAttention
        ).add_args(parser)

        parser.add_argument(
            "--mocha-chunk-size", type=int,
            required=True, help="Mocha chunk size"
        )
