import math
import os
import struct

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithPast,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .configuration_mixlab import MixlabConfig
from .pooling_mixlab import pool_sequence
from .ttt_mlp_mixlab import (
    MixlabTTTMLPBlock,
    MixlabTTTMLPState,
    require_right_padded_ttt_batch,
)


class MixlabRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


def make_mixlab_norm(config, dim):
    norm_type = str(getattr(config, "norm_type", "rmsnorm") or "rmsnorm").lower()
    eps = float(getattr(config, "norm_eps", 1e-5) or 1e-5)
    affine = bool(getattr(config, "norm_affine", True))
    if norm_type in ("rmsnorm", "rms_norm", "rms"):
        if not affine:
            raise ValueError("norm_affine=False is not supported with rmsnorm")
        return MixlabRMSNorm(dim, eps=eps)
    if norm_type in ("layernorm", "layer_norm", "layer"):
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)
    raise ValueError(f"unsupported norm_type {norm_type!r}")


def norm_placement(config):
    value = str(getattr(config, "norm_placement", "pre") or "pre").lower()
    if value in ("pre", "post", "sandwich"):
        return value
    raise ValueError(f"unsupported norm_placement {value!r}")


def normalize_attn_post_norm(value):
    value = str(value or "inherit").strip().lower()
    if value in ("", "inherit"):
        return "inherit"
    if value in ("none", "off", "disabled", "false"):
        return "none"
    if value in ("after", "after_out", "after_outproj", "after_out_proj", "after_output_projection"):
        return "after_outproj"
    if value in (
        "before",
        "before_out",
        "before_outproj",
        "before_out_proj",
        "before_output_projection",
        "pre_outproj",
        "pre_out_proj",
    ):
        return "before_outproj"
    return value


def effective_attn_post_norm(block_config, placement):
    value = normalize_attn_post_norm(block_config.get("attn_post_norm", "inherit"))
    if value == "inherit":
        return "after_outproj" if placement in ("post", "sandwich") else "none"
    if value in ("none", "after_outproj", "before_outproj"):
        return value
    raise ValueError(f"unsupported exported attn_post_norm {value!r}")


def normalize_relative_embedding_norm(value):
    value = str(value or "none").strip().lower()
    if value in ("", "none", "off", "disabled", "false"):
        return "none"
    if value in ("layernorm", "layer_norm", "ln"):
        return "layernorm"
    return value


class MixlabLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class MixlabDWAState:
    def __init__(self, alphas, static_embeddings):
        self.alphas = alphas
        self.history = [static_embeddings]
        self.step = 0

    def apply(self, x):
        if self.step >= len(self.alphas):
            raise ValueError("DWA alpha count is smaller than the emitted residual-point count")
        self.history.append(x)
        alpha = self.alphas[self.step].to(device=x.device, dtype=x.dtype)
        if alpha.numel() != len(self.history):
            raise ValueError(
                f"DWA alpha {self.step} has {alpha.numel()} entries, expected {len(self.history)}"
            )
        out = self.history[0] * alpha[0]
        for idx in range(1, len(self.history)):
            out = out + self.history[idx] * alpha[idx]
        self.step += 1
        return out

    def finish(self):
        if self.step != len(self.alphas):
            raise ValueError(
                f"DWA consumed {self.step} alpha vectors, expected {len(self.alphas)}"
            )


class MixlabDWAHeadState:
    def __init__(self, alphas, static_embeddings):
        self.alphas = alphas
        self.history = [static_embeddings]
        self.step = 0

    def apply(self, x):
        self.history.append(x)
        self.step += 1
        return x

    def finish(self):
        if self.step != len(self.alphas):
            raise ValueError(
                f"DWA captured {self.step} residual points, expected {len(self.alphas)}"
            )
        if len(self.alphas) == 0:
            raise ValueError("head-scoped DWA requires at least one alpha vector")
        alpha = self.alphas[len(self.alphas) - 1].to(device=self.history[-1].device, dtype=self.history[-1].dtype)
        if alpha.numel() != len(self.history):
            raise ValueError(
                f"head-scoped DWA alpha has {alpha.numel()} entries, expected {len(self.history)}"
            )
        out = self.history[0] * alpha[0]
        for idx in range(1, len(self.history)):
            out = out + self.history[idx] * alpha[idx]
        return out


def count_dwa_points(block_configs):
    count = 0
    for block in block_configs:
        block_type = str(block.get("type", "") or "").lower()
        if block_type == "plain":
            count += 2
        elif block_type in ("swiglu", "geglu", "mlp", "moe"):
            count += 1
        else:
            raise ValueError(f"unsupported exported DWA block type {block_type!r}")
    return count


def rotate_adjacent_rope(x, rope_dims, cos_t=None, sin_t=None, base=10000.0):
    head_dim = x.shape[-1]
    if rope_dims is None or rope_dims <= 0 or rope_dims >= head_dim:
        rope_dims = head_dim
    if rope_dims % 2 != 0:
        raise ValueError(f"rope_dims must be even, got {rope_dims}")

    seq_len = x.shape[-2]
    x_rot = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    pair_count = rope_dims // 2
    if cos_t is None or sin_t is None:
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        dim_idx = torch.arange(pair_count, device=x.device, dtype=x.dtype)
        freqs = torch.exp(dim_idx * (-math.log(base) * 2.0 / float(rope_dims)))
        angles = positions[:, None] * freqs[None, :]
        cos_t = torch.cos(angles).view(1, 1, seq_len, pair_count)
        sin_t = torch.sin(angles).view(1, 1, seq_len, pair_count)
    else:
        cos_t = cos_t[:, :, :seq_len, :]
        sin_t = sin_t[:, :, :seq_len, :]

    pairs = x_rot.view(*x_rot.shape[:-1], pair_count, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated = torch.stack((even * cos_t - odd * sin_t, even * sin_t + odd * cos_t), dim=-1)
    rotated = rotated.reshape(*x_rot.shape)
    if x_pass.shape[-1] == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


def rotate_half_rope(x, rope_dims, cos_t=None, sin_t=None, base=10000.0):
    head_dim = x.shape[-1]
    if rope_dims is None or rope_dims <= 0 or rope_dims >= head_dim:
        rope_dims = head_dim
    if rope_dims % 2 != 0:
        raise ValueError(f"rope_dims must be even, got {rope_dims}")

    seq_len = x.shape[-2]
    x_rot = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    pair_count = rope_dims // 2
    if cos_t is None or sin_t is None:
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        dim_idx = torch.arange(pair_count, device=x.device, dtype=x.dtype)
        freqs = torch.exp(dim_idx * (-math.log(base) * 2.0 / float(rope_dims)))
        angles = positions[:, None] * freqs[None, :]
        cos_t = torch.cos(angles).view(1, 1, seq_len, pair_count)
        sin_t = torch.sin(angles).view(1, 1, seq_len, pair_count)
    else:
        cos_t = cos_t[:, :, :seq_len, :]
        sin_t = sin_t[:, :, :seq_len, :]

    first = x_rot[..., :pair_count]
    second = x_rot[..., pair_count:]
    rotated = torch.cat((first * cos_t - second * sin_t, first * sin_t + second * cos_t), dim=-1)
    if x_pass.shape[-1] == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


class MixlabPlainBlock(nn.Module):
    def __init__(self, config, block_config):
        super().__init__()
        dim = config.model_dim
        heads = int(block_config["heads"])
        self.heads = heads
        self.kv_heads = int(block_config.get("kv_heads", 0) or heads)
        self.head_dim = dim // heads
        if heads % self.kv_heads != 0:
            raise ValueError(f"heads must be divisible by kv_heads, got heads={heads} kv_heads={self.kv_heads}")
        if int(block_config.get("kv_source", 0) or 0) > 0:
            raise ValueError("kv_source is not supported by Mixlab HF export")
        self.kv_group_size = heads // self.kv_heads
        self.rope_dims = int(block_config.get("rope_dims", 0) or 0)
        self.rope_convention = str(block_config.get("rope_convention", "") or "adjacent_pair").lower()
        if self.rope_convention not in ("adjacent_pair", "half_rotation"):
            raise ValueError(f"unsupported exported rope_convention {self.rope_convention!r}")
        self.positional_embedding = str(getattr(config, "positional_embedding", "rope") or "rope").lower()
        self.differential_attention = bool(block_config.get("differential_attention", False))
        self.diff_sub_head_dim = self.head_dim // 2
        self.differential_lambda_init = float(block_config.get("differential_lambda_init", 0.0) or 0.0)
        if self.differential_attention:
            if self.head_dim % 2 != 0:
                raise ValueError("differential_attention requires an even per-head width")
            if self.kv_heads != self.heads:
                raise ValueError("differential_attention does not support kv_heads in v1")
        self.effective_rope_dims = self.rope_dims
        rope_limit = self.diff_sub_head_dim if self.differential_attention else self.head_dim
        if self.effective_rope_dims <= 0 or self.effective_rope_dims >= rope_limit:
            self.effective_rope_dims = rope_limit
        self.attention_mask = str(block_config.get("attention_mask", "") or "causal").lower()
        self.window_size = int(block_config.get("window_size", 0) or 0)
        self.attn_bias = bool(block_config.get("attn_bias", False))
        self.attn_value_gate = bool(block_config.get("attn_value_gate", False))
        self.xsa = bool(block_config.get("xsa", False))
        self.sparse_attn_gate = bool(block_config.get("sparse_attn_gate", False))
        if self.differential_attention and self.attn_value_gate:
            raise ValueError("differential_attention does not support attn_value_gate in v1")
        if self.differential_attention and self.xsa:
            raise ValueError("differential_attention does not support xsa in v1")
        self.attn_gate_dim = min(dim, 12)
        self.relative_attention = str(block_config.get("relative_attention", "") or "none").lower()
        self.relative_attention_window = int(block_config.get("relative_attention_window", 0) or 128)
        self.relative_attention_parameterization = str(
            block_config.get("relative_attention_parameterization", "") or "per_block_projections"
        ).lower()
        self.norm_placement = norm_placement(config)
        self.attn_post_norm = effective_attn_post_norm(block_config, self.norm_placement)
        if self.differential_attention and self.attn_post_norm != "none":
            raise ValueError("differential_attention does not support attn_post_norm in v1")
        self.ffn_internal_norm_enabled = bool(getattr(config, "ffn_internal_norm", False))
        self.ffn_activation = str(block_config.get("ffn_activation", "") or "silu").lower()
        if self.ffn_activation not in ("silu", "gelu", "gelu_new", "geglu", "swiglu"):
            raise ValueError(f"unsupported exported plain ffn_activation {self.ffn_activation!r}")
        self.ffn_pre_norm = bool(block_config.get("ffn_pre_norm", False))
        self.ffn_bias = bool(block_config.get("ffn_bias", False))
        self.norm = make_mixlab_norm(config, dim) if self.norm_placement in ("pre", "sandwich") else None
        self.wq = MixlabLinear(dim, dim, bias=self.attn_bias)
        self.wk = MixlabLinear(dim, self.kv_heads * self.head_dim, bias=self.attn_bias)
        self.value_dim = self.kv_heads * self.head_dim
        self.wv = MixlabLinear(dim, self.value_dim + (dim if self.attn_value_gate else 0), bias=self.attn_bias)
        self.q_norm = None
        self.k_norm = None
        if bool(block_config.get("qk_norm", False)):
            if self.differential_attention:
                raise ValueError("differential_attention does not support qk_norm in v1")
            self.q_norm = MixlabRMSNorm(self.head_dim)
            self.k_norm = MixlabRMSNorm(self.head_dim)
        self.diff_lambda_q1 = None
        self.diff_lambda_k1 = None
        self.diff_lambda_q2 = None
        self.diff_lambda_k2 = None
        self.diff_subln = None
        if self.differential_attention:
            self.diff_lambda_q1 = nn.Parameter(torch.empty(self.diff_sub_head_dim))
            self.diff_lambda_k1 = nn.Parameter(torch.empty(self.diff_sub_head_dim))
            self.diff_lambda_q2 = nn.Parameter(torch.empty(self.diff_sub_head_dim))
            self.diff_lambda_k2 = nn.Parameter(torch.empty(self.diff_sub_head_dim))
            self.diff_subln = MixlabRMSNorm(self.head_dim, eps=1e-3)
            nn.init.normal_(self.diff_lambda_q1, mean=0.0, std=0.1)
            nn.init.normal_(self.diff_lambda_k1, mean=0.0, std=0.1)
            nn.init.normal_(self.diff_lambda_q2, mean=0.0, std=0.1)
            nn.init.normal_(self.diff_lambda_k2, mean=0.0, std=0.1)
        self.relative_embeddings = None
        self.w_pos_key = None
        self.w_pos_query = None
        if self.relative_attention == "deberta_p2c_c2p":
            if self.differential_attention:
                raise ValueError("differential_attention does not support relative_attention in v1")
            if self.relative_attention_parameterization not in ("per_block_projections", "shared_qk_reuse"):
                raise ValueError(
                    f"unsupported exported relative_attention_parameterization "
                    f"{self.relative_attention_parameterization!r}"
                )
            rel_rows = 2 * self.relative_attention_window - 1
            if self.relative_attention_parameterization == "per_block_projections":
                self.relative_embeddings = nn.Parameter(torch.empty(rel_rows, dim))
                self.w_pos_key = MixlabLinear(dim, dim)
                self.w_pos_query = MixlabLinear(dim, dim)
                nn.init.xavier_uniform_(self.relative_embeddings)
        elif self.relative_attention not in ("", "none"):
            raise ValueError(f"unsupported exported relative_attention {self.relative_attention!r}")
        elif self.relative_attention_parameterization == "shared_qk_reuse":
            raise ValueError("shared_qk_reuse requires relative_attention='deberta_p2c_c2p'")
        self.qk_gain = None
        if float(block_config.get("qk_gain", 0.0) or 0.0) > 0.0:
            if self.differential_attention:
                raise ValueError("differential_attention does not support qk_gain in v1")
            self.qk_gain = nn.Parameter(torch.empty(heads))
            nn.init.constant_(self.qk_gain, float(block_config["qk_gain"]))
        self.attn_gate_w = None
        if self.sparse_attn_gate:
            if self.differential_attention:
                raise ValueError("differential_attention does not support sparse_attn_gate in v1")
            self.attn_gate_w = nn.Parameter(torch.empty(heads, self.attn_gate_dim))
            nn.init.zeros_(self.attn_gate_w)
        self.post_attn_norm = make_mixlab_norm(config, dim) if self.attn_post_norm != "none" else None
        self.wo = MixlabLinear(dim, dim, bias=self.attn_bias)
        self.ffn_norm = make_mixlab_norm(config, dim) if self.norm_placement == "sandwich" or self.ffn_pre_norm else None
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.ff_gate = MixlabLinear(dim, ffn_dim) if self.ffn_activation in ("geglu", "swiglu") else None
        self.ff1 = MixlabLinear(dim, ffn_dim, bias=self.ffn_bias)
        self.ffn_internal_norm = make_mixlab_norm(config, ffn_dim) if self.ffn_internal_norm_enabled else None
        self.ff2 = MixlabLinear(ffn_dim, dim, bias=self.ffn_bias)
        self.post_ffn_norm = make_mixlab_norm(config, dim) if self.norm_placement in ("post", "sandwich") else None
        # NB: the causal mask and RoPE tables are computed dynamically in
        # forward() rather than cached as buffers. `from_pretrained` initializes
        # custom models on the meta device, where value-dependent buffers built
        # from torch.ones/torch.arange in __init__ are materialized as zeros and
        # silently disable masking / rotation. Recomputing per forward is cheap
        # for the eval-time graph and immune to that hazard.

    def _rope_tables(self, seq_len, device, dtype):
        pair_count = self.effective_rope_dims // 2
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        dim_idx = torch.arange(pair_count, device=device, dtype=torch.float32)
        freqs = torch.exp(dim_idx * (-math.log(10000.0) * 2.0 / float(self.effective_rope_dims)))
        angles = positions[:, None] * freqs[None, :]
        cos_t = torch.cos(angles).view(1, 1, seq_len, pair_count).to(dtype)
        sin_t = torch.sin(angles).view(1, 1, seq_len, pair_count).to(dtype)
        return cos_t, sin_t

    @staticmethod
    def _build_attention_mask(seq_len, window_size):
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        if window_size > 0 and window_size < seq_len:
            pos = torch.arange(seq_len)
            query_pos = pos.view(seq_len, 1)
            key_pos = pos.view(1, seq_len)
            too_old = key_pos < (query_pos - (window_size - 1))
            mask = mask | too_old
        return mask

    def _repeat_kv(self, x):
        if self.kv_group_size == 1:
            return x
        bsz, kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, kv_heads, self.kv_group_size, seq_len, head_dim)
        return x.reshape(bsz, self.heads, seq_len, head_dim)

    def _repeat_relative_kv(self, x):
        if self.kv_group_size == 1:
            return x
        kv_heads, rel_rows, head_dim = x.shape
        x = x[:, None, :, :].expand(kv_heads, self.kv_group_size, rel_rows, head_dim)
        return x.reshape(self.heads, rel_rows, head_dim)

    def _deberta_relative_bias(self, q, k, seq_len, shared_relative_embeddings=None):
        if self.relative_attention_parameterization == "shared_qk_reuse":
            if shared_relative_embeddings is None:
                raise ValueError("shared_qk_reuse requires model-level relative embeddings")
            rel_source = shared_relative_embeddings
            rel_key = self.wk(rel_source)
            rel_query = self.wq(rel_source)
        else:
            rel_key = self.w_pos_key(self.relative_embeddings)
            rel_query = self.w_pos_query(self.relative_embeddings)
        rel_rows = 2 * self.relative_attention_window - 1
        if self.relative_attention_parameterization == "shared_qk_reuse":
            rel_key = rel_key.view(rel_rows, self.kv_heads, self.head_dim).permute(1, 0, 2)
            rel_key = self._repeat_relative_kv(rel_key)
        else:
            rel_key = rel_key.view(rel_rows, self.heads, self.head_dim).permute(1, 0, 2)
        rel_query = rel_query.view(rel_rows, self.heads, self.head_dim).permute(1, 0, 2)
        pos = torch.arange(seq_len, device=q.device)
        rel = pos.view(seq_len, 1) - pos.view(1, seq_len)
        rel_idx = self._relative_position_bucket(rel, seq_len) + (self.relative_attention_window - 1)
        c2p_key = rel_key[:, rel_idx, :]
        p2c_query = rel_query[:, rel_idx, :]
        c2p_bias = torch.einsum("bhid,hijd->bhij", q, c2p_key)
        p2c_bias = torch.einsum("bhjd,hijd->bhij", k, p2c_query)
        return c2p_bias + p2c_bias

    def _relative_position_bucket(self, relative_pos, max_position):
        bucket_size = self.relative_attention_window
        if bucket_size <= 1:
            return torch.zeros_like(relative_pos)
        mid = bucket_size // 2
        sign = torch.sign(relative_pos)
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            torch.full_like(relative_pos, mid - 1),
            torch.abs(relative_pos).clamp(max=max(max_position - 1, 0)),
        )
        if mid > 0 and max_position - 1 > mid:
            denom = math.log(float(max_position - 1) / float(mid))
            log_pos = torch.ceil(
                torch.log(abs_pos.float() / float(mid)) / denom * float(mid - 1)
            ).to(torch.long) + mid
        else:
            log_pos = torch.full_like(relative_pos, bucket_size - 1)
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign)
        max_bucket = bucket_size - 1
        return torch.clamp(bucket_pos, -max_bucket, max_bucket).long()

    @staticmethod
    def _apply_attention_mask(scores, attention_mask):
        if attention_mask is None:
            return scores
        mask = attention_mask.to(device=scores.device)
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]
        elif mask.dim() != 4:
            raise ValueError(
                f"attention_mask must have rank 2, 3, or 4; got shape {tuple(attention_mask.shape)}"
            )
        if mask.dtype.is_floating_point and torch.any(mask < 0):
            return scores + mask.to(dtype=scores.dtype)
        return scores.masked_fill(~mask.bool(), -1e9)

    def forward(self, x, shared_relative_embeddings=None, dwa=None, attention_mask=None):
        residual = x
        x_norm = self.norm(x) if self.norm is not None else x
        bsz, seq_len, dim = x_norm.shape

        q = self.wq(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_norm).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.wv(x_norm)
        value_gate = None
        if self.attn_value_gate:
            v_proj, value_gate = torch.split(v_proj, [self.value_dim, dim], dim=-1)
            value_gate = torch.nn.functional.gelu(value_gate, approximate="tanh")
        v = v_proj.view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.differential_attention:
            q1, q2 = torch.split(q, [self.diff_sub_head_dim, self.diff_sub_head_dim], dim=-1)
            k1, k2 = torch.split(k, [self.diff_sub_head_dim, self.diff_sub_head_dim], dim=-1)
            if self.positional_embedding == "rope":
                cos_t, sin_t = self._rope_tables(seq_len, q.device, q.dtype)
                if self.rope_convention == "half_rotation":
                    q1 = rotate_half_rope(q1, self.effective_rope_dims, cos_t, sin_t)
                    k1 = rotate_half_rope(k1, self.effective_rope_dims, cos_t, sin_t)
                    q2 = rotate_half_rope(q2, self.effective_rope_dims, cos_t, sin_t)
                    k2 = rotate_half_rope(k2, self.effective_rope_dims, cos_t, sin_t)
                else:
                    q1 = rotate_adjacent_rope(q1, self.effective_rope_dims, cos_t, sin_t)
                    k1 = rotate_adjacent_rope(k1, self.effective_rope_dims, cos_t, sin_t)
                    q2 = rotate_adjacent_rope(q2, self.effective_rope_dims, cos_t, sin_t)
                    k2 = rotate_adjacent_rope(k2, self.effective_rope_dims, cos_t, sin_t)
            elif self.positional_embedding in ("learned_absolute", "none"):
                pass
            else:
                raise ValueError(f"unsupported positional_embedding {self.positional_embedding!r}")

            scores1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(float(self.diff_sub_head_dim))
            scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(float(self.diff_sub_head_dim))
            if self.attention_mask == "causal":
                causal_mask = self._build_attention_mask(seq_len, self.window_size).to(scores1.device)
                causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
                scores1 = scores1.masked_fill(causal_mask, -1e9)
                scores2 = scores2.masked_fill(causal_mask, -1e9)
            elif self.attention_mask in ("bidirectional", "none"):
                pass
            else:
                raise ValueError(f"unsupported exported attention_mask {self.attention_mask!r}")
            scores1 = self._apply_attention_mask(scores1, attention_mask)
            scores2 = self._apply_attention_mask(scores2, attention_mask)
            attn1 = torch.softmax(scores1, dim=-1)
            attn2 = torch.softmax(scores2, dim=-1)
            lambda_dot1 = torch.clamp(
                torch.sum(self.diff_lambda_q1 * self.diff_lambda_k1), min=-5.0, max=5.0
            )
            lambda_dot2 = torch.clamp(
                torch.sum(self.diff_lambda_q2 * self.diff_lambda_k2), min=-5.0, max=5.0
            )
            lambda_value = (
                torch.exp(lambda_dot1)
                - torch.exp(lambda_dot2)
                + self.differential_lambda_init
            )
            ctx = torch.matmul(attn1 - lambda_value * attn2, v)
            ctx = self.diff_subln(ctx) * (1.0 - self.differential_lambda_init)
        elif self.relative_attention == "deberta_p2c_c2p":
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores + self._deberta_relative_bias(q, k, seq_len, shared_relative_embeddings)
            scores = scores / math.sqrt(float(self.head_dim * 3))
        else:
            if self.positional_embedding == "rope":
                cos_t, sin_t = self._rope_tables(seq_len, q.device, q.dtype)
                if self.rope_convention == "half_rotation":
                    q = rotate_half_rope(q, self.effective_rope_dims, cos_t, sin_t)
                    k = rotate_half_rope(k, self.effective_rope_dims, cos_t, sin_t)
                else:
                    q = rotate_adjacent_rope(q, self.effective_rope_dims, cos_t, sin_t)
                    k = rotate_adjacent_rope(k, self.effective_rope_dims, cos_t, sin_t)
            elif self.positional_embedding in ("learned_absolute", "none"):
                pass
            else:
                raise ValueError(f"unsupported positional_embedding {self.positional_embedding!r}")
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        if self.qk_gain is not None:
            scores = scores * self.qk_gain.view(1, self.heads, 1, 1)
        if not self.differential_attention:
            if self.attention_mask == "causal":
                causal_mask = self._build_attention_mask(seq_len, self.window_size).to(scores.device)
                causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
                scores = scores.masked_fill(causal_mask, -1e9)
            elif self.attention_mask in ("bidirectional", "none"):
                pass
            else:
                raise ValueError(f"unsupported exported attention_mask {self.attention_mask!r}")
            scores = self._apply_attention_mask(scores, attention_mask)
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)
            if self.xsa:
                dot_yv = torch.sum(ctx * v, dim=-1, keepdim=True)
                dot_vv = torch.sum(v * v, dim=-1, keepdim=True)
                ctx = ctx - (dot_yv / (dot_vv + 1e-8)) * v
            if self.sparse_attn_gate:
                gate_in = residual[..., :self.attn_gate_dim]
                gate = torch.sigmoid(torch.matmul(gate_in, self.attn_gate_w.transpose(0, 1)))
                ctx = ctx * gate.transpose(1, 2).unsqueeze(-1)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        if value_gate is not None:
            ctx = ctx * value_gate
        if self.attn_post_norm == "before_outproj":
            ctx = self.post_attn_norm(ctx)
        attn_delta = self.wo(ctx)
        if self.attn_post_norm == "after_outproj":
            attn_delta = self.post_attn_norm(attn_delta)
        x = residual + attn_delta
        if dwa is not None:
            x = dwa.apply(x)

        ff_in = self.ffn_norm(x) if self.ffn_norm is not None else x
        if self.ffn_activation == "silu":
            ff_hidden = torch.nn.functional.silu(self.ff1(ff_in))
        elif self.ffn_activation == "gelu":
            ff_hidden = torch.nn.functional.gelu(self.ff1(ff_in), approximate="none")
        elif self.ffn_activation == "gelu_new":
            ff_hidden = torch.nn.functional.gelu(self.ff1(ff_in), approximate="tanh")
        else:
            gate = self.ff_gate(ff_in)
            if self.ffn_activation == "geglu":
                gate = torch.nn.functional.gelu(gate, approximate="tanh")
            else:
                gate = torch.nn.functional.silu(gate)
            ff_hidden = gate * self.ff1(ff_in)
        if self.ffn_internal_norm is not None:
            ff_hidden = self.ffn_internal_norm(ff_hidden)
        ff = self.ff2(ff_hidden)
        if self.post_ffn_norm is not None:
            ff = self.post_ffn_norm(ff)
        x = x + ff
        if dwa is not None:
            x = dwa.apply(x)
        return x


class MixlabMoEExpert(nn.Module):
    def __init__(self, config, expert_config):
        super().__init__()
        dim = config.model_dim
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.expert_type = str(expert_config.get("type", "") or "swiglu").lower()
        self.activation = str(expert_config.get("activation", "") or "silu").lower()
        self.leaky_slope = float(expert_config.get("leaky_slope", 0.0) or 0.5)
        if self.expert_type in ("swiglu", "geglu"):
            self.w_gate = MixlabLinear(dim, ffn_dim)
            self.w_up = MixlabLinear(dim, ffn_dim)
            self.w_down = MixlabLinear(ffn_dim, dim)
        elif self.expert_type == "mlp":
            self.w_up = MixlabLinear(dim, ffn_dim)
            self.w_down = MixlabLinear(ffn_dim, dim)
        else:
            raise ValueError(f"unsupported exported MoE expert type {self.expert_type!r}")

    def forward(self, x):
        if self.expert_type in ("swiglu", "geglu"):
            gate = self.w_gate(x)
            if self.expert_type == "swiglu":
                gate = torch.sigmoid(gate)
            else:
                gate = torch.nn.functional.gelu(gate, approximate="tanh")
            return self.w_down(gate * self.w_up(x))

        up = self.w_up(x)
        if self.activation == "silu":
            act = torch.nn.functional.silu(up)
        elif self.activation == "gelu":
            act = torch.nn.functional.gelu(up, approximate="tanh")
        elif self.activation == "relu":
            act = torch.relu(up)
        elif self.activation == "leaky_relu_sq":
            act = torch.nn.functional.leaky_relu(up, negative_slope=self.leaky_slope)
            act = act * act
        else:
            raise ValueError(f"unsupported exported MoE MLP activation {self.activation!r}")
        return self.w_down(act)


class MixlabMoEBlock(nn.Module):
    def __init__(self, config, block_config):
        super().__init__()
        dim = config.model_dim
        self.num_experts = int(block_config["num_experts"])
        self.top_k = int(block_config.get("top_k", 0) or min(2, self.num_experts))
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1,num_experts], got {self.top_k}")
        router = str(block_config.get("router", "") or "linear").lower()
        if router != "linear":
            raise ValueError(f"unsupported exported MoE router {router!r}")
        self.norm = MixlabRMSNorm(dim)
        self.router_w = nn.Parameter(torch.empty(dim, self.num_experts))
        nn.init.xavier_uniform_(self.router_w)
        expert_config = block_config.get("expert_block") or {"type": "swiglu"}
        self.experts = nn.ModuleList([MixlabMoEExpert(config, expert_config) for _ in range(self.num_experts)])

    def forward(self, x, dwa=None):
        x_norm = self.norm(x)
        router_logits = torch.matmul(x_norm, self.router_w)
        probs = torch.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / torch.clamp(top_probs.sum(dim=-1, keepdim=True), min=1e-12)
        dispatch = torch.zeros_like(probs).scatter(-1, top_idx, top_probs)
        flat_x = x_norm.reshape(-1, x_norm.shape[-1])
        flat_dispatch = dispatch.reshape(-1, self.num_experts)
        flat_delta = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            selected = torch.nonzero(flat_dispatch[:, expert_id] > 0, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            expert_out = expert(flat_x.index_select(0, selected))
            weight = flat_dispatch.index_select(0, selected)[:, expert_id].unsqueeze(-1)
            flat_delta.index_add_(0, selected, expert_out * weight)
        x = x + flat_delta.view_as(x)
        if dwa is not None:
            x = dwa.apply(x)
        return x


class MixlabSwiGLUBlock(nn.Module):
    def __init__(self, config, gate_activation="sigmoid"):
        super().__init__()
        dim = config.model_dim
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.gate_activation = gate_activation
        self.norm_placement = norm_placement(config)
        self.norm = make_mixlab_norm(config, dim) if self.norm_placement in ("pre", "sandwich") else None
        self.w_gate = MixlabLinear(dim, ffn_dim)
        self.w_up = MixlabLinear(dim, ffn_dim)
        self.internal_norm = make_mixlab_norm(config, ffn_dim) if bool(getattr(config, "ffn_internal_norm", False)) else None
        self.w_down = MixlabLinear(ffn_dim, dim)
        self.post_norm = make_mixlab_norm(config, dim) if self.norm_placement in ("post", "sandwich") else None

    def forward(self, x, dwa=None):
        x_norm = self.norm(x) if self.norm is not None else x
        gate = self.w_gate(x_norm)
        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(gate)
        elif self.gate_activation == "gelu":
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
        else:
            raise ValueError(f"unsupported exported gate activation {self.gate_activation!r}")
        gated = gate * self.w_up(x_norm)
        if self.internal_norm is not None:
            gated = self.internal_norm(gated)
        delta = self.w_down(gated)
        if self.post_norm is not None:
            delta = self.post_norm(delta)
        x = x + delta
        if dwa is not None:
            x = dwa.apply(x)
        return x


class MixlabMLPBlock(nn.Module):
    def __init__(self, config, block_config):
        super().__init__()
        dim = config.model_dim
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.activation = str(block_config.get("activation", "") or "silu").lower()
        self.leaky_slope = float(block_config.get("leaky_slope", 0.0) or 0.5)
        self.norm_placement = norm_placement(config)
        self.norm = make_mixlab_norm(config, dim) if self.norm_placement in ("pre", "sandwich") else None
        self.w_up = MixlabLinear(dim, ffn_dim)
        self.internal_norm = make_mixlab_norm(config, ffn_dim) if bool(getattr(config, "ffn_internal_norm", False)) else None
        self.w_down = MixlabLinear(ffn_dim, dim)
        self.post_norm = make_mixlab_norm(config, dim) if self.norm_placement in ("post", "sandwich") else None

    def forward(self, x, dwa=None):
        x_norm = self.norm(x) if self.norm is not None else x
        up = self.w_up(x_norm)
        if self.activation == "silu":
            act = torch.nn.functional.silu(up)
        elif self.activation == "gelu":
            act = torch.nn.functional.gelu(up, approximate="tanh")
        elif self.activation == "relu":
            act = torch.relu(up)
        elif self.activation == "leaky_relu_sq":
            act = torch.nn.functional.leaky_relu(up, negative_slope=self.leaky_slope)
            act = act * act
        else:
            raise ValueError(f"unsupported exported MLP activation {self.activation!r}")
        if self.internal_norm is not None:
            act = self.internal_norm(act)
        delta = self.w_down(act)
        if self.post_norm is not None:
            delta = self.post_norm(delta)
        x = x + delta
        if dwa is not None:
            x = dwa.apply(x)
        return x


def load_char_lookup(config):
    if int(getattr(config, "char_vocab_size", 0) or 0) <= 0:
        return None
    filename = getattr(config, "char_features_file", "") or "char_features.bin"
    base = getattr(config, "_name_or_path", "") or ""
    path = os.path.join(base, filename) if base else filename
    if not os.path.exists(path):
        raise ValueError(f"char features enabled but {path!r} is missing")
    vocab = int(config.vocab_size)
    char_vocab = int(config.char_vocab_size)
    slots = int(config.char_max_per_token)
    header_ints = 256
    header_bytes = header_ints * 4
    want = header_bytes + vocab * slots * 2
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) != want:
        raise ValueError(f"char feature file size={len(blob)} want={want}")
    header = struct.unpack("<" + "i" * header_ints, blob[:header_bytes])
    if header[0] != 20260526 or header[1] != 1:
        raise ValueError("invalid char feature magic/version")
    if header[2] != vocab or header[3] != char_vocab or header[4] != slots:
        raise ValueError("char feature metadata does not match config")
    payload = torch.tensor(
        struct.unpack("<" + "H" * (vocab * slots), blob[header_bytes:]),
        dtype=torch.long,
    )
    return payload.view(vocab, slots)


class MixlabModel(PreTrainedModel):
    config_class = MixlabConfig
    base_model_prefix = "mixlab"
    supports_gradient_checkpointing = True

    def __init__(self, config, blocks=None):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embeddings = None
        self.positional_embedding = str(getattr(config, "positional_embedding", "rope") or "rope").lower()
        if self.positional_embedding == "learned_absolute":
            self.position_embeddings = nn.Embedding(int(config.max_position_embeddings), config.model_dim)
        elif self.positional_embedding not in ("rope", "none"):
            raise ValueError(f"unsupported positional_embedding {self.positional_embedding!r}")
        self.embed_dropout = nn.Dropout(float(getattr(config, "embedding_dropout", 0.0) or 0.0))
        self.char_table = None
        self.char_proj = None
        self.char_scale = None
        self.bigram_table = None
        self.bigram_proj = None
        self.bigram_scale = None
        self.trigram_table = None
        self.trigram_proj = None
        self.trigram_scale = None
        if int(getattr(config, "char_vocab_size", 0) or 0) > 0:
            char_dim = int(getattr(config, "char_dim", 0) or config.model_dim)
            self.char_table = nn.Embedding(int(config.char_vocab_size), char_dim)
            if char_dim != config.model_dim:
                self.char_proj = MixlabLinear(char_dim, config.model_dim)
            self.char_scale = nn.Parameter(torch.ones(1))
            # Loaded lazily on first forward rather than registered as a buffer:
            # `from_pretrained` meta-device init zeroes non-persistent buffers,
            # which would silently drop the char feature channel.
            self.char_lookup = None
        if int(getattr(config, "bigram_vocab_size", 0) or 0) > 0:
            bigram_dim = int(getattr(config, "bigram_dim", 0) or config.model_dim)
            self.bigram_table = nn.Embedding(int(config.bigram_vocab_size), bigram_dim)
            if bigram_dim != config.model_dim:
                self.bigram_proj = MixlabLinear(bigram_dim, config.model_dim)
            self.bigram_scale = nn.Parameter(torch.ones(1))
        if int(getattr(config, "trigram_vocab_size", 0) or 0) > 0:
            trigram_dim = int(getattr(config, "trigram_dim", 0) or config.model_dim)
            self.trigram_table = nn.Embedding(int(config.trigram_vocab_size), trigram_dim)
            if trigram_dim != config.model_dim:
                self.trigram_proj = MixlabLinear(trigram_dim, config.model_dim)
            self.trigram_scale = nn.Parameter(torch.ones(1))
        modules = []
        block_configs = blocks if blocks is not None else self._default_backbone_blocks(config)
        self.layer_aggregation = str(getattr(config, "layer_aggregation", "") or "none").lower()
        if self.layer_aggregation in ("", "none"):
            self.layer_aggregation = "none"
        if self.layer_aggregation not in ("none", "dwa"):
            raise ValueError(f"unsupported exported layer_aggregation {self.layer_aggregation!r}")
        self.layer_aggregation_scope = str(getattr(config, "layer_aggregation_scope", "") or "trunk").lower()
        if self.layer_aggregation == "none":
            self.layer_aggregation_scope = "none"
        elif self.layer_aggregation_scope in ("", "default", "inline"):
            self.layer_aggregation_scope = "trunk"
        if self.layer_aggregation == "dwa" and self.layer_aggregation_scope not in ("trunk", "head"):
            raise ValueError(
                f"unsupported exported layer_aggregation_scope {self.layer_aggregation_scope!r}"
            )
        self.dwa_alphas = nn.ParameterList()
        if self.layer_aggregation == "dwa":
            for idx in range(count_dwa_points(block_configs)):
                alpha = torch.zeros(idx + 2)
                alpha[-1] = 1.0
                self.dwa_alphas.append(nn.Parameter(alpha))
        self.relative_embeddings = None
        self.relative_layer_norm = None
        shared_relative_window = None
        shared_relative_embedding_norm = None
        for block in block_configs:
            if block.get("type") != "plain":
                continue
            rel = str(block.get("relative_attention", "") or "none").lower()
            param = str(block.get("relative_attention_parameterization", "") or "per_block_projections").lower()
            if rel == "deberta_p2c_c2p" and param == "shared_qk_reuse":
                window = int(block.get("relative_attention_window", 0) or 128)
                embedding_norm = normalize_relative_embedding_norm(block.get("relative_attention_embedding_norm", "none"))
                if embedding_norm not in ("none", "layernorm"):
                    raise ValueError(
                        f"unsupported exported relative_attention_embedding_norm {embedding_norm!r}"
                    )
                if shared_relative_window is None:
                    shared_relative_window = window
                    shared_relative_embedding_norm = embedding_norm
                elif shared_relative_window != window:
                    raise ValueError("shared_qk_reuse blocks must use one relative_attention_window")
                elif shared_relative_embedding_norm != embedding_norm:
                    raise ValueError("shared_qk_reuse blocks must use one relative_attention_embedding_norm")
        if shared_relative_window is not None:
            rel_rows = 2 * shared_relative_window - 1
            self.relative_embeddings = nn.Parameter(torch.empty(rel_rows, config.model_dim))
            nn.init.xavier_uniform_(self.relative_embeddings)
            if shared_relative_embedding_norm == "layernorm":
                self.relative_layer_norm = nn.LayerNorm(
                    config.model_dim,
                    eps=float(getattr(config, "norm_eps", 1e-5) or 1e-5),
                    elementwise_affine=True,
                )
        self.mlm_head = str(getattr(config, "mlm_head", "linear") or "linear").lower()
        self.mlm_head_norm1 = None
        self.mlm_head_dense = None
        self.mlm_head_norm2 = None
        self.mlm_head_dropout = None
        self.mlm_head_output_bias = None
        if self.mlm_head == "bert":
            eps = float(getattr(config, "norm_eps", 1e-5) or 1e-5)
            self.mlm_head_norm1 = nn.LayerNorm(config.model_dim, eps=eps, elementwise_affine=False)
            self.mlm_head_dense = MixlabLinear(config.model_dim, config.model_dim, bias=True)
            self.mlm_head_norm2 = nn.LayerNorm(config.model_dim, eps=eps, elementwise_affine=False)
            self.mlm_head_dropout = nn.Dropout(float(getattr(config, "hidden_dropout", 0.0) or 0.0))
            self.mlm_head_output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        elif self.mlm_head not in ("", "linear", "none", "default"):
            raise ValueError(f"unsupported exported mlm_head {self.mlm_head!r}")
        for block in block_configs:
            block_type = block.get("type")
            if block_type == "plain":
                modules.append(MixlabPlainBlock(config, block))
            elif block_type == "swiglu":
                modules.append(MixlabSwiGLUBlock(config, "sigmoid"))
            elif block_type == "geglu":
                modules.append(MixlabSwiGLUBlock(config, "gelu"))
            elif block_type == "mlp":
                modules.append(MixlabMLPBlock(config, block))
            elif block_type == "moe":
                modules.append(MixlabMoEBlock(config, block))
            elif block_type == "ttt_mlp":
                modules.append(MixlabTTTMLPBlock(config, block))
            else:
                raise ValueError(f"unsupported exported Mixlab block type {block_type!r}")
        self.blocks = nn.ModuleList(modules)
        self.final_norm = make_mixlab_norm(config, config.model_dim)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _bigram_ids(self, input_ids):
        vocab = int(self.config.bigram_vocab_size)
        # Match Mixlab native feature construction, which hashes the flattened
        # token buffer and intentionally carries the predecessor across rows.
        flat = input_ids.reshape(-1).to(torch.long)
        ids = torch.empty_like(flat)
        if flat.numel() == 0:
            return ids.view_as(input_ids)
        modulus = vocab - 1
        ids[0] = modulus
        if flat.numel() > 1:
            ids[1:] = ((36313 * flat[1:]) ^ (27191 * flat[:-1])) % modulus
        return ids.view_as(input_ids)

    def _trigram_ids(self, input_ids):
        vocab = int(self.config.trigram_vocab_size)
        modulus = vocab - 1
        out = torch.zeros_like(input_ids, dtype=torch.long)
        if input_ids.shape[1] <= 2:
            return out
        t0 = input_ids[:, :-2].to(torch.long)
        t1 = input_ids[:, 1:-1].to(torch.long)
        t2 = input_ids[:, 2:].to(torch.long)
        h = ((t0 * vocab + t1) * vocab + t2) % modulus
        out[:, 2:] = h + 1
        return out

    def _embed_features(self, input_ids):
        x = self.embed_tokens(input_ids)
        if self.position_embeddings is not None:
            seq_len = input_ids.shape[1]
            if seq_len > int(self.config.max_position_embeddings):
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}"
                )
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            x = x + self.position_embeddings(position_ids).unsqueeze(0)
        if self.char_table is not None:
            if self.char_lookup is None:
                self.char_lookup = load_char_lookup(self.config).to(input_ids.device)
            char_ids = self.char_lookup[input_ids]
            char_emb = self.char_table(char_ids)
            char_emb = char_emb * char_ids.ne(0).unsqueeze(-1)
            char_state = char_emb.sum(dim=-2)
            if self.char_proj is not None:
                char_state = self.char_proj(char_state)
            x = x + char_state * self.char_scale
        if self.bigram_table is not None:
            ids = self._bigram_ids(input_ids)
            state = self.bigram_table(ids)
            if self.bigram_proj is not None:
                state = self.bigram_proj(state)
            x = x + state * self.bigram_scale
        if self.trigram_table is not None:
            ids = self._trigram_ids(input_ids)
            state = self.trigram_table(ids)
            if self.trigram_proj is not None:
                state = self.trigram_proj(state)
            x = x + state * self.trigram_scale
        x = self.embed_dropout(x)
        return x

    @staticmethod
    def _default_backbone_blocks(config):
        masked_blocks = getattr(config, "masked_blocks", None) or []
        return masked_blocks if masked_blocks else config.blocks

    def forward_hidden_with_state(
        self,
        input_ids=None,
        attention_mask=None,
        ttt_state=None,
        use_cache=False,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")
        x = self._embed_features(input_ids)
        dwa = None
        if self.layer_aggregation == "dwa":
            if self.layer_aggregation_scope == "head":
                dwa = MixlabDWAHeadState(self.dwa_alphas, x)
            else:
                dwa = MixlabDWAState(self.dwa_alphas, x)
        relative_embeddings = self.relative_embeddings
        if relative_embeddings is not None and self.relative_layer_norm is not None:
            relative_embeddings = self.relative_layer_norm(relative_embeddings)
        ttt_blocks = sum(isinstance(block, MixlabTTTMLPBlock) for block in self.blocks)
        if ttt_blocks:
            require_right_padded_ttt_batch(attention_mask)
        if ttt_state is None:
            ttt_state = (None,) * ttt_blocks
        elif len(ttt_state) != ttt_blocks:
            raise ValueError(f"ttt_state has {len(ttt_state)} blocks, expected {ttt_blocks}")
        next_ttt_state = []
        ttt_index = 0
        for block in self.blocks:
            if isinstance(block, MixlabPlainBlock):
                x = block(x, relative_embeddings, dwa, attention_mask)
            elif isinstance(block, MixlabTTTMLPBlock):
                x, block_state = block(
                    x,
                    dwa=dwa,
                    state=ttt_state[ttt_index],
                    use_cache=use_cache,
                )
                if use_cache:
                    next_ttt_state.append(block_state)
                ttt_index += 1
            else:
                x = block(x, dwa)
        if dwa is not None:
            if self.layer_aggregation_scope == "head":
                x = dwa.finish()
            else:
                dwa.finish()
        return self.final_norm(x), tuple(next_ttt_state) if use_cache else None

    def forward_hidden(self, input_ids=None, attention_mask=None):
        hidden, _ = self.forward_hidden_with_state(input_ids, attention_mask)
        return hidden

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return BaseModelOutput(last_hidden_state=self.forward_hidden(input_ids, attention_mask))

    def bert_mlm_logits(self, hidden):
        if self.mlm_head != "bert":
            raise ValueError("bert_mlm_logits requires mlm_head='bert'")
        x = self.mlm_head_norm1(hidden)
        x = self.mlm_head_dense(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.mlm_head_norm2(x)
        x = self.mlm_head_dropout(x)
        return torch.matmul(x, self.embed_tokens.weight.transpose(0, 1)) + self.mlm_head_output_bias


class MixlabForCausalLM(MixlabModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config, blocks=config.blocks)
        self.lm_head_weight = nn.Parameter(torch.empty(config.model_dim, config.vocab_size))
        nn.init.xavier_uniform_(self.lm_head_weight)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        ttt_state=None,
        use_cache=None,
        **kwargs,
    ):
        if ttt_state is not None and past_key_values is not None:
            raise ValueError("pass only one of ttt_state or past_key_values")
        if ttt_state is None:
            ttt_state = past_key_values
        use_cache = bool(use_cache) or ttt_state is not None
        x, next_ttt_state = self.forward_hidden_with_state(
            input_ids,
            attention_mask,
            ttt_state=ttt_state,
            use_cache=use_cache,
        )
        logits = torch.matmul(x, self.lm_head_weight)
        if getattr(self.config, "logit_softcap", 0.0):
            cap = float(self.config.logit_softcap)
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_ttt_state,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if past_key_values is None:
            return None
        return tuple(state.index_select(beam_idx) for state in past_key_values)


class MixlabForMaskedLM(MixlabModel):
    _tied_weights_keys = []

    def __init__(self, config):
        masked_blocks = getattr(config, "masked_blocks", None) or config.blocks
        super().__init__(config, blocks=masked_blocks)
        self.lm_head_weight = nn.Parameter(torch.empty(config.model_dim, config.vocab_size))
        nn.init.xavier_uniform_(self.lm_head_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.forward_hidden(input_ids, attention_mask)
        if getattr(self.config, "mlm_head", "linear") == "bert":
            logits = self.bert_mlm_logits(x)
        else:
            logits = torch.matmul(x, self.lm_head_weight)
        if getattr(self.config, "logit_softcap", 0.0):
            cap = float(self.config.logit_softcap)
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)


class MixlabForSequenceClassification(MixlabModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = int(config.num_labels)
        self.sequence_classification_pooling = str(
            getattr(config, "sequence_classification_pooling", "") or ""
        ).lower()
        if self.sequence_classification_pooling not in ("last", "mean"):
            raise ValueError(
                "sequence_classification_pooling must be 'last' or 'mean'; "
                "pass an explicit value when loading an ambiguous exported backbone"
            )
        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = float(getattr(config, "hidden_dropout", 0.0) or 0.0)
        self.classifier_dropout = nn.Dropout(float(classifier_dropout))
        # This task head is intentionally absent from the Mixlab checkpoint and
        # receives PyTorch's standard fresh Linear initialization for fine-tuning.
        self.classifier = nn.Linear(config.model_dim, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.forward_hidden(input_ids, attention_mask)
        pooled = pool_sequence(
            hidden,
            attention_mask,
            self.sequence_classification_pooling,
        )
        logits = self.classifier(self.classifier_dropout(pooled))

        loss = None
        if labels is not None:
            problem_type = getattr(self.config, "problem_type", None)
            if problem_type is None:
                if self.num_labels == 1:
                    problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    problem_type = "single_label_classification"
                else:
                    problem_type = "multi_label_classification"
                self.config.problem_type = problem_type

            if problem_type == "regression":
                if self.num_labels == 1:
                    loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels.squeeze(-1))
                else:
                    loss = torch.nn.functional.mse_loss(logits, labels)
            elif problem_type == "single_label_classification":
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif problem_type == "multi_label_classification":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            else:
                raise ValueError(f"unsupported problem_type {problem_type!r}")

        return SequenceClassifierOutput(loss=loss, logits=logits)
