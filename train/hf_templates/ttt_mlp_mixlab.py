from dataclasses import dataclass

import torch
from torch import nn


class TTTMLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)


class TTTMLPRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


@dataclass
class MixlabTTTMLPState:
    """Request-owned TTT state for one exported block."""

    mlp: torch.Tensor
    gradient: torch.Tensor
    conv: torch.Tensor
    offset: int

    def index_select(self, batch_indices):
        return MixlabTTTMLPState(
            mlp=self.mlp.index_select(0, batch_indices),
            gradient=self.gradient.index_select(0, batch_indices),
            conv=self.conv.index_select(0, batch_indices),
            offset=self.offset,
        )


def _gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def _gelu_tanh_derivative(x):
    tanh_out = torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x))
    return (
        0.5
        * x
        * (1.0 - tanh_out * tanh_out)
        * (0.79788456 + 0.1070322243 * x * x)
        + 0.5 * (1.0 + tanh_out)
    )


def _layer_norm(x, scale, bias, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(variance + eps) * scale + bias


def _layer_norm_l2_vjp(x, target, scale, bias, eps=1e-6):
    dim = x.shape[-1]
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    std = torch.sqrt(variance + eps)
    x_hat = centered / std
    grad_output = scale * x_hat + bias - target
    grad_x_hat = grad_output * scale
    return (
        grad_x_hat * dim
        - grad_x_hat.sum(dim=-1, keepdim=True)
        - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
    ) / (dim * std)


def _adjacent_rope_at(x, position):
    dim = x.shape[-1]
    pair_count = dim // 2
    dim_idx = torch.arange(pair_count, device=x.device, dtype=x.dtype)
    freqs = torch.exp(dim_idx * (-torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype)) * 2.0 / dim))
    angles = float(position) * freqs
    cos_t = torch.cos(angles).view(1, 1, 1, pair_count)
    sin_t = torch.sin(angles).view(1, 1, 1, pair_count)
    pairs = x.view(*x.shape[:-1], pair_count, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    return torch.stack((even * cos_t - odd * sin_t, even * sin_t + odd * cos_t), dim=-1).reshape_as(x)


def _causal_depthwise_conv(x, history, weight):
    # Mixlab stores taps in convolution order, so the newest sample uses the
    # final column and the oldest retained sample uses column zero.
    token_count = x.shape[1]
    combined = torch.cat((history, x), dim=1)
    out = torch.zeros_like(x)
    for lag in range(4):
        shifted = combined[:, 3 - lag : 3 - lag + token_count, :]
        out = out + shifted * weight[:, 3 - lag].view(1, 1, -1)
    return out, combined[:, -3:, :]


class MixlabTTTMLPBlock(nn.Module):
    """Exact Mixlab TTT-MLP forward with an explicit persistent cache."""

    def __init__(self, config, block_config):
        super().__init__()
        dim = int(config.model_dim)
        self.dim = dim
        self.heads = int(block_config["heads"])
        if self.heads <= 0 or dim % self.heads != 0:
            raise ValueError("ttt_mlp requires model_dim divisible by heads")
        self.head_dim = dim // self.heads
        if self.head_dim % 2 != 0:
            raise ValueError("ttt_mlp requires an even head_dim")
        self.chunk_size = int(block_config.get("chunk_size", 0) or 16)
        hidden_mult = float(block_config.get("inner_hidden_mult", 0.0) or 4.0)
        self.hidden_dim = int(round(self.head_dim * hidden_mult))
        self.inner_lr_base = float(block_config.get("inner_lr_base", 0.0) or 0.1)

        self.norm = TTTMLPRMSNorm(dim, eps=1e-6)
        self.w_qk = TTTMLPLinear(dim, dim)
        self.q_conv_weight = nn.Parameter(torch.empty(dim, 4))
        self.q_conv_bias = nn.Parameter(torch.zeros(dim))
        self.k_conv_weight = nn.Parameter(torch.empty(dim, 4))
        self.k_conv_bias = nn.Parameter(torch.zeros(dim))
        self.w_v = TTTMLPLinear(dim, dim)
        self.inner_lr_w = TTTMLPLinear(dim, self.heads)
        self.inner_lr_bias = nn.Parameter(torch.zeros(self.heads))
        self.inner_token_coeff = nn.Parameter(torch.zeros(self.chunk_size))
        self.inner_w1 = nn.Parameter(torch.empty(self.heads * self.head_dim, self.hidden_dim))
        self.inner_b1 = nn.Parameter(torch.zeros(self.heads, self.hidden_dim))
        self.inner_w2 = nn.Parameter(torch.empty(self.heads * self.hidden_dim, self.head_dim))
        self.inner_b2 = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.inner_norm_scale = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.inner_norm_bias = nn.Parameter(torch.zeros(self.heads, self.head_dim))
        self.post_norm = nn.LayerNorm(dim, eps=1e-6)
        self.w_out_gate = TTTMLPLinear(dim, dim)
        self.w_out = TTTMLPLinear(dim, dim)

    @property
    def state_size(self):
        return self.heads * (
            2 * self.head_dim * self.hidden_dim + self.hidden_dim + self.head_dim
        )

    def initial_state(self, batch_size, device, dtype=torch.float32):
        parts = (
            self.inner_w1.reshape(1, -1),
            self.inner_b1.reshape(1, -1),
            self.inner_w2.reshape(1, -1),
            self.inner_b2.reshape(1, -1),
        )
        mlp = torch.cat(parts, dim=1).to(device=device, dtype=dtype).expand(batch_size, -1).clone()
        return MixlabTTTMLPState(
            mlp=mlp,
            gradient=torch.zeros_like(mlp),
            conv=torch.zeros(batch_size, 2, 3, self.dim, device=device, dtype=dtype),
            offset=0,
        )

    def _unpack(self, packed):
        batch = packed.shape[0]
        w1_size = self.heads * self.head_dim * self.hidden_dim
        b1_size = self.heads * self.hidden_dim
        w2_size = self.heads * self.hidden_dim * self.head_dim
        cursor = 0
        w1 = packed[:, cursor : cursor + w1_size].reshape(
            batch, self.heads, self.head_dim, self.hidden_dim
        )
        cursor += w1_size
        b1 = packed[:, cursor : cursor + b1_size].reshape(batch, self.heads, 1, self.hidden_dim)
        cursor += b1_size
        w2 = packed[:, cursor : cursor + w2_size].reshape(
            batch, self.heads, self.hidden_dim, self.head_dim
        )
        cursor += w2_size
        b2 = packed[:, cursor:].reshape(batch, self.heads, 1, self.head_dim)
        return w1, b1, w2, b2

    @staticmethod
    def _pack(parts):
        batch = parts[0].shape[0]
        return torch.cat(tuple(part.reshape(batch, -1) for part in parts), dim=1)

    def _segment(self, qk, value, lr_logits, state):
        batch, token_count, _ = qk.shape
        if state.offset < 0 or state.offset + token_count > self.chunk_size:
            raise ValueError("ttt_mlp cached segment crosses a chunk boundary")
        work_dtype = torch.float32
        qk = qk.to(work_dtype)
        value = value.to(work_dtype)
        lr_logits = lr_logits.to(work_dtype)
        q_history = state.conv[:, 0].to(work_dtype)
        k_history = state.conv[:, 1].to(work_dtype)
        q_seq, q_next = _causal_depthwise_conv(qk, q_history, self.q_conv_weight.to(work_dtype))
        k_seq, k_next = _causal_depthwise_conv(qk, k_history, self.k_conv_weight.to(work_dtype))
        q_seq = q_seq + self.q_conv_bias.to(work_dtype).view(1, 1, -1)
        k_seq = k_seq + self.k_conv_bias.to(work_dtype).view(1, 1, -1)

        q = q_seq.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        k = k_seq.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        v = value.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        lr = lr_logits.view(batch, token_count, self.heads).transpose(1, 2)
        w1, b1, w2, b2 = self._unpack(state.mlp.to(work_dtype))
        grad_w1, grad_b1, grad_w2, grad_b2 = self._unpack(state.gradient.to(work_dtype))
        norm_scale = self.inner_norm_scale.to(work_dtype).view(1, self.heads, 1, self.head_dim)
        norm_bias = self.inner_norm_bias.to(work_dtype).view(1, self.heads, 1, self.head_dim)

        outputs = []
        for token_idx in range(token_count):
            position = state.offset + token_idx
            qt = _adjacent_rope_at(q[:, :, token_idx : token_idx + 1], position)
            kt = _adjacent_rope_at(k[:, :, token_idx : token_idx + 1], position)
            vt = v[:, :, token_idx : token_idx + 1]
            z1 = torch.matmul(kt, w1) + b1
            x2 = _gelu_tanh(z1)
            z2 = torch.matmul(x2, w2) + b2
            target = vt - kt
            grad_z2 = _layer_norm_l2_vjp(z2, target, norm_scale, norm_bias)
            grad_z1 = torch.matmul(grad_z2, w2.transpose(-2, -1)) * _gelu_tanh_derivative(z1)
            gate = (
                torch.sigmoid(lr[:, :, token_idx]).view(batch, self.heads, 1, 1)
                * (self.inner_lr_base / self.head_dim)
            )
            grad_w1 = grad_w1 + gate * torch.matmul(kt.transpose(-2, -1), grad_z1)
            grad_b1 = grad_b1 + gate * grad_z1
            grad_w2 = grad_w2 + gate * torch.matmul(x2.transpose(-2, -1), grad_z2)
            grad_b2 = grad_b2 + gate * grad_z2

            coeff = torch.clamp(
                self.inner_token_coeff[position].to(work_dtype) + 1.0 / float(position + 1),
                min=0.0,
            )
            query_w1 = w1 - coeff * grad_w1
            query_b1 = b1 - coeff * grad_b1
            query_w2 = w2 - coeff * grad_w2
            query_b2 = b2 - coeff * grad_b2
            query_z1 = torch.matmul(qt, query_w1) + query_b1
            query_z2 = torch.matmul(_gelu_tanh(query_z1), query_w2) + query_b2
            outputs.append(qt + _layer_norm(query_z2, norm_scale, norm_bias))

            if position + 1 == self.chunk_size:
                w1, b1, w2, b2 = query_w1, query_b1, query_w2, query_b2
                grad_w1 = torch.zeros_like(grad_w1)
                grad_b1 = torch.zeros_like(grad_b1)
                grad_w2 = torch.zeros_like(grad_w2)
                grad_b2 = torch.zeros_like(grad_b2)

        scan = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, token_count, self.dim)
        next_state = MixlabTTTMLPState(
            mlp=self._pack((w1, b1, w2, b2)),
            gradient=self._pack((grad_w1, grad_b1, grad_w2, grad_b2)),
            conv=torch.stack((q_next, k_next), dim=1),
            offset=(state.offset + token_count) % self.chunk_size,
        )
        return scan, next_state

    def forward(self, x, dwa=None, state=None, use_cache=False):
        batch, token_count, _ = x.shape
        if token_count <= 0:
            raise ValueError("ttt_mlp requires at least one token")
        x_norm = self.norm(x)
        qk = self.w_qk(x_norm)
        value = self.w_v(x_norm)
        lr_logits = self.inner_lr_w(x_norm) + self.inner_lr_bias
        if state is None:
            state = self.initial_state(batch, x.device)
        elif state.mlp.shape != (batch, self.state_size):
            raise ValueError(
                f"ttt_mlp state shape {tuple(state.mlp.shape)} != ({batch}, {self.state_size})"
            )

        outputs = []
        start = 0
        while start < token_count:
            segment_len = min(token_count - start, self.chunk_size - state.offset)
            scan, state = self._segment(
                qk[:, start : start + segment_len],
                value[:, start : start + segment_len],
                lr_logits[:, start : start + segment_len],
                state,
            )
            outputs.append(scan)
            start += segment_len
        scan = torch.cat(outputs, dim=1).to(dtype=x.dtype)
        post_norm = self.post_norm(scan)
        gate = _gelu_tanh(self.w_out_gate(x_norm))
        delta = self.w_out(post_norm * gate)
        x = x + delta
        if dwa is not None:
            x = dwa.apply(x)
        return x, state if use_cache else None
