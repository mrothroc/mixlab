from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


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


def _adjacent_rope(x, offset=0):
    dim = x.shape[-1]
    pair_count = dim // 2
    token_count = x.shape[-2]
    dim_idx = torch.arange(pair_count, device=x.device, dtype=x.dtype)
    freqs = torch.exp(dim_idx * (-math.log(10000.0) * 2.0 / dim))
    positions = torch.arange(offset, offset + token_count, device=x.device, dtype=x.dtype)
    angles = positions.view(token_count, 1) * freqs.view(1, pair_count)
    cos_t = torch.cos(angles).view(1, 1, token_count, pair_count)
    sin_t = torch.sin(angles).view(1, 1, token_count, pair_count)
    pairs = x.view(*x.shape[:-1], pair_count, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    return torch.stack((even * cos_t - odd * sin_t, even * sin_t + odd * cos_t), dim=-1).reshape_as(x)


def require_right_padded_ttt_batch(attention_mask):
    """Reject padding that a TTT recurrence cannot honor.

    TTT consumes every token into the inner MLP state, so a leading or interior pad
    would perform inner updates and shift chunk-relative positions before the first
    real token. Trailing pads are safe: causality keeps them out of earlier positions.
    Attention-style rank-3/4 masks are not padding masks and are left alone.
    """
    if attention_mask is None or attention_mask.dim() != 2:
        return
    mask = attention_mask.to(torch.bool)
    if bool(torch.all(mask[:, 1:] <= mask[:, :-1])):
        return
    raise ValueError(
        "ttt_mlp requires right-padded batches: a pad before a real token would "
        "advance the recurrent state and shift chunk-relative positions. Set the "
        "tokenizer to padding_side='right', or bucket sequences by length."
    )


def _causal_depthwise_conv(x, history, weight):
    combined = torch.cat((history, x), dim=1)
    # conv1d is cross-correlation: weight[:, 0] consumes the oldest retained
    # sample and weight[:, 3] consumes the current token, matching Mixlab's
    # stored convolution order.
    out = F.conv1d(
        combined.transpose(1, 2),
        weight.unsqueeze(1),
        groups=x.shape[-1],
    ).transpose(1, 2)
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
        # Full-meta-gradient fine-tuning is memory intensive. Checkpoint the
        # stateless scan by default and let Hugging Face's standard model-level
        # controls disable it when a caller explicitly prefers backward speed.
        self.gradient_checkpointing = True

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

    def _online_segment(self, qk, value, lr_logits, state):
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
            qt = _adjacent_rope(q[:, :, token_idx : token_idx + 1], position)
            kt = _adjacent_rope(k[:, :, token_idx : token_idx + 1], position)
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

    def _stateless_online_scan(self, qk, value, lr_logits):
        state = self.initial_state(qk.shape[0], qk.device)
        outputs = []
        start = 0
        while start < qk.shape[1]:
            segment_len = min(qk.shape[1] - start, self.chunk_size - state.offset)
            scan, state = self._online_segment(
                qk[:, start : start + segment_len],
                value[:, start : start + segment_len],
                lr_logits[:, start : start + segment_len],
                state,
            )
            outputs.append(scan)
            start += segment_len
        return torch.cat(outputs, dim=1)

    def _stateless_dual_scan(self, qk, value, lr_logits):
        """Vectorized full-sequence scan equivalent to the online recurrence."""
        batch, token_count, _ = qk.shape
        work_dtype = torch.float32
        qk = qk.to(work_dtype)
        value = value.to(work_dtype)
        lr_logits = lr_logits.to(work_dtype)
        history = torch.zeros(batch, 3, self.dim, device=qk.device, dtype=work_dtype)
        q_seq, _ = _causal_depthwise_conv(
            qk, history, self.q_conv_weight.to(work_dtype)
        )
        k_seq, _ = _causal_depthwise_conv(
            qk, history, self.k_conv_weight.to(work_dtype)
        )
        q_seq = q_seq + self.q_conv_bias.to(work_dtype).view(1, 1, -1)
        k_seq = k_seq + self.k_conv_bias.to(work_dtype).view(1, 1, -1)

        q_all = q_seq.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        k_all = k_seq.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        v_all = value.view(batch, token_count, self.heads, self.head_dim).transpose(1, 2)
        lr_all = lr_logits.view(batch, token_count, self.heads).transpose(1, 2)
        w1 = self.inner_w1.to(work_dtype).reshape(
            self.heads, self.head_dim, self.hidden_dim
        ).unsqueeze(0).expand(batch, -1, -1, -1)
        b1 = self.inner_b1.to(work_dtype).reshape(
            1, self.heads, 1, self.hidden_dim
        ).expand(batch, -1, -1, -1)
        w2 = self.inner_w2.to(work_dtype).reshape(
            self.heads, self.hidden_dim, self.head_dim
        ).unsqueeze(0).expand(batch, -1, -1, -1)
        b2 = self.inner_b2.to(work_dtype).reshape(
            1, self.heads, 1, self.head_dim
        ).expand(batch, -1, -1, -1)
        norm_scale = self.inner_norm_scale.to(work_dtype).view(
            1, self.heads, 1, self.head_dim
        )
        norm_bias = self.inner_norm_bias.to(work_dtype).view(
            1, self.heads, 1, self.head_dim
        )

        outputs = []
        for start in range(0, token_count, self.chunk_size):
            end = min(token_count, start + self.chunk_size)
            chunk_tokens = end - start
            q = _adjacent_rope(q_all[:, :, start:end])
            k = _adjacent_rope(k_all[:, :, start:end])
            v = v_all[:, :, start:end]
            lr_gate = torch.sigmoid(lr_all[:, :, start:end])
            positions = torch.arange(
                1, chunk_tokens + 1, device=q.device, dtype=work_dtype
            )
            token_coeff = torch.clamp(
                positions.reciprocal()
                + self.inner_token_coeff[:chunk_tokens].to(work_dtype),
                min=0.0,
            )
            eta = (
                token_coeff.view(1, 1, chunk_tokens, 1)
                * lr_gate.view(batch, self.heads, 1, chunk_tokens)
                * (self.inner_lr_base / self.head_dim)
            )
            eta_lower = torch.tril(eta)

            z1 = torch.matmul(k, w1) + b1
            x2 = _gelu_tanh(z1)
            z2 = torch.matmul(x2, w2) + b2
            target = v - k
            grad_z2 = _layer_norm_l2_vjp(
                z2, target, norm_scale, norm_bias
            )
            grad_z1 = (
                torch.matmul(grad_z2, w2.transpose(-2, -1))
                * _gelu_tanh_derivative(z1)
            )

            attn1 = torch.tril(torch.matmul(q, k.transpose(-2, -1)))
            b1_bar = b1 - torch.matmul(eta_lower, grad_z1)
            z1_bar = (
                torch.matmul(q, w1)
                - torch.matmul(eta * attn1, grad_z1)
                + b1_bar
            )
            x2_bar = _gelu_tanh(z1_bar)
            attn2 = torch.tril(torch.matmul(x2_bar, x2.transpose(-2, -1)))
            b2_bar = b2 - torch.matmul(eta_lower, grad_z2)
            z2_bar = (
                torch.matmul(x2_bar, w2)
                - torch.matmul(eta * attn2, grad_z2)
                + b2_bar
            )
            outputs.append(q + _layer_norm(z2_bar, norm_scale, norm_bias))

            last_eta = eta[:, :, -1, :].unsqueeze(-1)
            w1 = w1 - torch.matmul((last_eta * k).transpose(-2, -1), grad_z1)
            b1 = b1 - torch.sum(last_eta * grad_z1, dim=2, keepdim=True)
            w2 = w2 - torch.matmul((last_eta * x2).transpose(-2, -1), grad_z2)
            b2 = b2 - torch.sum(last_eta * grad_z2, dim=2, keepdim=True)

        return torch.cat(outputs, dim=2).transpose(1, 2).reshape(
            batch, token_count, self.dim
        )

    def forward(self, x, dwa=None, state=None, use_cache=False):
        batch, token_count, _ = x.shape
        if token_count <= 0:
            raise ValueError("ttt_mlp requires at least one token")
        x_norm = self.norm(x)
        qk = self.w_qk(x_norm)
        value = self.w_v(x_norm)
        lr_logits = self.inner_lr_w(x_norm) + self.inner_lr_bias
        if state is not None and state.mlp.shape != (batch, self.state_size):
            raise ValueError(
                f"ttt_mlp state shape {tuple(state.mlp.shape)} != ({batch}, {self.state_size})"
            )

        if state is None and not use_cache:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                checkpoint_fn = getattr(self, "_gradient_checkpointing_func", None)
                if checkpoint_fn is None:
                    scan = torch_checkpoint(
                        self._stateless_dual_scan,
                        qk,
                        value,
                        lr_logits,
                        use_reentrant=False,
                    )
                else:
                    scan = checkpoint_fn(
                        self._stateless_dual_scan,
                        qk,
                        value,
                        lr_logits,
                    )
            else:
                scan = self._stateless_dual_scan(qk, value, lr_logits)
            next_state = None
        else:
            if state is None:
                state = self.initial_state(batch, x.device)
            outputs = []
            start = 0
            while start < token_count:
                segment_len = min(token_count - start, self.chunk_size - state.offset)
                segment, state = self._online_segment(
                    qk[:, start : start + segment_len],
                    value[:, start : start + segment_len],
                    lr_logits[:, start : start + segment_len],
                    state,
                )
                outputs.append(segment)
                start += segment_len
            scan = torch.cat(outputs, dim=1)
            next_state = state
        scan = scan.to(dtype=x.dtype)
        post_norm = self.post_norm(scan)
        gate = _gelu_tanh(self.w_out_gate(x_norm))
        delta = self.w_out(post_norm * gate)
        x = x + delta
        if dwa is not None:
            x = dwa.apply(x)
        return x, next_state if use_cache else None
