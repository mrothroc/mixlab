#!/usr/bin/env python3
"""
Dump a K_KVShare_Wider-style GatedDeltaNet parity fixture for mixlab using a
pure-PyTorch reference recurrence (no triton). Mathematically equivalent to
fla.ops.gated_delta_rule.chunk_gated_delta_rule but slow / sequential.

Recurrence (per head, per token):
    decay_t  in (0, 1] from -exp(A_log) * softplus(a_proj(x) + dt_bias) -> exp(.)
    beta_t   in (0, 1) from sigmoid(b_proj(x))
    pred_t   = k_t^T (decay_t * S_{t-1})           # FLA delta rule prediction
    err_t    = v_t - pred_t                        # error-correcting term
    S_t      = decay_t * S_{t-1} + beta_t * outer(k_t, err_t)
    o_t      = q_t S_t

Note the err_t subtraction — this is the *delta* rule. Without it, this collapses
to plain gated linear attention.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight


def causal_short_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Causal depthwise 1D conv (kernel_size=K) followed by SiLU.
    x: [B, T, C]
    weight: [C, K]  (depthwise — one filter per channel)
    Returns: [B, T, C]
    """
    B, T, C = x.shape
    K = weight.shape[1]
    assert weight.shape[0] == C
    pad = F.pad(x, (0, 0, K - 1, 0))  # left-pad time by K-1
    out = torch.zeros_like(x)
    for k in range(K):
        out = out + pad[:, k : k + T, :] * weight[:, K - 1 - k].view(1, 1, C)
    return F.silu(out)


def gated_delta_rule_recurrence(
    q: torch.Tensor,  # [B, T, H, Dk] (already L2-normalized)
    k: torch.Tensor,  # [B, T, H, Dk] (already L2-normalized)
    v: torch.Tensor,  # [B, T, H, Dv]
    beta: torch.Tensor,  # [B, T, H]
    decay: torch.Tensor,  # [B, T, H]   (per-token gate in (0,1])
) -> torch.Tensor:
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, dtype=q.dtype, device=q.device)
    out = torch.zeros(B, T, H, Dv, dtype=q.dtype, device=q.device)
    for t in range(T):
        d = decay[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        S = d * S
        kt = k[:, t]  # [B, H, Dk]
        vt = v[:, t]  # [B, H, Dv]
        # pred_t = k_t^T S
        pred = torch.einsum("bhd,bhde->bhe", kt, S)  # [B, H, Dv]
        err = vt - pred  # delta-rule error term
        # outer(k_t, err)
        update = torch.einsum("bhd,bhe->bhde", kt, err)  # [B, H, Dk, Dv]
        b = beta[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        S = S + b * update
        # o_t = q_t S
        qt = q[:, t]  # [B, H, Dk]
        o = torch.einsum("bhd,bhde->bhe", qt, S)
        out[:, t] = o
    return out


@dataclass
class Config:
    model_dim: int
    heads: int
    d_k: int
    d_v: int
    seq_len: int
    layers: int
    kv_share: bool
    seed: int


class ReferenceBlock(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        key_dim = cfg.heads * cfg.d_k
        val_dim = cfg.heads * cfg.d_v
        self.norm_scale = torch.nn.Parameter(torch.ones(cfg.model_dim))
        self.wq = torch.nn.Linear(cfg.model_dim, key_dim, bias=False)
        if cfg.kv_share:
            self.w_kv = torch.nn.Linear(cfg.model_dim, val_dim, bias=False)
        else:
            self.w_k = torch.nn.Linear(cfg.model_dim, key_dim, bias=False)
            self.w_v = torch.nn.Linear(cfg.model_dim, val_dim, bias=False)
        # depthwise conv weights [C, K=4]
        self.q_conv_w = torch.nn.Parameter(torch.empty(key_dim, 4))
        self.k_conv_w = torch.nn.Parameter(torch.empty(key_dim, 4))
        self.v_conv_w = torch.nn.Parameter(torch.empty(val_dim, 4))
        torch.nn.init.kaiming_uniform_(self.q_conv_w.view(key_dim, 1, 4), a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.k_conv_w.view(key_dim, 1, 4), a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.v_conv_w.view(val_dim, 1, 4), a=math.sqrt(5))
        self.a_proj = torch.nn.Linear(cfg.model_dim, cfg.heads, bias=False)
        self.A_log = torch.nn.Parameter(torch.empty(cfg.heads))
        self.dt_bias = torch.nn.Parameter(torch.empty(cfg.heads))
        self.b_proj = torch.nn.Linear(cfg.model_dim, cfg.heads, bias=False)
        self.g_proj = torch.nn.Linear(cfg.model_dim, val_dim, bias=False)
        self.o_norm_scale = torch.nn.Parameter(torch.ones(cfg.d_v))
        self.wo = torch.nn.Linear(val_dim, cfg.model_dim, bias=False)

        torch.nn.init.uniform_(self.A_log, np.log(1.0), np.log(16.0))
        torch.nn.init.uniform_(self.dt_bias, -4.0, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch_size = x.shape[0]
        x_norm = rmsnorm(x, self.norm_scale)

        q = self.wq(x_norm)
        q = causal_short_conv1d(q, self.q_conv_w)
        q = F.normalize(q.view(batch_size, cfg.seq_len, cfg.heads, cfg.d_k), dim=-1)
        q = q * (cfg.d_k ** -0.5)  # FLA kernel applies 1/sqrt(d_k) scale; mixlab does it pre-recurrence

        if cfg.kv_share:
            kv = self.w_kv(x_norm)
            v_full = kv.view(batch_size, cfg.seq_len, cfg.heads, cfg.d_v)
            k = v_full[..., : cfg.d_k].reshape(batch_size, cfg.seq_len, cfg.heads * cfg.d_k)
            v = v_full.reshape(batch_size, cfg.seq_len, cfg.heads * cfg.d_v)
        else:
            k = self.w_k(x_norm)
            v = self.w_v(x_norm)
        k = causal_short_conv1d(k, self.k_conv_w).view(batch_size, cfg.seq_len, cfg.heads, cfg.d_k)
        k = F.normalize(k, dim=-1)
        v = causal_short_conv1d(v, self.v_conv_w).view(batch_size, cfg.seq_len, cfg.heads, cfg.d_v)

        beta = torch.sigmoid(self.b_proj(x_norm)).view(batch_size, cfg.seq_len, cfg.heads)
        dt = F.softplus(self.a_proj(x_norm) + self.dt_bias)
        decay = torch.exp(-torch.exp(self.A_log).view(1, 1, cfg.heads) * dt)  # in (0, 1]
        o = gated_delta_rule_recurrence(q, k, v, beta, decay)

        gate = self.g_proj(x_norm).view(batch_size, cfg.seq_len, cfg.heads, cfg.d_v)
        o = rmsnorm(o, self.o_norm_scale) * F.silu(gate)
        o = self.wo(o.reshape(batch_size, cfg.seq_len, cfg.heads * cfg.d_v))
        return x + o


class ReferenceStack(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = torch.nn.ModuleList([ReferenceBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm_scale = torch.nn.Parameter(torch.ones(cfg.model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return rmsnorm(x, self.final_norm_scale)


def conv_weight_to_mixlab(weight: torch.Tensor) -> np.ndarray:
    # FLA / mixlab convention: weight shape [C, K], save as [K, C]? Match the
    # original dump_fla_reference.py format. The original does:
    #   if rank-3 [C, 1, K]: w[:, 0, :].transpose(0, 1).contiguous() -> [K, C]
    #   if rank-2 [C, K]:    transpose(0, 1).contiguous()           -> [K, C]
    w = weight.detach().float().cpu()
    if w.ndim == 2:
        return w.transpose(0, 1).contiguous().numpy()
    raise ValueError(f"unsupported conv weight rank {w.ndim}")


def save_fixture(path: str, cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    model = ReferenceStack(cfg).eval()
    x = torch.randn(1, cfg.seq_len, cfg.model_dim, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    arrays: dict[str, np.ndarray] = {
        "model_dim": np.array([cfg.model_dim], dtype=np.int64),
        "seq_len": np.array([cfg.seq_len], dtype=np.int64),
        "heads": np.array([cfg.heads], dtype=np.int64),
        "d_k": np.array([cfg.d_k], dtype=np.int64),
        "d_v": np.array([cfg.d_v], dtype=np.int64),
        "layers": np.array([cfg.layers], dtype=np.int64),
        "kv_share": np.array([1 if cfg.kv_share else 0], dtype=np.int64),
        "input": x.squeeze(0).cpu().numpy().astype(np.float32),
        "expected_x_hidden": y.cpu().numpy().astype(np.float32),
        "final_norm_scale": model.final_norm_scale.detach().cpu().numpy().astype(np.float32),
    }
    for blockIdx, block in enumerate(model.blocks):
        prefix = f"block_{blockIdx}_"
        arrays[prefix+"norm_scale"] = block.norm_scale.detach().cpu().numpy().astype(np.float32)
        arrays[prefix+"wq"] = block.wq.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays[prefix+"q_conv"] = conv_weight_to_mixlab(block.q_conv_w).astype(np.float32)
        arrays[prefix+"k_conv"] = conv_weight_to_mixlab(block.k_conv_w).astype(np.float32)
        arrays[prefix+"v_conv"] = conv_weight_to_mixlab(block.v_conv_w).astype(np.float32)
        arrays[prefix+"w_a"] = block.a_proj.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays[prefix+"A_log"] = block.A_log.detach().cpu().numpy().astype(np.float32)
        arrays[prefix+"dt_bias"] = block.dt_bias.detach().cpu().numpy().astype(np.float32)
        arrays[prefix+"w_beta"] = block.b_proj.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays[prefix+"w_out_gate"] = block.g_proj.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays[prefix+"o_norm_scale"] = block.o_norm_scale.detach().cpu().numpy().astype(np.float32)
        arrays[prefix+"wo"] = block.wo.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        if cfg.kv_share:
            arrays[prefix+"w_kv"] = block.w_kv.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        else:
            arrays[prefix+"wk"] = block.w_k.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
            arrays[prefix+"wv"] = block.w_v.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)

        if cfg.layers == 1 and blockIdx == 0:
            for name in (
                "norm_scale", "wq", "q_conv", "k_conv", "v_conv", "w_a", "A_log",
                "dt_bias", "w_beta", "w_out_gate", "o_norm_scale", "wo",
            ):
                arrays[name] = arrays[prefix+name]
            if cfg.kv_share:
                arrays["w_kv"] = arrays[prefix+"w_kv"]
            else:
                arrays["wk"] = arrays[prefix+"wk"]
                arrays["wv"] = arrays[prefix+"wv"]

    np.savez(path, **arrays)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-k", type=int, default=8)
    parser.add_argument("--d-v", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-kv-share", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        model_dim=args.model_dim,
        heads=args.heads,
        d_k=args.d_k,
        d_v=args.d_v,
        seq_len=args.seq_len,
        layers=args.layers,
        kv_share=not args.no_kv_share,
        seed=args.seed,
    )
    save_fixture(args.output, cfg)


if __name__ == "__main__":
    main()
