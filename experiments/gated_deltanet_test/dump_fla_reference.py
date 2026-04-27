#!/usr/bin/env python3
"""
Dump a K_KVShare_Wider-style GatedDeltaNet parity fixture for mixlab.

Requires:
  pip install torch numpy flash-linear-attention
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fla.modules.convolution import ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight


def conv_weight_to_mixlab(weight: torch.Tensor) -> np.ndarray:
    w = weight.detach().float().cpu()
    if w.ndim == 3:
        if w.shape[1] != 1:
            raise ValueError(f"expected depthwise conv weight second dim 1, got {tuple(w.shape)}")
        return w[:, 0, :].transpose(0, 1).contiguous().numpy()
    if w.ndim == 2:
        return w.transpose(0, 1).contiguous().numpy()
    raise ValueError(f"unsupported conv weight rank {w.ndim}")


@dataclass
class Config:
    model_dim: int
    heads: int
    d_k: int
    d_v: int
    seq_len: int
    kv_share: bool
    seed: int


class ReferenceBlock(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        key_dim = cfg.heads * cfg.d_k
        val_dim = cfg.heads * cfg.d_v
        self.norm_scale = torch.nn.Parameter(torch.ones(cfg.model_dim))
        self.final_norm_scale = torch.nn.Parameter(torch.ones(cfg.model_dim))
        self.wq = torch.nn.Linear(cfg.model_dim, key_dim, bias=False)
        if cfg.kv_share:
            self.w_kv = torch.nn.Linear(cfg.model_dim, val_dim, bias=False)
            self.wk = None
            self.wv = None
        else:
            self.w_k = torch.nn.Linear(cfg.model_dim, key_dim, bias=False)
            self.w_v = torch.nn.Linear(cfg.model_dim, val_dim, bias=False)
            self.w_kv = None
        self.q_conv = ShortConvolution(key_dim, kernel_size=4, activation="silu")
        self.k_conv = ShortConvolution(key_dim, kernel_size=4, activation="silu")
        self.v_conv = ShortConvolution(val_dim, kernel_size=4, activation="silu")
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
        x_norm = rmsnorm(x, self.norm_scale)

        q = self.wq(x_norm)
        q = self.q_conv(q)
        q = F.normalize(q.view(1, cfg.seq_len, cfg.heads, cfg.d_k), dim=-1)

        if cfg.kv_share:
            kv = self.w_kv(x_norm).view(1, cfg.seq_len, cfg.heads, cfg.d_v)
            k = kv[..., : cfg.d_k].reshape(1, cfg.seq_len, cfg.heads * cfg.d_k)
            v = kv.reshape(1, cfg.seq_len, cfg.heads * cfg.d_v)
        else:
            k = self.w_k(x_norm)
            v = self.w_v(x_norm)
        k = self.k_conv(k).view(1, cfg.seq_len, cfg.heads, cfg.d_k)
        k = F.normalize(k, dim=-1)
        v = self.v_conv(v).view(1, cfg.seq_len, cfg.heads, cfg.d_v)

        beta = torch.sigmoid(self.b_proj(x_norm)).view(1, cfg.seq_len, cfg.heads)
        dt = F.softplus(self.a_proj(x_norm) + self.dt_bias)
        decay = -torch.exp(self.A_log).view(1, 1, cfg.heads) * dt
        o = chunk_gated_delta_rule(q, k, v, beta, decay)

        gate = self.g_proj(x_norm).view(1, cfg.seq_len, cfg.heads, cfg.d_v)
        o = rmsnorm(o, self.o_norm_scale) * F.silu(gate)
        o = self.wo(o.reshape(1, cfg.seq_len, cfg.heads * cfg.d_v))
        x = x + o
        return rmsnorm(x, self.final_norm_scale)


def save_fixture(path: str, cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    model = ReferenceBlock(cfg).eval()
    x = torch.randn(1, cfg.seq_len, cfg.model_dim, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    arrays: dict[str, np.ndarray] = {
        "model_dim": np.array([cfg.model_dim], dtype=np.int64),
        "seq_len": np.array([cfg.seq_len], dtype=np.int64),
        "heads": np.array([cfg.heads], dtype=np.int64),
        "d_k": np.array([cfg.d_k], dtype=np.int64),
        "d_v": np.array([cfg.d_v], dtype=np.int64),
        "kv_share": np.array([1 if cfg.kv_share else 0], dtype=np.int64),
        "input": x.squeeze(0).cpu().numpy().astype(np.float32),
        "expected_x_hidden": y.cpu().numpy().astype(np.float32),
        "final_norm_scale": model.final_norm_scale.detach().cpu().numpy().astype(np.float32),
        "norm_scale": model.norm_scale.detach().cpu().numpy().astype(np.float32),
        "wq": model.wq.weight.detach().t().cpu().numpy().astype(np.float32),
        "q_conv": conv_weight_to_mixlab(model.q_conv.weight).astype(np.float32),
        "k_conv": conv_weight_to_mixlab(model.k_conv.weight).astype(np.float32),
        "v_conv": conv_weight_to_mixlab(model.v_conv.weight).astype(np.float32),
        "w_a": model.a_proj.weight.detach().t().cpu().numpy().astype(np.float32),
        "A_log": model.A_log.detach().cpu().numpy().astype(np.float32),
        "dt_bias": model.dt_bias.detach().cpu().numpy().astype(np.float32),
        "w_beta": model.b_proj.weight.detach().t().cpu().numpy().astype(np.float32),
        "w_out_gate": model.g_proj.weight.detach().t().cpu().numpy().astype(np.float32),
        "o_norm_scale": model.o_norm_scale.detach().cpu().numpy().astype(np.float32),
        "wo": model.wo.weight.detach().t().cpu().numpy().astype(np.float32),
    }
    if cfg.kv_share:
        arrays["w_kv"] = model.w_kv.weight.detach().t().cpu().numpy().astype(np.float32)
    else:
        arrays["wk"] = model.w_k.weight.detach().t().cpu().numpy().astype(np.float32)
        arrays["wv"] = model.w_v.weight.detach().t().cpu().numpy().astype(np.float32)

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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-kv-share", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        model_dim=args.model_dim,
        heads=args.heads,
        d_k=args.d_k,
        d_v=args.d_v,
        seq_len=args.seq_len,
        kv_share=not args.no_kv_share,
        seed=args.seed,
    )
    save_fixture(args.output, cfg)


if __name__ == "__main__":
    main()
