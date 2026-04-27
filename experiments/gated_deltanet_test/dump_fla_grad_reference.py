#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dump_fla_reference_pure import Config, ReferenceStack, conv_weight_to_mixlab


def save_grad_fixture(path: str, cfg: Config) -> None:
    if cfg.layers != 1:
        raise ValueError("gradient fixture currently supports exactly one layer")

    torch.manual_seed(cfg.seed)
    model = ReferenceStack(cfg).train()
    x = torch.randn(1, cfg.seq_len, cfg.model_dim, dtype=torch.float32)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()

    block = model.blocks[0]
    arrays: dict[str, np.ndarray] = {
        "model_dim": np.array([cfg.model_dim], dtype=np.int64),
        "seq_len": np.array([cfg.seq_len], dtype=np.int64),
        "heads": np.array([cfg.heads], dtype=np.int64),
        "d_k": np.array([cfg.d_k], dtype=np.int64),
        "d_v": np.array([cfg.d_v], dtype=np.int64),
        "layers": np.array([cfg.layers], dtype=np.int64),
        "kv_share": np.array([1 if cfg.kv_share else 0], dtype=np.int64),
        "loss": np.array([loss.detach().cpu().item()], dtype=np.float32),
        "input": x.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "expected_x_hidden": y.detach().cpu().numpy().astype(np.float32),
        "final_norm_scale": model.final_norm_scale.detach().cpu().numpy().astype(np.float32),
        "final_norm_scale_grad": model.final_norm_scale.grad.detach().cpu().numpy().astype(np.float32),
        "norm_scale": block.norm_scale.detach().cpu().numpy().astype(np.float32),
        "norm_scale_grad": block.norm_scale.grad.detach().cpu().numpy().astype(np.float32),
        "wq": block.wq.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "wq_grad": block.wq.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "q_conv": conv_weight_to_mixlab(block.q_conv_w).astype(np.float32),
        "q_conv_grad": conv_weight_to_mixlab(block.q_conv_w.grad).astype(np.float32),
        "k_conv": conv_weight_to_mixlab(block.k_conv_w).astype(np.float32),
        "k_conv_grad": conv_weight_to_mixlab(block.k_conv_w.grad).astype(np.float32),
        "v_conv": conv_weight_to_mixlab(block.v_conv_w).astype(np.float32),
        "v_conv_grad": conv_weight_to_mixlab(block.v_conv_w.grad).astype(np.float32),
        "w_a": block.a_proj.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "w_a_grad": block.a_proj.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "A_log": block.A_log.detach().cpu().numpy().astype(np.float32),
        "A_log_grad": block.A_log.grad.detach().cpu().numpy().astype(np.float32),
        "dt_bias": block.dt_bias.detach().cpu().numpy().astype(np.float32),
        "dt_bias_grad": block.dt_bias.grad.detach().cpu().numpy().astype(np.float32),
        "w_beta": block.b_proj.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "w_beta_grad": block.b_proj.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "w_out_gate": block.g_proj.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "w_out_gate_grad": block.g_proj.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "o_norm_scale": block.o_norm_scale.detach().cpu().numpy().astype(np.float32),
        "o_norm_scale_grad": block.o_norm_scale.grad.detach().cpu().numpy().astype(np.float32),
        "wo": block.wo.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        "wo_grad": block.wo.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32),
    }
    if cfg.kv_share:
        arrays["w_kv"] = block.w_kv.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays["w_kv_grad"] = block.w_kv.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32)
    else:
        arrays["wk"] = block.w_k.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays["wk_grad"] = block.w_k.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays["wv"] = block.w_v.weight.detach().t().contiguous().cpu().numpy().astype(np.float32)
        arrays["wv_grad"] = block.w_v.weight.grad.detach().t().contiguous().cpu().numpy().astype(np.float32)

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
        layers=1,
        kv_share=not args.no_kv_share,
        seed=args.seed,
    )
    save_grad_fixture(args.output, cfg)


if __name__ == "__main__":
    main()
