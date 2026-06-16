#!/usr/bin/env python3
"""Generate static HGRN2/mLSTM full-block reference fixtures.

This script intentionally uses only the Python standard library. It is not a
runtime test dependency; the checked-in JSON fixture is consumed by Go tests.
The math mirrors Mixlab's public block definitions from independent scalar
loops so the fixture catches full-block wiring mistakes, not just scan kernels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "train" / "testdata" / "hgrn2_mlstm_full_block_reference.json"
EPS = 1e-5


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def rms_norm_rows(x: list[float], rows: int, cols: int, scale: list[float]) -> list[float]:
    out = []
    for r in range(rows):
        base = r * cols
        mean_sq = sum(x[base + c] * x[base + c] for c in range(cols)) / cols
        inv = 1.0 / math.sqrt(mean_sq + EPS)
        for c in range(cols):
            out.append(x[base + c] * inv * scale[c])
    return out


def matmul(a: list[float], b: list[float], rows: int, inner: int, cols: int) -> list[float]:
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            acc = 0.0
            for k in range(inner):
                acc += a[r * inner + k] * b[k * cols + c]
            out[r * cols + c] = acc
    return out


def add_rows(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def hgrn2_scan(
    q: list[float],
    k: list[float],
    v: list[float],
    gate: list[float],
    batch: int,
    seq: int,
    heads: int,
    d_state: int,
    d_value: int,
) -> list[float]:
    state = [0.0] * (batch * heads * d_state * d_value)
    out = [0.0] * (batch * seq * heads * d_value)
    for b in range(batch):
        for t in range(seq):
            for h in range(heads):
                for ds in range(d_state):
                    qv = q[((b * seq + t) * heads + h) * d_state + ds]
                    kv = k[((b * seq + t) * heads + h) * d_state + ds]
                    gv = gate[((b * seq + t) * heads + h) * d_state + ds]
                    for dv in range(d_value):
                        si = ((b * heads + h) * d_state + ds) * d_value + dv
                        vv = v[((b * seq + t) * heads + h) * d_value + dv]
                        state[si] = gv * state[si] + kv * vv
                        out[((b * seq + t) * heads + h) * d_value + dv] += qv * state[si]
    return out


def mlstm_scan(
    q: list[float],
    k: list[float],
    v: list[float],
    input_gate: list[float],
    forget_gate: list[float],
    batch: int,
    seq: int,
    heads: int,
    d_key: int,
    d_value: int,
) -> list[float]:
    c_state = [0.0] * (batch * heads * d_key * d_value)
    n_state = [0.0] * (batch * heads * d_key)
    m_state = [0.0] * (batch * heads)
    out = [0.0] * (batch * seq * heads * d_value)
    q_scale = 1.0 / math.sqrt(d_key)
    for b in range(batch):
        for h in range(heads):
            m_prev = 0.0
            for t in range(seq):
                i_preact = input_gate[(b * seq + t) * heads + h]
                f_preact = forget_gate[(b * seq + t) * heads + h]
                m_t = max(f_preact + m_prev, i_preact)
                f = math.exp(f_preact + m_prev - m_t)
                i = math.exp(i_preact - m_t)
                m_prev = m_t
                m_state[b * heads + h] = m_t

                for dk in range(d_key):
                    k_val = k[((b * seq + t) * heads + h) * d_key + dk]
                    n_idx = (b * heads + h) * d_key + dk
                    n_state[n_idx] = f * n_state[n_idx] + i * k_val
                    for dv in range(d_value):
                        c_idx = ((b * heads + h) * d_key + dk) * d_value + dv
                        v_val = v[((b * seq + t) * heads + h) * d_value + dv]
                        c_state[c_idx] = f * c_state[c_idx] + i * k_val * v_val

                denom = 0.0
                for dk in range(d_key):
                    q_val = q[((b * seq + t) * heads + h) * d_key + dk] * q_scale
                    denom += q_val * n_state[(b * heads + h) * d_key + dk]
                denom = max(abs(denom), 1.0)

                for dv in range(d_value):
                    acc = 0.0
                    for dk in range(d_key):
                        q_val = q[((b * seq + t) * heads + h) * d_key + dk] * q_scale
                        c_idx = ((b * heads + h) * d_key + dk) * d_value + dv
                        acc += q_val * c_state[c_idx]
                    out[((b * seq + t) * heads + h) * d_value + dv] = acc / denom
    return out


def embed_weights(vocab: int, dim: int, x: list[float]) -> list[float]:
    out = [0.0] * (vocab * dim)
    out[: len(x)] = x
    return out


def hgrn2_fixture() -> dict:
    batch, seq, dim, vocab, heads, d_state = 1, 3, 4, 8, 2, 2
    head_dim = dim // heads
    tokens = [0, 1, 2]
    targets = [1, 2, 3]
    x = [
        0.20, -0.10, 0.35, 0.45,
        -0.30, 0.25, 0.10, -0.40,
        0.55, 0.15, -0.20, 0.30,
    ]
    norm = [1.10, 0.90, 1.05, 0.95]
    w_v = [
        0.20, -0.10, 0.05, 0.30,
        -0.25, 0.40, 0.15, -0.05,
        0.10, 0.35, -0.20, 0.25,
        0.30, -0.15, 0.45, 0.10,
    ]
    w_q = [
        0.30, -0.20, 0.10, 0.25,
        0.15, 0.35, -0.30, 0.05,
        -0.25, 0.10, 0.40, -0.15,
        0.20, -0.05, 0.30, 0.45,
    ]
    w_f = [
        -0.10, 0.25, 0.30, -0.20,
        0.35, -0.15, 0.05, 0.40,
        0.20, 0.10, -0.25, 0.15,
        -0.30, 0.45, 0.20, -0.05,
    ]
    out_norm = [1.20, 0.85]
    w_o = [
        0.25, -0.10, 0.05, 0.30,
        -0.20, 0.15, 0.35, -0.05,
        0.40, 0.10, -0.15, 0.20,
        0.05, -0.30, 0.25, 0.10,
    ]

    x_norm = rms_norm_rows(x, batch * seq, dim, norm)
    v_raw = matmul(x_norm, w_v, batch * seq, dim, dim)
    v = [silu(v) for v in v_raw]
    q = [sigmoid(v) for v in matmul(x_norm, w_q, batch * seq, dim, heads * d_state)]
    gate = [sigmoid(v) for v in matmul(x_norm, w_f, batch * seq, dim, heads * d_state)]
    k = [1.0 - v for v in gate]
    scan = hgrn2_scan(q, k, v, gate, batch, seq, heads, d_state, head_dim)
    scan_norm = rms_norm_rows(scan, batch * seq * heads, head_dim, out_norm)
    projected = matmul(scan_norm, w_o, batch * seq, dim, dim)
    expected = add_rows(x, projected)

    return {
        "name": "hgrn2_full_block",
        "config": {
            "name": "hgrn2_reference_fixture",
            "model_dim": dim,
            "vocab_size": vocab,
            "seq_len": seq,
            "blocks": [{"type": "hgrn2", "heads": heads, "d_state": d_state}],
            "training": {
                "steps": 1,
                "batch_tokens": batch * seq,
                "seed": 7,
                "objective": "mntp",
                "mlm_mask_token_id": vocab - 1,
                "data2vec": {"loss_weight": 1.0, "top_k_layers": 1},
            },
        },
        "output": "data2vec_layer_00_hidden_flat",
        "batch": batch,
        "seq_len": seq,
        "model_dim": dim,
        "tokens": tokens,
        "targets": targets,
        "loss_mask": [1.0, 1.0, 1.0],
        "weights": [
            {"name": "embed", "shape": [vocab, dim], "values": embed_weights(vocab, dim, x)},
            {"name": "head", "shape": [dim, vocab], "values": [0.0] * (dim * vocab)},
            {"name": "final_norm", "shape": [dim], "values": [1.0] * dim},
            {"name": "norm_scale", "shape": [dim], "values": norm},
            {"name": "w_v", "shape": [dim, dim], "values": w_v},
            {"name": "w_q", "shape": [dim, heads * d_state], "values": w_q},
            {"name": "w_f", "shape": [dim, heads * d_state], "values": w_f},
            {"name": "o_norm_scale", "shape": [head_dim], "values": out_norm},
            {"name": "wo", "shape": [dim, dim], "values": w_o},
        ],
        "expected_hidden": expected,
    }


def mlstm_fixture() -> dict:
    batch, seq, dim, vocab, heads, d_key, d_value = 1, 3, 4, 8, 2, 2, 2
    val_dim = heads * d_value
    tokens = [0, 1, 2]
    targets = [1, 2, 3]
    x = [
        0.12, -0.33, 0.27, 0.41,
        0.50, 0.08, -0.22, 0.15,
        -0.31, 0.44, 0.19, -0.28,
    ]
    norm = [0.95, 1.05, 0.90, 1.10]
    wq = [
        0.18, -0.22, 0.31, 0.07,
        -0.12, 0.27, -0.05, 0.34,
        0.29, 0.11, -0.24, 0.16,
        -0.08, 0.38, 0.21, -0.19,
    ]
    wk = [
        -0.15, 0.26, 0.09, -0.32,
        0.33, -0.07, 0.20, 0.12,
        0.04, -0.28, 0.36, 0.18,
        0.22, 0.14, -0.11, 0.30,
    ]
    wv = [
        0.24, -0.09, 0.17, 0.28,
        -0.31, 0.19, 0.06, -0.22,
        0.13, 0.35, -0.27, 0.08,
        0.05, -0.18, 0.32, 0.21,
    ]
    w_i = [
        0.20, -0.10,
        -0.15, 0.25,
        0.30, 0.05,
        -0.22, 0.18,
    ]
    b_i = [0.05, -0.10]
    w_f = [
        -0.12, 0.28,
        0.21, -0.16,
        0.07, 0.33,
        0.25, -0.05,
    ]
    b_f = [0.20, -0.15]
    out_norm = [1.15, 0.80]
    w_out_gate = [
        0.16, -0.21, 0.27, 0.11,
        0.05, 0.32, -0.18, 0.24,
        -0.29, 0.14, 0.08, -0.35,
        0.22, -0.06, 0.31, 0.19,
    ]
    w_o = [
        0.23, -0.17, 0.09, 0.31,
        -0.14, 0.26, 0.33, -0.04,
        0.37, 0.12, -0.21, 0.15,
        0.06, -0.34, 0.28, 0.20,
    ]

    x_norm = rms_norm_rows(x, batch * seq, dim, norm)
    q = matmul(x_norm, wq, batch * seq, dim, heads * d_key)
    k = matmul(x_norm, wk, batch * seq, dim, heads * d_key)
    v = matmul(x_norm, wv, batch * seq, dim, val_dim)
    ig = add_rows(matmul(x_norm, w_i, batch * seq, dim, heads), b_i * (batch * seq))
    fg = add_rows(matmul(x_norm, w_f, batch * seq, dim, heads), b_f * (batch * seq))
    scan = mlstm_scan(q, k, v, ig, fg, batch, seq, heads, d_key, d_value)
    scan_norm = rms_norm_rows(scan, batch * seq * heads, d_value, out_norm)
    out_gate = [sigmoid(v) for v in matmul(x_norm, w_out_gate, batch * seq, dim, val_dim)]
    gated = [a * b for a, b in zip(scan_norm, out_gate)]
    projected = matmul(gated, w_o, batch * seq, val_dim, dim)
    expected = add_rows(x, projected)

    return {
        "name": "mlstm_full_block",
        "config": {
            "name": "mlstm_reference_fixture",
            "model_dim": dim,
            "vocab_size": vocab,
            "seq_len": seq,
            "blocks": [{"type": "mlstm", "heads": heads, "d_k": d_key, "d_v": d_value}],
            "training": {
                "steps": 1,
                "batch_tokens": batch * seq,
                "seed": 11,
                "objective": "mntp",
                "mlm_mask_token_id": vocab - 1,
                "data2vec": {"loss_weight": 1.0, "top_k_layers": 1},
            },
        },
        "output": "data2vec_layer_00_hidden_flat",
        "batch": batch,
        "seq_len": seq,
        "model_dim": dim,
        "tokens": tokens,
        "targets": targets,
        "loss_mask": [1.0, 1.0, 1.0],
        "weights": [
            {"name": "embed", "shape": [vocab, dim], "values": embed_weights(vocab, dim, x)},
            {"name": "head", "shape": [dim, vocab], "values": [0.0] * (dim * vocab)},
            {"name": "final_norm", "shape": [dim], "values": [1.0] * dim},
            {"name": "norm_scale", "shape": [dim], "values": norm},
            {"name": "wq", "shape": [dim, heads * d_key], "values": wq},
            {"name": "wk", "shape": [dim, heads * d_key], "values": wk},
            {"name": "wv", "shape": [dim, val_dim], "values": wv},
            {"name": "w_i", "shape": [dim, heads], "values": w_i},
            {"name": "b_i", "shape": [heads], "values": b_i},
            {"name": "w_f", "shape": [dim, heads], "values": w_f},
            {"name": "b_f", "shape": [heads], "values": b_f},
            {"name": "o_norm_scale", "shape": [d_value], "values": out_norm},
            {"name": "w_out_gate", "shape": [dim, val_dim], "values": w_out_gate},
            {"name": "wo", "shape": [val_dim, dim], "values": w_o},
        ],
        "expected_hidden": expected,
    }


def main() -> None:
    payload = {
        "version": 1,
        "description": "Static full-block HGRN2 and mLSTM reference fixtures generated from independent scalar reference loops.",
        "fixtures": [hgrn2_fixture(), mlstm_fixture()],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
