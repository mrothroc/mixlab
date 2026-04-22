#!/usr/bin/env python3
"""PyTorch numerical reference for mixlab plain+swiglu forward loss."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


BATCH = 1
SEQ_LEN = 8
MODEL_DIM = 32
HEADS = 2
VOCAB_SIZE = 64
MLP_MULT = 2.67
FFN_DIM = max(MODEL_DIM, round(MODEL_DIM * MLP_MULT))
EPS = 1e-5
ROPE_BASE = 10000.0


def new_weight(name, shape, weights, norm_scale=False):
    if norm_scale:
        value = torch.ones(shape, dtype=torch.float32)
    else:
        value = torch.empty(shape, dtype=torch.float32)
        value.normal_(0.0, 0.02)
    weights.append(
        {
            "name": name,
            "shape": list(shape),
            "values": value.reshape(-1).tolist(),
        }
    )
    return value


def rms_norm(x, scale):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS) * scale


def apply_rope(q, k, rope_dims=0):
    head_dim = q.shape[-1]
    if rope_dims <= 0 or rope_dims >= head_dim:
        rope_dims = head_dim
    dim_idx = torch.arange(0, rope_dims // 2, dtype=torch.float32)
    freqs = torch.exp(dim_idx * (-math.log(ROPE_BASE) * 2.0 / rope_dims))
    positions = torch.arange(0, SEQ_LEN, dtype=torch.float32)
    angles = positions[:, None] * freqs[None, :]
    cos_t = torch.cos(angles).reshape(1, 1, SEQ_LEN, rope_dims // 2)
    sin_t = torch.sin(angles).reshape(1, 1, SEQ_LEN, rope_dims // 2)

    def rotate(x):
        x_rot = x[..., :rope_dims]
        even = x_rot[..., 0::2]
        odd = x_rot[..., 1::2]
        rot_even = even * cos_t - odd * sin_t
        rot_odd = even * sin_t + odd * cos_t
        rotated = torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x_rot)
        return torch.cat((rotated, x[..., rope_dims:]), dim=-1)

    return rotate(q), rotate(k)


def main():
    torch.manual_seed(42)
    weights = []

    # This order matches arch.CollectWeightShapes for:
    # blocks=[plain(heads=2), swiglu], untied embeddings, no block scales.
    embed = new_weight("embed", (VOCAB_SIZE, MODEL_DIM), weights)
    head = new_weight("head", (MODEL_DIM, VOCAB_SIZE), weights)
    final_norm = new_weight("final_norm", (MODEL_DIM,), weights, norm_scale=True)

    attn_norm = new_weight("norm_scale", (MODEL_DIM,), weights, norm_scale=True)
    wq = new_weight("wq", (MODEL_DIM, MODEL_DIM), weights)
    wk = new_weight("wk", (MODEL_DIM, MODEL_DIM), weights)
    wv = new_weight("wv", (MODEL_DIM, MODEL_DIM), weights)
    wo = new_weight("wo", (MODEL_DIM, MODEL_DIM), weights)
    ff1 = new_weight("ff1", (MODEL_DIM, FFN_DIM), weights)
    ff2 = new_weight("ff2", (FFN_DIM, MODEL_DIM), weights)

    swiglu_norm = new_weight("ffn_norm_scale", (MODEL_DIM,), weights, norm_scale=True)
    w_gate = new_weight("w_gate", (MODEL_DIM, FFN_DIM), weights)
    w_up = new_weight("w_up", (MODEL_DIM, FFN_DIM), weights)
    w_down = new_weight("w_down", (FFN_DIM, MODEL_DIM), weights)

    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64)
    targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0], dtype=torch.int64)

    x = embed[tokens].reshape(BATCH * SEQ_LEN, MODEL_DIM)

    # Plain attention block as emitted by arch/blocks.go. Note that mixlab's
    # current plain block includes a SiLU FF tail after attention.
    x_norm = rms_norm(x, attn_norm)
    q = x_norm @ wq
    k = x_norm @ wk
    v = x_norm @ wv

    head_dim = MODEL_DIM // HEADS
    qh = q.reshape(BATCH, SEQ_LEN, HEADS, head_dim).transpose(1, 2)
    kh = k.reshape(BATCH, SEQ_LEN, HEADS, head_dim).transpose(1, 2)
    vh = v.reshape(BATCH, SEQ_LEN, HEADS, head_dim).transpose(1, 2)
    qh, kh = apply_rope(qh, kh, 0)

    scores = qh @ kh.transpose(-2, -1)
    scores = scores * (1.0 / math.sqrt(head_dim))
    causal = torch.triu(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.reshape(1, 1, SEQ_LEN, SEQ_LEN), -1e9)
    attn = torch.softmax(scores, dim=-1)
    ctx = attn @ vh
    flat = ctx.transpose(1, 2).reshape(BATCH * SEQ_LEN, MODEL_DIM)
    x = x + (flat @ wo)

    plain_ff = F.silu(x @ ff1) @ ff2
    x = x + plain_ff

    # SwiGLU block as emitted by arch/blocks.go. The checked-in IR uses
    # sigmoid(gate) * up, not SiLU(gate) * up.
    x_norm = rms_norm(x, swiglu_norm)
    gate = torch.sigmoid(x_norm @ w_gate)
    up = x_norm @ w_up
    x = x + ((gate * up) @ w_down)

    logits = (rms_norm(x, final_norm).reshape(BATCH, SEQ_LEN, MODEL_DIM) @ head).reshape(
        BATCH * SEQ_LEN, VOCAB_SIZE
    )
    loss = F.cross_entropy(logits, targets)

    payload = {
        "config": {
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "model_dim": MODEL_DIM,
            "heads": HEADS,
            "vocab_size": VOCAB_SIZE,
            "mlp_mult": MLP_MULT,
            "ffn_dim": FFN_DIM,
            "blocks": [{"type": "plain", "heads": HEADS}, {"type": "swiglu"}],
        },
        "tokens": tokens.reshape(-1).tolist(),
        "targets": targets.tolist(),
        "loss": float(loss.item()),
        "weights": weights,
    }
    out_path = Path(__file__).with_name("reference_weights.json")
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    print(f"loss={loss.item():.8f}")


if __name__ == "__main__":
    main()
