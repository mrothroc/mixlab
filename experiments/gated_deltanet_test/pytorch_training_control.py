#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dump_fla_reference_pure import Config as GDNConfig
from dump_fla_reference_pure import ReferenceBlock, rmsnorm

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def xavier_uniform_(tensor: torch.Tensor) -> None:
    fan_in, fan_out = tensor.shape[0], tensor.shape[1]
    limit = math.sqrt(6.0 / float(fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


class SwiGLUBlock(nn.Module):
    def __init__(self, model_dim: int, mult: float = 4.0 / 3.0) -> None:
        super().__init__()
        hidden = int(round(model_dim * model_dim * mult / model_dim))
        self.norm_scale = nn.Parameter(torch.ones(model_dim))
        self.w_gate = nn.Parameter(torch.empty(model_dim, hidden))
        self.w_up = nn.Parameter(torch.empty(model_dim, hidden))
        self.w_down = nn.Parameter(torch.empty(hidden, model_dim))
        xavier_uniform_(self.w_gate)
        xavier_uniform_(self.w_up)
        xavier_uniform_(self.w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = rmsnorm(x, self.norm_scale)
        gate = torch.sigmoid(x_norm @ self.w_gate)
        up = x_norm @ self.w_up
        return x + (gate * up) @ self.w_down


class ReferenceTrainModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.model_dim = int(cfg["model_dim"])
        self.vocab_size = int(cfg["vocab_size"])
        self.seq_len = int(cfg["seq_len"])
        self.embed = nn.Embedding(self.vocab_size, self.model_dim)
        self.head = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.blocks = nn.ModuleList()
        self.final_norm_scale = nn.Parameter(torch.ones(self.model_dim))
        xavier_uniform_(self.embed.weight)
        xavier_uniform_(self.head.weight)

        for block in cfg["blocks"]:
            block_type = str(block["type"]).strip().lower()
            if block_type == "gated_deltanet":
                kv_share = block.get("kv_share", True)
                gdn_cfg = GDNConfig(
                    model_dim=self.model_dim,
                    heads=int(block["heads"]),
                    d_k=int(block["d_k"]),
                    d_v=int(block.get("d_v", 2 * int(block["d_k"]))),
                    seq_len=self.seq_len,
                    layers=1,
                    kv_share=kv_share,
                    seed=0,
                )
                self.blocks.append(ReferenceBlock(gdn_cfg))
            elif block_type == "swiglu":
                self.blocks.append(SwiGLUBlock(self.model_dim))
            else:
                raise ValueError(f"unsupported block type {block['type']!r}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = rmsnorm(x, self.final_norm_scale)
        return x @ self.head.weight.t()


def load_data_shard(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32, count=HEADER_INTS)
    if len(raw) < HEADER_INTS:
        raise ValueError(f"{path}: short header")
    if int(raw[0]) != SHARD_MAGIC or int(raw[1]) != SHARD_VERSION:
        raise ValueError(f"{path}: bad header magic/version {raw[:2]}")
    n_tok = int(raw[2])
    toks = np.fromfile(path, dtype=np.uint16, offset=HEADER_INTS * 4)
    if len(toks) != n_tok:
        raise ValueError(f"{path}: token count mismatch got={len(toks)} want={n_tok}")
    return toks.astype(np.int64, copy=False)


def shuffle_chunks(tokens: np.ndarray, chunk_size: int, rng: random.Random) -> None:
    n = len(tokens) // chunk_size
    if n <= 1:
        return
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        if i == j:
            continue
        a0 = i * chunk_size
        b0 = j * chunk_size
        tmp = tokens[a0 : a0 + chunk_size].copy()
        tokens[a0 : a0 + chunk_size] = tokens[b0 : b0 + chunk_size]
        tokens[b0 : b0 + chunk_size] = tmp


class TokenStream:
    def __init__(self, pattern: str, seed: int, chunk_size: int) -> None:
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no shards matched {pattern!r}")
        self.rng = random.Random(seed)
        self.rng.shuffle(self.files)
        self.chunk_size = chunk_size
        self.idx = 0
        self.tok = load_data_shard(Path(self.files[0])).copy()
        shuffle_chunks(self.tok, self.chunk_size, self.rng)
        self.pos = self.rng.randint(0, len(self.tok) // 2) if len(self.tok) > 10000 else 0

    def advance(self) -> None:
        self.idx = (self.idx + 1) % len(self.files)
        self.tok = load_data_shard(Path(self.files[self.idx])).copy()
        shuffle_chunks(self.tok, self.chunk_size, self.rng)
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            if self.pos >= len(self.tok):
                self.advance()
            avail = min(n - filled, len(self.tok) - self.pos)
            out[filled : filled + avail] = self.tok[self.pos : self.pos + avail]
            self.pos += avail
            filled += avail
        return out


class Loader:
    def __init__(self, pattern: str, seed: int, chunk_size: int) -> None:
        self.stream = TokenStream(pattern, seed, chunk_size)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.stream.take(batch_tokens + 1)
        x = torch.from_numpy(tok[:-1].copy()).view(batch_tokens // seq_len, seq_len)
        y = torch.from_numpy(tok[1:].copy()).view(batch_tokens // seq_len, seq_len)
        return x, y


def replace_first_train_with_val(pattern: str) -> str:
    return pattern.replace("train", "val", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/Users/mrothroc/IdeaProjects/mixlab/experiments/gated_deltanet_test/gated_deltanet_compare.json")
    parser.add_argument("--train", default="/Users/mrothroc/IdeaProjects/mixlab/data/example/train_*.bin")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    seq_len = int(cfg["seq_len"])
    batch_tokens = int(cfg["training"]["batch_tokens"])
    if batch_tokens % seq_len != 0:
        raise ValueError(f"batch_tokens={batch_tokens} must be divisible by seq_len={seq_len}")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ReferenceTrainModel(cfg).to(device=device, dtype=torch.float32)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    loader = Loader(args.train, args.seed, seq_len)
    val_loader = Loader(replace_first_train_with_val(args.train), args.seed, seq_len)

    first_loss = None
    last_loss = None

    for step in range(args.steps):
        model.train()
        x, y = loader.next_batch(batch_tokens, seq_len)
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step={step} train_loss={last_loss:.6f}")

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(args.val_batches):
            x, y = val_loader.next_batch(batch_tokens, seq_len)
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            val_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            val_losses.append(float(val_loss.item()))
    final_val = sum(val_losses) / len(val_losses)

    print("=== PYTORCH CONTROL ===")
    print(f"device={device}")
    print(f"first_train_loss={first_loss:.6f}")
    print(f"final_train_loss={last_loss:.6f}")
    print(f"final_val_loss={final_val:.6f}")


if __name__ == "__main__":
    main()
