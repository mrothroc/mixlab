#!/usr/bin/env python3
"""CLI native-vs-Hugging Face parity checker for Mixlab exports.

The Go side computes native MLX loss and writes a bounded sample of native
logits. This script loads the exported Hugging Face directory with the actual
shipped custom-code model, computes HF loss on the same flat token stream, and
compares sample logits.
"""

import argparse
import array
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


TOKENS_MAGIC = 0x504B544D
LOGITS_MAGIC = 0x474C504D
FILE_VERSION = 1


def read_tokens(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != TOKENS_MAGIC or version != FILE_VERSION:
            raise ValueError(f"{path}: unsupported token file header magic={magic:#x} version={version}")
        (count,) = struct.unpack("<Q", f.read(8))
        payload = f.read()
    if len(payload) != count * 2:
        raise ValueError(f"{path}: token payload has {len(payload)} bytes, expected {count * 2}")
    values = array.array("H")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return torch.tensor(values, dtype=torch.long)


def read_native_logits(path: str) -> tuple[torch.Tensor, int, int]:
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != LOGITS_MAGIC or version != FILE_VERSION:
            raise ValueError(f"{path}: unsupported logits file header magic={magic:#x} version={version}")
        pairs, vocab = struct.unpack("<QQ", f.read(16))
        payload = f.read()
    expected = pairs * vocab * 4
    if len(payload) != expected:
        raise ValueError(f"{path}: logits payload has {len(payload)} bytes, expected {expected}")
    logits = torch.frombuffer(payload, dtype=torch.float32).to(torch.float64).clone()
    return logits.view(int(pairs), int(vocab)), int(pairs), int(vocab)


def score_hf_loss(model, tokens: torch.Tensor, batch_tokens: int, seq_len: int, vocab_size: int) -> tuple[float, int]:
    pairs = tokens.numel() - 1
    if pairs <= 0 or pairs % batch_tokens != 0:
        raise ValueError(f"token pair count {pairs} must be a positive multiple of batch_tokens={batch_tokens}")
    if batch_tokens % seq_len != 0:
        raise ValueError(f"batch_tokens={batch_tokens} must be divisible by seq_len={seq_len}")
    batch_size = batch_tokens // seq_len

    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, pairs, batch_tokens):
            window = tokens[start : start + batch_tokens + 1]
            x = window[:-1].view(batch_size, seq_len)
            y = window[1:].view(-1)
            logits = model(input_ids=x).logits.reshape(-1, vocab_size).to(torch.float64)
            if tuple(logits.shape) != (batch_tokens, vocab_size):
                raise ValueError(f"HF logits shape {tuple(logits.shape)} != ({batch_tokens}, {vocab_size})")
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
    return total_loss / pairs, int(pairs)


def compare_sample_logits(
    model,
    tokens: torch.Tensor,
    native_logits: torch.Tensor,
    sample_pairs: int,
    batch_tokens: int,
    seq_len: int,
    vocab_size: int,
) -> float:
    if sample_pairs <= 0 or sample_pairs % batch_tokens != 0:
        raise ValueError(f"sample_pairs={sample_pairs} must be a positive multiple of batch_tokens={batch_tokens}")
    if native_logits.shape != (sample_pairs, vocab_size):
        raise ValueError(f"native logits shape {tuple(native_logits.shape)} != ({sample_pairs}, {vocab_size})")
    batch_size = batch_tokens // seq_len

    hf_batches = []
    with torch.no_grad():
        for start in range(0, sample_pairs, batch_tokens):
            window = tokens[start : start + batch_tokens + 1]
            x = window[:-1].view(batch_size, seq_len)
            logits = model(input_ids=x).logits.reshape(-1, vocab_size).to(torch.float64)
            hf_batches.append(logits)
    hf_logits = torch.cat(hf_batches, dim=0)
    return (hf_logits - native_logits).abs().max().item()


def check_backbone_and_tokenizer(export_dir: str, model, vocab_size: int) -> int:
    tok = AutoTokenizer.from_pretrained(export_dir)
    if tok.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id is None")

    backbone = AutoModel.from_pretrained(export_dir, trust_remote_code=True)
    backbone.eval()

    encoded = tok(["a b c", "a b"], return_tensors="pt", padding=True)
    with torch.no_grad():
        lm_logits = model(**encoded).logits
        hidden = backbone(**encoded).last_hidden_state
        unmasked_hidden = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=torch.ones_like(encoded["attention_mask"]),
        ).last_hidden_state
    if lm_logits.shape[-1] != vocab_size:
        raise ValueError(f"batched LM vocab dim {lm_logits.shape[-1]} != {vocab_size}")
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "model_dim", 0)))
    if hidden.shape[-1] != hidden_size:
        raise ValueError(f"backbone hidden dim {hidden.shape[-1]} != {hidden_size}")
    if not torch.isfinite(lm_logits).all() or not torch.isfinite(hidden).all():
        raise ValueError("batched AutoModel/AutoModelForCausalLM outputs contain non-finite values")
    if (encoded["attention_mask"] == 0).any():
        mask_diff = (hidden - unmasked_hidden).abs().max().item()
        if mask_diff <= 1e-8:
            raise ValueError("AutoModel hidden states are unchanged when attention_mask hides padding")
    with open(os.path.join(export_dir, "config.json")) as f:
        config_doc = json.load(f)
    if "AutoModelForMaskedLM" in config_doc.get("auto_map", {}):
        if getattr(backbone.blocks[0], "attention_mask", "") == "causal":
            raise ValueError("AutoModel uses causal blocks for a masked-capable export")
    return hidden_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Mixlab exported HF parity evaluator")
    parser.add_argument("--dir", required=True, help="exported Hugging Face directory")
    parser.add_argument("--tokens", required=True, help="binary token stream from mixlab")
    parser.add_argument("--native-logits", required=True, help="binary native sample logits from mixlab")
    parser.add_argument("--batch-tokens", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--vocab-size", required=True, type=int)
    args = parser.parse_args()

    tokens = read_tokens(args.tokens)
    native_logits, sample_pairs, native_vocab = read_native_logits(args.native_logits)
    if native_vocab != args.vocab_size:
        raise ValueError(f"native logits vocab {native_vocab} != --vocab-size {args.vocab_size}")

    if not os.path.isdir(args.dir):
        raise ValueError(f"HF export directory does not exist: {args.dir}")

    model = AutoModelForCausalLM.from_pretrained(args.dir, trust_remote_code=True)
    model.eval()
    hidden_size = check_backbone_and_tokenizer(args.dir, model, args.vocab_size)

    hf_loss, scored_pairs = score_hf_loss(model, tokens, args.batch_tokens, args.seq_len, args.vocab_size)
    max_logit_diff = compare_sample_logits(
        model,
        tokens,
        native_logits,
        sample_pairs,
        args.batch_tokens,
        args.seq_len,
        args.vocab_size,
    )
    print(
        json.dumps(
            {
                "hf_loss": hf_loss,
                "max_logit_diff": max_logit_diff,
                "scored_pairs": scored_pairs,
                "sample_pairs": sample_pairs,
                "backbone_hidden_size": hidden_size,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"HF parity checker error: {exc}", file=sys.stderr)
        sys.exit(2)
