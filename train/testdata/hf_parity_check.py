#!/usr/bin/env python3
"""Native-vs-HF parity comparator for `mixlab -mode export-hf` output.

Loads an exported Hugging Face directory through `AutoModelForCausalLM` with
`trust_remote_code=True`, runs the embedded `modeling_mixlab.py` forward on a
fixed token batch, and compares the resulting logits against the golden logits
produced by the native MLX forward (written by the Go test harness).

This is the load-bearing check that makes the FR-1 convention-drift bug class
impossible by construction: the *actual shipped* Python forward is compared to
the *actual* kernels, not a hand-maintained shadow.

Inputs (all under --dir):
  config.json, modeling_mixlab.py, configuration_mixlab.py, model.safetensors
  parity_tokens.json        -> {"input_ids": [[...]]}  (one [B, T] batch)
  parity_native_logits.json -> {"batch": B, "seq_len": T, "vocab": V,
                                "logits": [...] }       (row-major B*T*V)

Exit status is nonzero with a diagnostic message when either threshold is
exceeded.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Mixlab native-vs-HF logits parity check")
    parser.add_argument("--dir", required=True, help="exported HF directory")
    parser.add_argument("--max-logit-diff", type=float, default=1e-3)
    parser.add_argument("--max-loss-diff", type=float, default=1e-4)
    args = parser.parse_args()

    with open(os.path.join(args.dir, "parity_tokens.json")) as f:
        tokens = json.load(f)["input_ids"]
    with open(os.path.join(args.dir, "parity_native_logits.json")) as f:
        native = json.load(f)

    batch = int(native["batch"])
    seq_len = int(native["seq_len"])
    vocab = int(native["vocab"])
    native_logits = torch.tensor(native["logits"], dtype=torch.float64).view(batch, seq_len, vocab)

    input_ids = torch.tensor(tokens, dtype=torch.long)
    if tuple(input_ids.shape) != (batch, seq_len):
        print(f"input_ids shape {tuple(input_ids.shape)} != ({batch}, {seq_len})", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.dir)
    if tok.pad_token_id is None:
        print("FAIL: tokenizer.pad_token_id is None", file=sys.stderr)
        return 2

    model = AutoModelForCausalLM.from_pretrained(args.dir, trust_remote_code=True)
    model.eval()
    with torch.no_grad():
        hf_logits = model(input_ids=input_ids).logits.to(torch.float64)

    if tuple(hf_logits.shape) != (batch, seq_len, vocab):
        print(f"HF logits shape {tuple(hf_logits.shape)} != ({batch}, {seq_len}, {vocab})", file=sys.stderr)
        return 2

    backbone = AutoModel.from_pretrained(args.dir, trust_remote_code=True)
    backbone.eval()
    encoded = tok(["a b c", "a b"], return_tensors="pt", padding=True)
    with torch.no_grad():
        batched_lm = model(**encoded).logits
        hidden = backbone(**encoded).last_hidden_state
    if batched_lm.shape[-1] != vocab:
        print(f"batched LM vocab dim {batched_lm.shape[-1]} != {vocab}", file=sys.stderr)
        return 2
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "model_dim", 0)))
    if hidden.shape[-1] != hidden_size:
        print(f"backbone hidden dim {hidden.shape[-1]} != {hidden_size}", file=sys.stderr)
        return 2
    if not torch.isfinite(hidden).all() or not torch.isfinite(batched_lm).all():
        print("batched AutoModel/AutoModelForCausalLM outputs contain non-finite values", file=sys.stderr)
        return 2

    with open(os.path.join(args.dir, "config.json")) as f:
        config_doc = json.load(f)
    if "AutoModelForMaskedLM" in config_doc.get("auto_map", {}):
        masked = AutoModelForMaskedLM.from_pretrained(args.dir, trust_remote_code=True)
        masked.eval()
        labels = input_ids.clone()
        if labels.numel() > 1:
            labels[:, :-1] = -100
        with torch.no_grad():
            masked_out = masked(input_ids=input_ids, labels=labels)
        if tuple(masked_out.logits.shape) != (batch, seq_len, vocab):
            print(f"MaskedLM logits shape {tuple(masked_out.logits.shape)} != ({batch}, {seq_len}, {vocab})", file=sys.stderr)
            return 2
        if masked_out.loss is None or not torch.isfinite(masked_out.loss):
            print("MaskedLM loss is missing or non-finite", file=sys.stderr)
            return 2

    if os.environ.get("PARITY_DEBUG") == "1":
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
        print("=== native logits ===\n", native_logits, file=sys.stderr)
        print("=== hf logits ===\n", hf_logits, file=sys.stderr)
        print("=== abs diff ===\n", (hf_logits - native_logits).abs(), file=sys.stderr)

    max_logit_diff = (hf_logits - native_logits).abs().max().item()

    # Mean next-token loss on each side from its own logits, using the shifted
    # labels implied by the input batch. A divergence here independent of the
    # raw logit diff flags a reduction/scaling discrepancy.
    shift_labels = input_ids[:, 1:].reshape(-1)
    native_loss = torch.nn.functional.cross_entropy(
        native_logits[:, :-1, :].reshape(-1, vocab), shift_labels
    ).item()
    hf_loss = torch.nn.functional.cross_entropy(
        hf_logits[:, :-1, :].reshape(-1, vocab), shift_labels
    ).item()
    loss_diff = abs(native_loss - hf_loss)

    print(
        f"parity: max_logit_diff={max_logit_diff:.3e} "
        f"loss_diff={loss_diff:.3e} native_loss={native_loss:.6f} hf_loss={hf_loss:.6f}"
    )

    ok = True
    if max_logit_diff >= args.max_logit_diff:
        print(f"FAIL: max_logit_diff {max_logit_diff:.3e} >= {args.max_logit_diff:.3e}", file=sys.stderr)
        ok = False
    if loss_diff >= args.max_loss_diff:
        print(f"FAIL: loss_diff {loss_diff:.3e} >= {args.max_loss_diff:.3e}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
