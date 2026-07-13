#!/usr/bin/env python3
"""Native-vs-HF parity comparator for `mixlab -mode export-hf` output.

Loads an exported Hugging Face directory through either the custom Mixlab
`AutoModelForCausalLM` path or a stock `GPT2LMHeadModel`, runs a fixed token
batch, and compares the resulting logits against the golden logits produced by
the native MLX forward (written by the Go test harness).

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
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, GPT2LMHeadModel


def compare_logits(args, input_ids, native_logits, hf_logits, batch, seq_len, vocab) -> int:
    if tuple(hf_logits.shape) != (batch, seq_len, vocab):
        print(f"HF logits shape {tuple(hf_logits.shape)} != ({batch}, {seq_len}, {vocab})", file=sys.stderr)
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


def is_native_gpt2_export(config_doc) -> bool:
    architectures = config_doc.get("architectures", [])
    return config_doc.get("model_type") == "gpt2" or "GPT2LMHeadModel" in architectures


def has_ttt_mlp(config_doc) -> bool:
    return any(str(block.get("type", "")).lower() == "ttt_mlp" for block in config_doc.get("blocks", []))


def compare_ttt_cached_state(args, model, input_ids, vocab, path) -> int:
    with open(path) as f:
        expected = json.load(f)
    split = int(expected["split"])
    with torch.no_grad():
        prefix = model(input_ids=input_ids[:, :split], use_cache=True)
        continuation = model(
            input_ids=input_ids[:, split:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
    expected_logits = torch.tensor(
        expected["continuation_logits"], dtype=torch.float64
    ).view(1, input_ids.shape[1] - split, vocab)
    actual_logits = continuation.logits.to(torch.float64)
    max_logit_diff = (actual_logits - expected_logits).abs().max().item()
    if max_logit_diff >= args.max_logit_diff:
        print(
            f"FAIL: TTT cached max_logit_diff {max_logit_diff:.3e} >= {args.max_logit_diff:.3e}",
            file=sys.stderr,
        )
        return 1

    states = continuation.past_key_values
    if states is None or len(states) != len(expected["blocks"]):
        print("FAIL: TTT cached state block count mismatch", file=sys.stderr)
        return 2
    max_state_diff = 0.0
    for idx, (state, want) in enumerate(zip(states, expected["blocks"])):
        if int(state.offset) != int(want["offset"]):
            print(
                f"FAIL: TTT block {idx} offset={state.offset} want={want['offset']}",
                file=sys.stderr,
            )
            return 1
        for field in ("mlp", "gradient", "conv"):
            actual = getattr(state, field).detach().to(torch.float64).reshape(-1)
            target = torch.tensor(want[field], dtype=torch.float64)
            if actual.numel() != target.numel():
                print(
                    f"FAIL: TTT block {idx} {field} size={actual.numel()} want={target.numel()}",
                    file=sys.stderr,
                )
                return 2
            if actual.numel() > 0:
                max_state_diff = max(max_state_diff, (actual - target).abs().max().item())
    print(
        f"ttt_cached_parity: max_logit_diff={max_logit_diff:.3e} "
        f"max_state_diff={max_state_diff:.3e}"
    )
    if max_state_diff >= args.max_logit_diff:
        print(
            f"FAIL: TTT cached max_state_diff {max_state_diff:.3e} >= {args.max_logit_diff:.3e}",
            file=sys.stderr,
        )
        return 1
    return 0


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
    native_masked_logits = None
    masked_path = os.path.join(args.dir, "parity_native_masked_logits.json")
    if os.path.exists(masked_path):
        with open(masked_path) as f:
            native_masked = json.load(f)
        masked_batch = int(native_masked["batch"])
        masked_seq_len = int(native_masked["seq_len"])
        masked_vocab = int(native_masked["vocab"])
        if (masked_batch, masked_seq_len, masked_vocab) != (batch, seq_len, vocab):
            print(
                "native masked logits shape metadata "
                f"({masked_batch}, {masked_seq_len}, {masked_vocab}) != ({batch}, {seq_len}, {vocab})",
                file=sys.stderr,
            )
            return 2
        native_masked_logits = torch.tensor(native_masked["logits"], dtype=torch.float64).view(batch, seq_len, vocab)

    input_ids = torch.tensor(tokens, dtype=torch.long)
    if tuple(input_ids.shape) != (batch, seq_len):
        print(f"input_ids shape {tuple(input_ids.shape)} != ({batch}, {seq_len})", file=sys.stderr)
        return 2

    with open(os.path.join(args.dir, "config.json")) as f:
        config_doc = json.load(f)

    if is_native_gpt2_export(config_doc):
        model = GPT2LMHeadModel.from_pretrained(args.dir)
        model.eval()
        with torch.no_grad():
            hf_logits = model(input_ids=input_ids).logits.to(torch.float64)
        return compare_logits(args, input_ids, native_logits, hf_logits, batch, seq_len, vocab)

    tok = AutoTokenizer.from_pretrained(args.dir)
    if tok.pad_token_id is None:
        print("FAIL: tokenizer.pad_token_id is None", file=sys.stderr)
        return 2

    model = AutoModelForCausalLM.from_pretrained(args.dir, trust_remote_code=True)
    model.eval()
    with torch.no_grad():
        hf_logits = model(input_ids=input_ids).logits.to(torch.float64)

    ttt_state_path = os.path.join(args.dir, "parity_ttt_state.json")
    if os.path.exists(ttt_state_path):
        status = compare_ttt_cached_state(args, model, input_ids, vocab, ttt_state_path)
        if status != 0:
            return status

    backbone = AutoModel.from_pretrained(args.dir, trust_remote_code=True)
    backbone.eval()
    encoded = tok(["a b c", "a b"], return_tensors="pt", padding=True)
    with torch.no_grad():
        batched_lm = model(**encoded).logits
        hidden = backbone(**encoded).last_hidden_state
        unmasked_hidden = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=torch.ones_like(encoded["attention_mask"]),
        ).last_hidden_state
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
    if (encoded["attention_mask"] == 0).any() and not has_ttt_mlp(config_doc):
        mask_diff = (hidden - unmasked_hidden).abs().max().item()
        if mask_diff <= 1e-8:
            print("FAIL: AutoModel hidden states are unchanged when attention_mask hides padding", file=sys.stderr)
            return 2

    if "AutoModelForMaskedLM" in config_doc.get("auto_map", {}):
        if getattr(backbone.blocks[0], "attention_mask", "") == "causal":
            print("FAIL: AutoModel uses causal blocks for a masked-capable export", file=sys.stderr)
            return 2
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
        if tok.mask_token is None or tok.mask_token_id is None:
            print("FAIL: tokenizer.mask_token is None for a masked-LM export", file=sys.stderr)
            return 2
        if native_masked_logits is not None:
            masked_status = compare_logits(
                args, input_ids, native_masked_logits, masked_out.logits.to(torch.float64), batch, seq_len, vocab
            )
            if masked_status != 0:
                return masked_status
    elif native_masked_logits is not None:
        print("native masked logits were provided, but export has no AutoModelForMaskedLM", file=sys.stderr)
        return 2

    return compare_logits(args, input_ids, native_logits, hf_logits, batch, seq_len, vocab)


if __name__ == "__main__":
    sys.exit(main())
