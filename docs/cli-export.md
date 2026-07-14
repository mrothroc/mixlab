# CLI: Hugging Face Export And Parity

Use `export-hf` to write a Hugging Face directory, then use `parity` to compare
that directory against native Mixlab inference on real token shards.

## `export-hf`

Export a supported Mixlab checkpoint:

```bash
./mixlab -mode export-hf \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -export-dir runs/plain_3L/hf \
  -tokenizer-path data/example/tokenizer.json
```

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON config used to train the checkpoint. |
| `-safetensors-load` | Required. Safetensors checkpoint to export. |
| `-export-dir` | Destination Hugging Face model directory. Preferred alias for legacy `-output`. |
| `-output` | Legacy destination directory. |
| `-tokenizer-path` | Tokenizer JSON to bundle with the export. |

The default export format is Mixlab custom-code Hugging Face export. Configs
with `hf_export_format: "gpt2"` export as native `GPT2LMHeadModel` when they
meet the strict GPT-2 compatibility rules. See [hf-export.md](hf-export.md)
for supported features, sequence-classification pooling, and load examples.

## `parity`

Compare a Hugging Face export against native Mixlab inference:

```bash
./mixlab -mode parity \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -hf runs/plain_3L/hf \
  -train 'data/example/val_*.bin'
```

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON config used for native inference. |
| `-safetensors-load` | Required. Native safetensors checkpoint to compare. |
| `-hf` | Required. Hugging Face export directory to load. |
| `-train` | Required. Shard glob used as the comparison token stream. |
| `-parity-loss-threshold` | Maximum allowed native-vs-HF mean NLL difference. Preferred alias for legacy `-threshold`. Default: `0.05`. |
| `-threshold` | Legacy parity loss threshold flag. |
| `-max-logit-diff` | Maximum allowed absolute logit difference on sampled rows. `<=0` disables the logit gate. Default: `0.001`. |
| `-parity-logit-tokens` | Number of token pairs to sample for logit comparison, rounded up to full eval batches. `0` uses one batch. |
| `-parity-python` | Python interpreter for the HF checker. Defaults to `HF_PARITY_PYTHON` or `python3`. |

The Python checker needs the packages in `requirements-hf.txt`. Run `parity`
after changing export templates, weight mapping, tokenizer metadata, or
supported block features.
