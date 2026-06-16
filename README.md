# mixlab

Explore ML architectures fast. Define a model in JSON, train it on your Mac
in seconds and iterate until you find a winner, then ship the same config to 
a cloud GPU for full-scale runs. One JSON file, two platforms, no code changes.

```text
laptop (Metal)                          cloud GPU (CUDA)
mixlab -config my_model.json    ===>    mixlab -config my_model.json
       -train 'data/*.bin'                     -train 'data/*.bin'
```

mixlab compiles JSON configs into a typed Go IR and executes them on GPU
through the MLX backend. No Python model code to write, no framework
translation between local and remote. Supports plain transformers, GQA, Mamba, RetNet, RWKV, Perceiver,
U-Net layouts, and fully custom JSON-defined blocks.

**Platforms:** macOS (Apple Silicon) and Linux (NVIDIA CUDA via Docker).

## Prerequisites (macOS)

- Apple Silicon Mac (M1 or later)
- Go 1.24+ ([go.dev/dl](https://go.dev/dl/))
- Python 3.10+ (for data preparation)
- Xcode Command Line Tools (`xcode-select --install`)

## Quickstart (macOS)

```bash
# 1. Set up Python environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy tokenizers

# 2. Build mixlab
make build

# 3. Download example data (~5 MB of public domain text)
bash scripts/download_example_data.sh

# 4. Train a 3-layer attention model
./mixlab -mode arch \
    -config examples/plain_3L.json \
    -train 'data/example/train_*.bin'
```

You should see training loss printed every 100 steps. The example config trains
for 200 steps and finishes in seconds.

## Quickstart (Docker / NVIDIA GPU)

```bash
# Pull the pre-built CLI image, or build your own; see docker/README.md.
docker pull michaelrothrock/mixlab:latest

# Smoke test
docker run --gpus all michaelrothrock/mixlab:latest -mode smoke

# Train (mount your data directory)
docker run --gpus all -v $(pwd)/data:/data michaelrothrock/mixlab:latest \
    -mode arch -config /examples/plain_3L.json -train '/data/*.bin'
```

## Features

- JSON-first model definition: no Go or Python changes required for most experiments.
- Built-in block families: `plain`, `swiglu`, `mamba`, `gated_linear_ssm`,
  `mamba3-canonical`, `retnet`, `rwkv`, `perceiver`, `bottleneck`,
  `cross_attention`, `token_blend`, `mlp`, `custom`.
- Architecture features: U-Net skip connections, parallel residuals, recurrence,
  residual mixing, tied embeddings, hashed bigram embeddings, configurable MLP width,
  QK-Gain, partial RoPE, XSA (V-orthogonal projection).
- Trainer features: configurable optimizer (Muon or AdamW for matrix weights),
  configurable weight init (Xavier uniform or normal), phase-based LR schedules,
  SWA/EMA averaging, validation-loss early stopping, LoRA-TTT (`ttt_mode: "lora"`),
  safetensors import/export.
- Compute accounting: analytical FLOPs estimation, tok/s throughput, optional MFU
  logging via `training.hardware_tflops`.
- Quantization: int8 and int6 with per-row scaling, SDClip clipping (`-quant-method sdclip`),
  quantization-aware training (`training.qat: "int6"`).
- Public Go API hooks: reusable `train.InferenceSession` for repeated evals and
  `train.TrainOptions.OptimizerOverride` for custom optimizer-group plans such
  as frozen-weight classes.
- Custom blocks: declare weights and op graphs directly in JSON.

## When to use mixlab

**Good fit:**
- Rapid architecture iteration — edit a JSON config, train on Metal, see results in seconds
- Mac-first workflow — prototype on Apple Silicon, scale to cloud GPU with the same config
- Comparing block families (attention vs Mamba vs RetNet) on the same data
- Teaching/learning — visible JSON configs make architecture choices explicit
- Fast block development — import mixlab, register your block, build in <2s, test in ~5s

**Not the right tool for:**
- Production training at scale — use PyTorch, JAX, or dedicated frameworks
- Custom CUDA kernels or operator-level debugging
- Distributed training across multiple GPUs

mixlab is an architecture exploration tool, not a training framework.
It trades generality for speed of iteration.

**Extensibility:** Need ops beyond JSON custom blocks? Create a Go package
that imports `github.com/mrothroc/mixlab/arch`, registers new block types
via `arch.RegisterBlock()`, and emits IR using the public op API
(`prog.MatMul()`, `prog.RMSNorm()`, etc.). Your blocks compile into the
same binary and automatically inherit everything mixlab provides: Metal and
CUDA backends, the training loop, Muon/AdamW optimizers, safetensors
import/export, checkpointing, profiling, and all CLI modes. No C++
extensions, no custom build systems — just a Go import and an `init()`
function. See [`arch/registry.go`](arch/registry.go) for the registration
API.

## Install

### Homebrew (recommended)

```bash
brew install mrothroc/tap/mixlab
```

This installs MLX automatically as a dependency.

For data preparation (`prepare` mode) and FineWeb downloads, also install Python deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy tokenizers datasets
```

### Build from source (macOS Apple Silicon)

Requires Go 1.24+ and MLX (`brew install mlx` or `pip install mlx`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy tokenizers   # MLX + data preparation deps
make build
```

This produces a `mixlab` binary in the project root.

### Docker (NVIDIA CUDA)

For Linux with an NVIDIA GPU. The pre-built images support A100, A40,
RTX 3090, RTX 4090, L40, and L40S (sm_80/86/89). For other GPUs (H100,
RTX 5090), see [docker/README.md](docker/README.md) to build with your
architecture.

```bash
# Pull the pre-built CLI image.
docker pull michaelrothrock/mixlab:latest

# Run smoke test
docker run --gpus all michaelrothrock/mixlab:latest -mode smoke

# Train a model (mount your data directory)
docker run --gpus all -v $(pwd)/data:/data michaelrothrock/mixlab:latest \
    -mode arch -config /examples/plain_3L.json -train '/data/train_*.bin'
```

## Usage

mixlab has eight modes, selected with `-mode`:

| Mode | Description |
|------|-------------|
| `arch` | Train a single architecture from a JSON config. The default mode. |
| `arch_race` | Train every JSON config in a directory and compare results. |
| `smoke` | Run diagnostic checks (MLX availability, GPU health). |
| `prepare` | Tokenize raw text or JSONL into binary training shards. |
| `count` | Print parameter, size, block, and IR op counts for a config. |
| `eval` | Load safetensors and evaluate validation loss. |
| `hiddenstats` | Export one batch of hidden states as float32 binary. |
| `generate` | Generate token IDs from a safetensors checkpoint. |

### arch (default)

```bash
./mixlab -mode arch -config examples/plain_3L.json -train 'data/example/train_*.bin'
```

Additional flags:

| Flag | Description |
|------|-------------|
| `-eval` | Run full validation BPB evaluation after training |
| `-safetensors FILE` | Export weights to safetensors after training |
| `-safetensors-load FILE` | Load weights before training (resume or eval-only) |
| `-quantize MODE` | Weight quantization: `none` (default), `int8`, `int6` |
| `-lut-dir DIR` | Directory for BPB lookup tables (default: `data`) |
| `-checkpoint-dir DIR` | Directory for periodic safetensors checkpoints |
| `-checkpoint-every N` | Save a checkpoint every `N` training steps (`0` disables) |

### arch_race

```bash
./mixlab -mode arch_race -configs examples/ -train 'data/example/train_*.bin'
```

Trains every `.json` config in the given directory and prints a ranked summary.

### count

```bash
./mixlab -mode count -config examples/plain_3L.json
```

### eval

```bash
./mixlab -mode eval -config examples/plain_3L.json \
  -safetensors-load weights.st -train 'data/example/train_*.bin'
```

Per-token export flags (all optional, can be combined in a single eval pass):

| Flag | Output |
|------|--------|
| `-logprobs-out PATH` | Binary file of per-token NLLs (`logprobs.Record{TokenID, NLL}`); enables BPB / perplexity post-processing. |
| `-ranks-out PATH` | Binary file of per-token target ranks (`ranks.Record{TokenID, Rank}`); enables Hit@K, MRR, and rank-conditional calibration. Rank is 0-indexed; rank 0 means the target was the model's argmax. |
| `-uncertainty-out PATH` | Binary file of per-token candidate uncertainty (`uncertainty.Record{TokenID, Top1Prob, Entropy, Margin}`); enables selective prediction, abstention, calibration, and decoding diagnostics without using the gold token. |
| `-logits-out PATH` | Binary file of per-token full-vocab outputs (`logits.Record{TokenID, Values[vocab]}`); enables vocabulary masking, log-prob composition (e.g. product-of-experts), routing-signal sweeps, ensembling, distillation, and any analysis requiring `p(v \| context)` for arbitrary `v`. Use `-logits-dtype float16` (default) or `float32`, and `-logits-form raw` (default) or `logprobs` (stores `log_softmax(logits)`). |

When multiple export flags are supplied, the records are aligned
position-by-position (same `TokenID` at each index) and are derived from a
single GPU pass over the validation shard.

Example reading ranks.bin from Python:

```python
import struct, numpy as np
header = struct.Struct("<IIII")  # magic, version, vocab, total_tokens
record = np.dtype([("token_id", "<u2"), ("rank", "<u2")])
with open("ranks.bin", "rb") as f:
    _, _, vocab, n = header.unpack(f.read(16))
    arr = np.fromfile(f, dtype=record, count=n)
hit_at_1  = (arr["rank"] == 0).mean()
hit_at_5  = (arr["rank"] <  5).mean()
hit_at_10 = (arr["rank"] < 10).mean()
mrr       = (1.0 / (arr["rank"].astype(np.float64) + 1.0)).mean()
print(f"Hit@1={hit_at_1:.4f} Hit@5={hit_at_5:.4f} Hit@10={hit_at_10:.4f} MRR={mrr:.4f}")
```

Example reading uncertainty.bin from Python:

```python
import struct, numpy as np
header = struct.Struct("<IIII")  # magic, version, vocab, total_tokens
record = np.dtype([("token_id", "<u2"), ("top1_prob", "<f4"),
                   ("entropy", "<f4"), ("margin", "<f4")])
with open("uncertainty.bin", "rb") as f:
    _, _, vocab, n = header.unpack(f.read(16))
    arr = np.fromfile(f, dtype=record, count=n)
mean_top1 = arr["top1_prob"].mean()
mean_entropy = arr["entropy"].mean()
mean_margin = arr["margin"].mean()
threshold = np.quantile(arr["top1_prob"], 0.10)
abstain_mask = arr["top1_prob"] < threshold
print(f"mean top1_prob={mean_top1:.4f} entropy={mean_entropy:.4f} margin={mean_margin:.4f}")
print(f"abstaining on bottom 10% by top1_prob: threshold={threshold:.4f} positions={abstain_mask.sum()}")
```

Example reading logits.bin from Python (the 20-byte header carries the dtype
and form, so a single reader handles both float16/float32 and raw/logprobs
variants):

```python
import struct, numpy as np

header = struct.Struct("<IIIIBB2s")  # magic, version, vocab, total, dtype, form, reserved
DTYPE = {0: np.float16, 1: np.float32}
FORM  = {0: "raw",      1: "logprobs"}

with open("logits.bin", "rb") as f:
    magic, version, vocab, n, dtype_id, form_id, _ = header.unpack(f.read(20))
    if magic != 0x4C4F4754 or version != 1:
        raise ValueError(f"unexpected logits header: magic={magic:#x} version={version}")
    dt = DTYPE[dtype_id]
    record = np.dtype([("token_id", "<u2"), ("logits", dt, (vocab,))])
    arr = np.fromfile(f, dtype=record, count=n)
print(f"loaded {n} positions @ vocab={vocab} dtype={dt.__name__} form={FORM[form_id]}")

# Example 1: recover the model's top-1 prediction at each position.
top1 = arr["logits"].argmax(axis=1)

# Example 2: re-derive per-token NLLs from raw logits.
if FORM[form_id] == "raw":
    row = arr["logits"].astype(np.float32)
    logZ = np.log(np.exp(row - row.max(axis=1, keepdims=True)).sum(axis=1)) + row.max(axis=1)
    nll  = logZ - row[np.arange(n), arr["token_id"]]
    print(f"mean NLL = {nll.mean():.6f}")
```

When `-logits-out` is combined with `-logprobs-out`, the recovered NLLs match
the values in `logprobs.bin` to float32/float16 tolerance (this is enforced
by the eval-mode parity tests).

### hiddenstats

```bash
./mixlab -mode hiddenstats -config examples/plain_3L.json \
  -safetensors-load weights.st -train 'data/example/train_*.bin' \
  -output hidden.bin
```

### generate

```bash
./mixlab -mode generate -config examples/plain_3L.json \
  -safetensors-load weights.st -prompt token_ids:0,1,2
```

| Flag | Description |
|------|-------------|
| `-max-tokens` | Maximum generated tokens (default: `256`) |
| `-temperature` | Sampling temperature (default: `0.8`) |
| `-top-k` | Top-k sampling cutoff; `0` disables the cutoff |
| `-prompt` | Prompt token IDs in `token_ids:0,1,2` form |

### prepare

Requires Python 3 with `numpy` and `tokenizers` (`pip install numpy tokenizers`).
Tokens are stored as uint16, so `vocab-size` must be 65,535 or less.

```bash
./mixlab -mode prepare -input raw_text/ -output data/shards/ -vocab-size 1024
```

| Flag | Description |
|------|-------------|
| `-input` | Input text file, JSONL file, or directory |
| `-output` | Output directory for binary shards |
| `-vocab-size` | BPE vocabulary size (default: 1024) |
| `-val-split` | Fraction of tokens reserved for validation (default: 0.1) |
| `-tokenizer-path` | Path to a pre-trained `tokenizer.json` (optional) |
| `-text-field` | JSON field name for text in JSONL input (default: `text`) |

## Training data

### Quick start: example data (~5 MB)

The quickstart above uses `scripts/download_example_data.sh` to download
public domain books from Project Gutenberg. This is enough to verify your
setup and see loss curves, but too small for real experiments.

### Real-world: FineWeb-Edu (10B tokens)

For serious architecture exploration, use
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(a curated, deduplicated web text corpus). The download script handles
everything: fetching the data from HuggingFace, training a BPE tokenizer at
your chosen vocab size, and writing binary shards ready for mixlab.

```bash
pip install numpy tokenizers datasets   # if not already installed

# Quick test (~30 seconds, streams 5000 documents)
python3 scripts/download_fineweb.py --output data/fineweb_sp1024 \
    --vocab-size 1024 --max-docs 5000

# Full dataset — SP-1024 (small vocab, fast iteration, good for architecture search)
python3 scripts/download_fineweb.py --output data/fineweb_sp1024 --vocab-size 1024

# Full dataset — SP-8192 (larger vocab, better BPB for production models)
python3 scripts/download_fineweb.py --output data/fineweb_sp8192 --vocab-size 8192
```

The `--max-docs` flag streams only a subset — useful for verifying your setup
before committing to the full download. Without it, the first run downloads
~20 GB from HuggingFace (cached for subsequent runs). Tokenization takes
30-60 minutes. The output is a set of binary shards plus a tokenizer and BPB
lookup tables.

Train on the prepared data:

```bash
mixlab -mode arch -config examples/plain_3L.json \
    -train 'data/fineweb_sp1024/train_*.bin'

# Race two architectures head-to-head
mixlab -mode arch_race -configs examples/ \
    -train 'data/fineweb_sp8192/train_*.bin'
```

### Bring your own data

mixlab can tokenize any UTF-8 text file, directory of text files, or JSONL:

```bash
# Single text file
mixlab -mode prepare -input corpus.txt -output data/my_data -vocab-size 1024

# Directory of text files
mixlab -mode prepare -input texts/ -output data/my_data -vocab-size 4096

# JSONL (specify which field contains text)
mixlab -mode prepare -input data.jsonl -output data/my_data \
    -vocab-size 8192 -text-field content
```

Or use a pre-trained tokenizer:

```bash
mixlab -mode prepare -input corpus.txt -output data/my_data \
    -tokenizer-path path/to/tokenizer.json
```

### Data/config compatibility

**The `vocab_size` in your JSON config must match the tokenizer used to create
the `.bin` shards.** This is the most common source of silent failures. The
example data uses vocab 1024; the larger example configs use 8192. If you see
garbage loss values, check this first.

```jsonc
// Config says vocab_size: 1024 → train on SP-1024 shards
{"vocab_size": 1024, ...}   // use with data/fineweb_sp1024/train_*.bin

// Config says vocab_size: 8192 → train on SP-8192 shards
{"vocab_size": 8192, ...}   // use with data/fineweb_sp8192/train_*.bin
```

## Customizing configs

Start by copying `examples/plain_3L.json` and changing these five knobs:

| Knob | Field | Effect |
|------|-------|--------|
| **Width** | `model_dim` | Hidden size. Larger = more capacity, more memory |
| **Depth** | Add more blocks to `blocks` array | More layers = more capacity |
| **Heads** | `heads` on `plain` blocks | Attention heads (must divide `model_dim`) |
| **Vocab** | `vocab_size` | Must match your tokenizer (see above) |
| **Context** | `seq_len` | Sequence length in tokens |

To try a different block family, swap `"type": "plain"` for `"mamba"`,
`"retnet"`, `"rwkv"`, or any other registered type. See
[docs/config-reference.md](docs/config-reference.md) for all fields and
[examples/](examples/) for working configs at various complexity levels.

## Architecture

```text
JSON config --> Go IR builder --> IR program (typed ops) --> MLX runtime
                                                         --> Metal (macOS)
                                                         --> CUDA (Linux)
```

The Go IR builder compiles JSON configs into a typed intermediate
representation. The runtime executes that IR on GPU and applies grouped
optimization: AdamW for embedding/head/scalar groups and Muon for matrix
weights.

Sequential models can run as plain stacks or as U-Net layouts with learned skip
connections. Config features such as `parallel_residual`, `recurrence`,
`resid_mix`, `tie_embeddings`, and `mlp_mult` all lower directly into the IR.

## Block types

These built-in block types are available in JSON configs:

| Type | Description |
|------|-------------|
| `plain` | Causal self-attention + FFN. Requires `heads`. Optional `kv_heads` (GQA), `qk_gain` (learnable QK scaling), `rope_dims` (partial RoPE), `xsa` (V-orthogonal projection), `kv_source` (share K/V from earlier block). |
| `swiglu` | SwiGLU feed-forward block with residual connection. |
| `mamba` | Mamba selective state-space block. Optional `inner_dim`. |
| `gated_linear_ssm` | Simplified gated linear SSM formerly exposed as `mamba3`. Optional `inner_dim`. |
| `mamba3-canonical` | Canonical Mamba-3 block. Optional `inner_dim`, `state_size`, `n_groups`, `dt_rank`, `conv_kernel`, `use_conv`, `scan_chunk_size` (default `64`; `0` restores full-sequence scan). |
| `retnet` | RetNet retention block. Requires `heads`. Optional `decay` field in config. |
| `rwkv` | RWKV-style recurrent mixing block. |
| `perceiver` | Perceiver latent bottleneck. Requires `heads`. Optional `num_latents`. |
| `bottleneck` | Smaller latent bottleneck block. Requires `heads`. Optional `num_latents`. |
| `cross_attention` | Cross-attention from the current stream into `source_stream`. Requires `heads`, `source_stream`. |
| `token_blend` | Learned token blending gate over adjacent positions. |
| `mlp` | Feed-forward with configurable activation (`silu`, `gelu`, `relu`, `leaky_relu_sq`). 2-weight layout (up + down), lighter than `swiglu`. |
| `custom` | User-defined block with declared weights and IR ops. |


## Advanced Architecture Example

`examples/recurrent_parallel.json` demonstrates the most advanced config surface —
depth recurrence, parallel residuals, GQA, tied embeddings, and Muon optimizer:

```jsonc
{
  "name": "recurrent_parallel",
  "model_dim": 512,
  "vocab_size": 8192,
  "seq_len": 2048,
  "tie_embeddings": true,       // head reuses embedding weights
  "parallel_residual": true,    // attn + MLP see same input
  "block_scales": true,         // learned per-block output scaling
  "resid_mix": true,            // learned blend with original embedding
  "recurrence": [0,1,2,3,4,5,2,3,4,5]  // layers 3-4 reuse weights from 1-2
}
```

See also `examples/unet_transformer.json` for a U-Net architecture with skip
connections, and the other examples for commented configs covering the common
feature paths.

## Custom blocks

Custom blocks let you define novel architectures entirely in JSON.

```json
{
  "type": "custom",
  "name": "geglu",
  "weights": [
    {"name": "w_gate", "shape": ["D", "FFN"]},
    {"name": "w_up", "shape": ["D", "FFN"]},
    {"name": "w_down", "shape": ["FFN", "D"]}
  ],
  "ops": [
    {"op": "matmul", "inputs": ["x", "w_gate"], "output": "gate"},
    {"op": "silu", "inputs": ["gate"], "output": "gate_act"},
    {"op": "matmul", "inputs": ["x", "w_up"], "output": "up"},
    {"op": "mul", "inputs": ["gate_act", "up"], "output": "ff"},
    {"op": "matmul", "inputs": ["ff", "w_down"], "output": "ff_out"},
    {"op": "add", "inputs": ["x", "ff_out"], "output": "x"}
  ]
}
```

See `examples/custom_geglu.json` for a runnable example and
[`docs/config-reference.md`](docs/config-reference.md#custom-blocks) for the
canonical custom-block schema, shape symbols, supported-op reference, and
training fields.

## Troubleshooting

### MLX not found / wrong Python version

The Makefile auto-detects your MLX install path via `python3 -c "import mlx; ..."`.
Run `make check-mlx` to see what it found.

If detection fails (e.g. MLX is in a virtualenv or a different Python):

```bash
# Point to MLX manually
make build MLX_PREFIX=$(python3.12 -c "import mlx, os; print(os.path.dirname(mlx.__file__))")

# Or export it for the session
export MLX_PREFIX=/opt/homebrew/lib/python3.12/site-packages/mlx
make build
```

## Performance

On Apple M1 Max (Metal): ~8.5 seconds per 100 training steps at `d=1024`,
`seq_len=1024`. Smaller models (`d=128`) train in seconds.

**MLX graph tuning:** MLX can default to small graph batches on some GPUs.
mixlab auto-tunes this based on your model's IR op count, with a larger floor
for `gated_deltanet` because its compact IR scan op expands into many MLX ops.
This typically gives ~10% speedup on GPUs like the A40. If you still see low
GPU utilization, you can override manually:

```bash
MLX_MAX_OPS_PER_BUFFER=2000 MLX_MAX_MB_PER_BUFFER=4000 ./mixlab -mode arch ...
```

Higher values batch more kernels into each graph, reducing dispatch overhead at
the cost of more GPU memory. The auto-tuned default (3x IR ops, or at least
16000 ops for `gated_deltanet`) captures most of the benefit.

### Step timing breakdown

Add `-timing` to see where each training step spends time:

```bash
./mixlab -mode arch -config model.json -train 'data/*.bin' -timing
```

Prints at each progress interval:
```
[model] [timing] data=1.2ms gpu=142.5ms val=11.3ms log=0.2ms
```

- `data` — time waiting for the next batch (should be ~0 if prefetch keeps up)
- `gpu` — forward + backward + optimizer (the useful work)
- `val` — validation loss computation (0 on non-validation steps)
- `log` — progress formatting

If `data` is consistently high, the data loader can't keep up with the GPU.
If `gpu` dominates and `data` is near zero, the pipeline is healthy.

### Profiling

mixlab is written in Go, which has built-in profiling with zero overhead
when disabled. No extra tools to install.

```bash
# CPU profile — where is time spent?
./mixlab -mode arch -config my_model.json -train 'data/*.bin' -cpuprofile cpu.prof
go tool pprof -http :8080 cpu.prof    # interactive flame graph in your browser

# Memory profile — what's allocating?
./mixlab -mode arch -config my_model.json -train 'data/*.bin' -memprofile mem.prof
go tool pprof mem.prof
```

Both flags are safe for real training runs — profiling adds negligible
overhead and the output is a standard pprof file that works with `go tool
pprof`, Speedscope, or any pprof-compatible viewer.

**Remote GPU profiling (RunPod / cloud):** Generate a signed upload URL,
pass the training command in `setup`, and upload the profile in `post`:

```bash
# Generate a signed URL (1 hour expiry)
gcloud storage sign-url gs://your-bucket/profiles/cpu.prof \
    --http-verb=PUT --duration=1h \
    --impersonate-service-account=your-sa@project.iam.gserviceaccount.com

# Submit RunPod job with profiling
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{
  "input": {
    "setup": [
      "mixlab -mode arch -config /examples/plain_3L.json -train /data/*.bin -cpuprofile /tmp/cpu.prof"
    ],
    "mode": "smoke",
    "post": [
      "curl -X PUT -H '"'"'Content-Type: application/octet-stream'"'"' --data-binary @/tmp/cpu.prof '"'"'SIGNED_URL'"'"'"
    ]
  }
}'

# Download and view the flame graph locally
gcloud storage cp gs://your-bucket/profiles/cpu.prof .
go tool pprof -http :8080 cpu.prof
```

## Contributing

Before submitting changes, run `make test` from the repository root. The block
registry in `arch/registry.go` and config validation in `arch/config.go` must
both be updated when adding a new block type.

## License

MIT. See [LICENSE](LICENSE).

## Author

[Michael Rothrock](https://michael.roth.rocks) · [GitHub](https://github.com/mrothroc)

Other work:
- [Trust Topology](https://michael.roth.rocks/research/trust-topology/) — a framework for engineering reliability from unreliable AI agents
- [claude-code-log-analyzer](https://github.com/mrothroc/claude-code-log-analyzer) — compute overlap ratios on your own Claude Code session logs
