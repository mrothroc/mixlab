# Releasing a new version of mixlab

## Version scheme

Semantic versioning: `vMAJOR.MINOR.PATCH`
- MAJOR: breaking config/API changes
- MINOR: new features, performance improvements
- PATCH: bug fixes

## Checklist

### 1. Pre-release

- [ ] All changes committed and pushed
- [ ] `go test ./...` passes
- [ ] `CGO_ENABLED=1 go test -tags mlx ./gpu/...` passes
- [ ] Quick smoke: `./mixlab -mode arch -config examples/plain_3L.json -train 'data/*.bin'`

### 2. Tag

```bash
git tag -a vX.Y.Z -m "$(cat <<'EOF'
vX.Y.Z

- feature: ...
- fix: ...
EOF
)"
git push --tags
```

### 3. GitHub Release

```bash
gh release create vX.Y.Z --title "vX.Y.Z: Title" --notes "$(cat <<'EOF'
### Feature Name

Description.

### Other Changes

- **fix**: ...
- **refactor**: ...

### Install

\`\`\`bash
brew tap mrothroc/mixlab https://github.com/mrothroc/mixlab
brew install mixlab
\`\`\`

Or build from source:

\`\`\`bash
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab/
\`\`\`
EOF
)"
```

### 4. Homebrew Formula

Update `Formula/mixlab.rb`:
```ruby
url "https://github.com/mrothroc/mixlab.git",
    tag:      "vX.Y.Z",
    revision: "<full commit hash from git rev-parse vX.Y.Z^{commit}>"
```

Commit and push:
```bash
git add Formula/mixlab.rb
git commit -m "chore: update formula to vX.Y.Z"
git push
```

### 5. Verify

```bash
brew upgrade mixlab  # or brew install mrothroc/mixlab/mixlab
mixlab -mode smoke
```

### 6. Downstream (if applicable)

- Update `go.mod` in mixlab-jazz: `go get github.com/mrothroc/mixlab@vX.Y.Z`
- Rebuild and push RunPod Docker image if GPU-side changes

## Known gotchas

- **MLX API drift between Homebrew and the local MLX install.** Homebrew `mlx` may ship a newer version than what `mlx-c` headers expect (e.g. the v0.31.2 Metal `Device::get_command_encoder` → `metal::get_command_encoder` change). Verify the formula install (`brew test mrothroc/mixlab/mixlab` on arm64 Homebrew) before declaring the release done. If it fails, patch the source for compatibility, force-update the tag, refresh the GitHub release notes, and re-verify.
- **Cloud Build can fail Step 13 with `libcuda.so.1 not found`.** The CUDA driver lib is runtime-provided by NVIDIA Container Toolkit on the GPU host, not present at Cloud Build time. The Dockerfile's `ldd` check excludes it; if you add new ldd-sensitive logic, preserve the `grep -qv 'libcuda\.so\.1'` filter.
- **GitHub CI doesn't have nvcc.** It builds the binary without CUDA kernels (empty registry) and skips MLX-tagged tests (`CGO_ENABLED=0`). CI green only confirms the Go/C++ wiring compiles. CUDA kernel correctness has to be verified by smoke-testing on RunPod after the new image lands. After Cloud Build, autonomously update the RunPod template image SHA (see `docker/README.md`) and cycle workersMax to force fresh worker pulls.

## Verifying a CUDA kernel on RunPod

A "did not crash" smoke test is not enough: most CUDA kernels have a host-side MLX
fallback, so a kernel that is subtly wrong (e.g. reversed conv taps) still returns
finite, plausible-looking numbers. Verify **differentially** — run the same workload
twice from identical weights and seed, once on the CUDA path and once with the
kernel's `MIXLAB_*_DISABLE_CUDA_PRIMITIVE` env var set to force the fallback, and
require the outputs to match. Use the job's `env` field to toggle the fallback:

```jsonc
{"input": {
  "mode": "arch", "config": "/examples/ttt_mlp_tiny.json",
  "train": "/data/example/train_*.bin", "safetensors": "/tmp/w.st",
  "post": [
    "mixlab -mode generate -config $MIXLAB_CONFIG -safetensors-load /tmp/w.st -max-tokens 16",
    "MIXLAB_TTT_MLP_DISABLE_CUDA_PRIMITIVE=1 mixlab -mode generate -config $MIXLAB_CONFIG -safetensors-load /tmp/w.st -max-tokens 16"
  ]
}}
```

Confirm the CUDA path was actually live before trusting a match — if the primitive is
unavailable, *both* runs take the fallback and match trivially. `mixlab -mode smoke`
prints the MLX device (expect a GPU, not CPU), which is what gates
`*_cuda_primitive_available()` on Linux.
