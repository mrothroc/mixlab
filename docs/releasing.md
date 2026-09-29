# Releasing a new version of mixlab

Pushing a `v*` tag starts the **Release** workflow
(`.github/workflows/release.yml`). After you approve its `release` environment, it
builds the signed, notarized macOS disk image and the Linux `mixlab-cluster`
tarballs and attaches them to a draft release. It never publishes anything; step 3
below does. Publishing a release then lets the Homebrew tap bump itself (step 4).
RunPod container images still come from Cloud Build (step 6). For private, locally signed
candidates, see [macOS distribution](macos-distribution.md).

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
git push origin vX.Y.Z
```

Push the one tag, not `git push --tags`: every `v*` tag starts a signing run, so a
stray local tag would too. The `release tags` ruleset stops `v*` tags from being
moved or deleted except by repository admins.

The push starts the Release workflow. Its macOS job waits for your approval of the
`release` environment, which holds the signing and notarization secrets. Approve it
from the run page:

```bash
gh run list --workflow release.yml --limit 1
gh run watch <run-id>
```

A tag with a suffix, such as `vX.Y.Z-rc.1`, produces a draft **prerelease**, which
is how to rehearse the workflow without cutting a release.

The workflow refuses to start when `go.mod`'s `toolchain` line names a Go release that
is out of support, and refuses to attach any binary in which govulncheck finds a
reachable known vulnerability. The `Security` workflow runs the same two checks on
every push, pull request and weekly, so a Go release going out of support fails the
build before a release is attempted. To fix either, move the `toolchain` line to a
supported release; `TestGoToolchainHasOneSource` lists every file that must follow.

### 3. GitHub Release

When every Release job has passed, confirm the draft carries all six assets, then
write the notes and publish it:

```bash
gh release view vX.Y.Z --json assets --jq '.assets[].name'
# mixlab-vX.Y.Z-macos-arm64.dmg, SHA256SUMS-macos-arm64, mixlab-signed.rb,
# mixlab-cluster-vX.Y.Z-linux-amd64.tar.gz, mixlab-cluster-vX.Y.Z-linux-arm64.tar.gz,
# SHA256SUMS-linux

gh release edit vX.Y.Z --draft=false --title "vX.Y.Z: Title" --notes "$(cat <<'EOF'
### Feature Name

Description.

### Other Changes

- **fix**: ...
- **refactor**: ...

### Install

\`\`\`bash
brew install mrothroc/tap/mixlab
\`\`\`

Or build from source:

\`\`\`bash
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab/
\`\`\`
EOF
)"
```

The generated `mixlab-signed.rb` is a reviewed input to the tap, not an automatic
tap update. After publishing the matching DMG, verify its checksum against the
cask, publish it under `Casks/mixlab-signed.rb` in the tap, and test installation
without signature changes. Keep the source-built formula as a separate channel.
Do not publish R1.2 as accepted until the [service acceptance gates](cluster-services.md#release-acceptance)
have passed, including signed upgrade/Keychain and login/reboot recovery.

### 4. Homebrew

The formula lives in `mrothroc/homebrew-tap`, which maintains itself with Homebrew's
standard workflows. This repository holds no Homebrew credentials.

1. **Bump.** The tap's autobump workflow checks daily for a new published release and
   opens a version-bump pull request. The formula's livecheck follows GitHub's latest
   release, so prereleases are never bumped. To bump now rather than wait:

   ```bash
   gh workflow run autobump.yml -R mrothroc/homebrew-tap
   ```

2. **Test.** `brew test-bot` runs on that pull request: `brew audit`, a from-source
   build, `brew test`, and bottles for macOS 15 and 26.
3. **Publish.** When the checks pass, publish it with its bottles:

   ```bash
   gh pr list -R mrothroc/homebrew-tap
   gh workflow run publish.yml -R mrothroc/homebrew-tap -f pull_request=<number>
   ```

   `brew pr-pull` uploads the bottles, adds them to the formula, and pushes to `main`,
   which closes the pull request. Do not merge it with GitHub's merge button: that
   ships the formula without bottles, so every user compiles.

Autobump opens its pull requests with the tap's `HOMEBREW_BUMP_TOKEN` secret, a
fine-grained token limited to that repository's contents and pull requests. Pull
requests opened with a workflow's default token do not trigger other workflows, so
without it the bump would arrive untested. The tap's weekly `bump-token-expiry` workflow
reads the token's expiry from GitHub and opens an issue there, with the rotation steps,
once 30 days or fewer remain; it closes the issue when it sees the replacement.

To change the formula itself, open a pull request against the tap; the same checks
run. If you widen `packaging/mlx-tested-range.txt` here, change `MLX_TESTED_MINIMUM` and
`MLX_TESTED_BELOW` in the formula in that release's bump pull request so they match.

The formula lives in the tap because of two past failures. A hand-copied tap formula
drifted for five months while an archived tap kept serving it. Later this repository
was made tappable itself, so machines that tapped both saw "Formulae found in multiple
taps" for a plain `brew info mixlab`. Keeping the one formula in the tap, maintained by
Homebrew's own tooling, removes both.

### 5. Verify

```bash
brew update && brew upgrade mrothroc/tap/mixlab   # the path README documents
brew info mrothroc/tap/mixlab                     # must report vX.Y.Z
mixlab -mode smoke
```

Upgrade by the full name, as above. Homebrew 7 loads formulae from a tap you have not
trusted only when the command names them in full, so a plain `brew upgrade` would skip
mixlab on a machine that has not run `brew trust mrothroc/tap`.

Verify that the installed binary can prepare data outside a source checkout:

```bash
tmpdir="$(mktemp -d)"
printf '>a\nACGT\n>b\nTGCA\n>c\nAAAA\n>d\nCCCC\n' > "$tmpdir/input.fasta"
(cd "$tmpdir" && mixlab -mode prepare \
  -input input.fasta -input-format fasta \
  -prepare-output-dir prepared -val-split 0.25)
```

### 6. Container images

The CLI and RunPod images are built by Cloud Build from `docker/cloudbuild-ci.yaml`.
The maintainer's operator runbook holds the exact commands, since they name private
infrastructure; this section is the part that holds for anyone running the build.

- **Rebuild both images** for CLI or container packaging changes, and for any GPU-side
  change. The app build must pass its non-root embedded preparation check, not just the
  optional GPU smoke.
- **There is no tag trigger.** Automatic builds run on pushes to main and stamp the
  images `version=dev`, so a release image is submitted by hand, from a worktree pinned
  at the tag, with `_RELEASE_VERSION=vX.Y.Z` and `_SOURCE_REVISION=<release commit>`.
- **Submit it last, after every branch build has drained.** The build publishes only the
  mutable tags `latest` and `runpod`, so whichever build finishes last owns them. If a
  branch build lands after the release build, the release labels are silently replaced.
  This happened when each release pushed a formula commit to main; releases no longer
  do, but any other push to main has the same effect. When checking the queue, name the
  build region: the global region reports no builds while regional ones are running.
- **Then tag the release digests immutably** as `vX.Y.Z` and `vX.Y.Z-runpod`. Even a
  release that wins the race keeps correct labels on `latest` only until the next push
  to main; v0.117.0's lasted two days.

Pin RunPod templates and any reproducibility-sensitive consumer to `vX.Y.Z-runpod` or to
the digest, never to `runpod`. The digest is the one identifier nothing can move. Docker
Hub publishes only the mutable tags. Inspect the published image's OCI version/revision
labels and digest before pointing an endpoint at it; see
[image provenance](../docker/README.md#image-provenance).

## Known gotchas

- **MLX API drift between Homebrew and the local MLX install.** Mixlab requires
  MLX 0.32.0 or newer. Homebrew may ship a newer version than the currently
  tested runtime. The tap's `brew test-bot` builds each bump pull request from source
  and runs `brew test` on macOS 15 and 26 before `brew pr-pull` can publish it.
  CUDA upgrades must rebuild `docker/base.Dockerfile`; rebuilding only the
  add-architecture or app layers retains the old MLX source. See
  `docs/mlx-0.32-upgrade.md` for the dependency and NCCL acceptance contract.
  If verification fails, fix the source and cut a patch release. Do not move the
  tag: the `release tags` ruleset blocks it for everyone but admins, and the signed
  assets and the tap formula already point at the original commit.
- **MLX 0.32.1 tightened gather VJP.** `take_along_axis` now raises
  `[gather_axis] Cannot calculate VJP with respect to indices` instead of
  silently ignoring the indices operand, which broke every MoE and bf16
  training path (the router's `argsort` indices trace back to the
  differentiable probabilities). Fixed by wrapping every `take_along_axis`
  index argument in `mx::stop_gradient`; indices are discrete, so this is
  numerically a no-op on 0.32.0 and restores 0.32.1. The formula in
  `mrothroc/homebrew-tap` declares `depends_on "mlx"` unpinned and checks the tested
  range only when mixlab is installed, so `brew upgrade mlx` under an installed
  mixlab still reaches users before any test. The main CI cannot see it either,
  because it builds with `CGO_ENABLED=0` and never links MLX. After any Homebrew MLX upgrade, run the
  `-tags mlx` suite locally before releasing.
- **Cloud Build can fail Step 13 with `libcuda.so.1 not found`.** The CUDA driver lib is runtime-provided by NVIDIA Container Toolkit on the GPU host, not present at Cloud Build time. The Dockerfile's `ldd` check excludes it; if you add new ldd-sensitive logic, preserve the `grep -qv 'libcuda\.so\.1'` filter.
- **A failed Docker Hub push also strands Artifact Registry.** `docker/cloudbuild-ci.yaml`
  pushes to Docker Hub as its last step, and the `images:` block that publishes to
  Artifact Registry runs only after *every* step succeeds. An expired Docker Hub token
  therefore fails the build after the images are built, and Artifact Registry silently
  keeps serving the previous digest. This went unnoticed for 14 builds across four
  releases. Repeated build failures with `denied: requested access to the resource is
  denied` mean the token, not the code. An empty `_DOCKERHUB_USER` skips the Docker Hub
  step, which publishes to Artifact Registry alone while the token is replaced.
- **GitHub CI doesn't have nvcc.** It builds the binary without CUDA kernels (empty registry) and skips MLX-tagged tests (`CGO_ENABLED=0`). CI green only confirms the Go/C++ wiring compiles. CUDA kernel correctness has to be verified on RunPod after the new image lands; see below.

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
