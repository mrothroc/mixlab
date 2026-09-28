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

### 3. GitHub Release

When every Release job has passed, confirm the draft carries all five assets, then
write the notes and publish it:

```bash
gh release view vX.Y.Z --json assets --jq '.assets[].name'
# mixlab-vX.Y.Z-macos-arm64.dmg, SHA256SUMS-macos-arm64,
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

### 6. Downstream (if applicable)

- Update `go.mod` in mixlab-jazz: `go get github.com/mrothroc/mixlab@vX.Y.Z`
- Rebuild and push RunPod Docker image if GPU-side changes

For CLI/container packaging changes, rebuild both app and RunPod images. The
app build must pass its non-root embedded preparation check, not just the
optional GPU smoke. Supply `_RELEASE_VERSION=vX.Y.Z` and
`_SOURCE_REVISION=<release commit>` for manual Cloud Build submissions.

**There is no tag trigger.** The only trigger, `build-mixlab-images`, fires on
`push.branch: ^main$`, so `${TAG_NAME:-dev}` always resolves to `dev` for
automatic builds. Every release image must be submitted by hand:

```bash
gcloud builds submit --region=us-central1 \
  --project=zapbox-cloud --account=michael.rothrock@gmail.com \
  --config=docker/cloudbuild-ci.yaml \
  --substitutions=_REGISTRY_PREFIX=us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf,\
_MLX_BASE_IMAGE=us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf/golf-mlx-cuda:latest,\
_DOCKERHUB_USER=michaelrothrock,_RELEASE_VERSION=vX.Y.Z,_SOURCE_REVISION=<release commit>
```

**Submit it last, after every branch build has drained.** The build publishes
only the mutable tags `latest` and `runpod`, so whichever build finishes last
owns those tags. Every push to main starts a branch build stamped
`version=dev`; if one lands after the release build, the release labels are
silently replaced. This happened when each release pushed a formula commit to
main. Releases no longer do, but any other push to main has the same effect.
Wait for the queue to empty first:

```bash
gcloud builds list --ongoing --region=us-central1 \
  --project=zapbox-cloud --account=michael.rothrock@gmail.com
```

`--region=us-central1` is not optional. Omitting it queries the global region,
which reports `Listed 0 items.` while builds are running in us-central1 — it
reads as a drained queue at exactly the moment the check matters.

Because both published tags are mutable, the labels on `latest` are only
correct until the next push to main. This is not hypothetical: v0.117.0's
labels survived two days before the next feature push re-stamped `latest` with
`version=dev`.

So after verifying the build, tag the release digests immutably. This is a
registry-side operation; it copies nothing and rebuilds nothing:

```bash
REG=us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf
gcloud artifacts docker tags add $REG/mixlab:latest $REG/mixlab:vX.Y.Z \
  --project=zapbox-cloud --account=michael.rothrock@gmail.com
gcloud artifacts docker tags add $REG/mixlab:runpod $REG/mixlab:vX.Y.Z-runpod \
  --project=zapbox-cloud --account=michael.rothrock@gmail.com
```

Pin RunPod templates and any reproducibility-sensitive consumer to `vX.Y.Z-runpod`
or to the digest, never to `runpod`. Record the image **digest** as well; it is
the one identifier nothing can move. Docker Hub still publishes only the mutable
tags. Inspect the published image's OCI
version/revision labels and digest before updating the RunPod endpoint. See
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
  builds both images, then pushes to Docker Hub as its last step; the `images:` block
  that publishes to Artifact Registry only runs after *every* step succeeds. So an
  expired `dockerhub-token` fails the build after the images are already built, and AR
  silently keeps serving the previous digest. This went unnoticed for 14 builds across
  four releases. If `gcloud builds list` shows repeated FAILUREs, check step 2 for
  `denied: requested access to the resource is denied` before suspecting the code, and
  rotate the secret with `gcloud secrets versions add dockerhub-token`. To publish to AR
  alone while that is broken, submit the build manually with an empty `_DOCKERHUB_USER`,
  which the config already treats as "skip the Docker Hub step".
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
