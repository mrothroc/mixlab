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
