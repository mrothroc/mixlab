package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// go.mod's toolchain line is the one record of the Go that builds mixlab.
// Until 2026-09-28 the version was written in nine places and they drifted
// apart: go.mod said 1.24.0, CI and release builds resolved '1.24' to 1.24.13,
// and the container base downloaded 1.24.4. Go 1.24 had been out of support
// for seven months by then, and govulncheck found 16 reachable standard-library
// vulnerabilities in the release binaries and 28 in the container build.
func TestGoToolchainHasOneSource(t *testing.T) {
	root := filepath.Join("..", "..")
	read := func(relative string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, relative))
		if err != nil {
			t.Fatalf("read %s: %v", relative, err)
		}
		return string(data)
	}

	match := regexp.MustCompile(`(?m)^toolchain go(\d+\.\d+\.\d+)$`).FindStringSubmatch(read("go.mod"))
	if match == nil {
		t.Fatal("go.mod has no `toolchain goX.Y.Z` line; it is the single record of the build toolchain")
	}
	version := match[1]

	// The README's stated minimum is the go directive, the floor for importers.
	directive := regexp.MustCompile(`(?m)^go (\d+\.\d+)(?:\.\d+)?$`).FindStringSubmatch(read("go.mod"))
	if directive == nil {
		t.Fatal("go.mod has no go directive")
	}
	claims := regexp.MustCompile(`Go (\d+\.\d+)\+`).FindAllStringSubmatch(read("README.md"), -1)
	if len(claims) == 0 {
		t.Error("README.md no longer states the minimum Go version")
	}
	for _, claim := range claims {
		if claim[1] != directive[1] {
			t.Errorf("README.md says Go %s+ but go.mod requires go %s", claim[1], directive[1])
		}
	}

	// setup-go reads the toolchain directive only from v6 on; v5 silently
	// falls back to the go directive, the minimum for importers.
	workflows, err := filepath.Glob(filepath.Join(root, ".github", "workflows", "*.yml"))
	if err != nil || len(workflows) == 0 {
		t.Fatalf("no workflows found: %v", err)
	}
	setupGo := regexp.MustCompile(`uses: actions/setup-go@([0-9a-f]{40})? *(?:# *v(\d+))?`)
	for _, path := range workflows {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		name := filepath.Base(path)
		uses := setupGo.FindAllStringSubmatch(text, -1)
		for _, use := range uses {
			major, _ := strconv.Atoi(use[2])
			if use[1] == "" || major < 6 {
				t.Errorf("%s: %q must be pinned by commit SHA with a `# vN` comment, N >= 6", name, strings.TrimSpace(use[0]))
			}
		}
		if files := strings.Count(text, "go-version-file: go.mod"); files != len(uses) {
			t.Errorf("%s: %d setup-go steps but %d read go-version-file: go.mod", name, len(uses), files)
		}
		if strings.Contains(text, "go-version:") {
			t.Errorf("%s: go-version: names a version outside go.mod", name)
		}
	}

	// Local lint runs on the same toolchain as CI, read from go.mod.
	literal := regexp.MustCompile(`go1\.\d+`)
	for _, relative := range []string{"Makefile", filepath.Join(".githooks", "pre-commit")} {
		text := read(relative)
		if found := literal.FindString(text); found != "" {
			t.Errorf("%s names %s; derive the lint toolchain from go.mod", relative, found)
		}
		if !strings.Contains(text, "go.mod") {
			t.Errorf("%s does not read its lint toolchain from go.mod", relative)
		}
	}

	// A Dockerfile cannot read go.mod before it knows which archive to fetch,
	// and each archive needs its own pinned checksum, so the base image keeps a
	// copy that this test holds to go.mod.
	base := read(filepath.Join("docker", "base.Dockerfile"))
	if !strings.Contains(base, "ARG GO_VERSION="+version+"\n") {
		t.Errorf("docker/base.Dockerfile must declare ARG GO_VERSION=%s", version)
	}
	if !regexp.MustCompile(`ARG GO_SHA256=[0-9a-f]{64}\n`).MatchString(base) || !strings.Contains(base, "sha256sum -c") {
		t.Error("docker/base.Dockerfile must verify the Go archive against a pinned GO_SHA256")
	}
	if found := literal.FindString(base); found != "" {
		t.Errorf("docker/base.Dockerfile names %s outside GO_VERSION", found)
	}

	// Base-image tags carry the Go version, so a rebuild on a new toolchain
	// never replaces the image the build trigger is using.
	tag := regexp.MustCompile(`golf-mlx-cuda(?:-base)?:[A-Za-z0-9._-]+|_IMAGE_TAG: "[^"]*"`)
	for _, relative := range []string{filepath.Join("docker", "cloudbuild-mlx-cuda-base.yaml"), filepath.Join("docker", "cloudbuild-golf-mlx-cuda.yaml")} {
		for _, found := range tag.FindAllString(read(relative), -1) {
			if !strings.Contains(found, "-go"+version) {
				t.Errorf("%s: image tag %q does not carry -go%s", relative, found, version)
			}
		}
	}
}
