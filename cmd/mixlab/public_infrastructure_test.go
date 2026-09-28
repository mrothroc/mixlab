package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// This repository is public. The maintainer's Google Cloud project, Artifact
// Registry and gcloud account belong in the private operator runbook, not here:
// until 2026-09-28 docs/releasing.md and the base-image Cloud Build configs
// named all three. Examples use my-project, and build configs take the registry
// as a substitution. The patterns below match the shape of a real identifier,
// so this test names none of them.
var privateInfrastructurePatterns = []struct {
	pattern *regexp.Regexp
	allowed map[string]bool
	what    string
}{
	{regexp.MustCompile(`[a-z0-9-]+-docker\.pkg\.dev/([A-Za-z0-9_.-]+)`), map[string]bool{"my-project": true}, "an Artifact Registry project"},
	{regexp.MustCompile(`--project[= ]([A-Za-z0-9_.-]+)`), map[string]bool{"my-project": true}, "a gcloud project"},
	{regexp.MustCompile(`--account[= ]([^\s'"\\]+@[^\s'"\\]+)`), nil, "a gcloud account"},
}

func TestPublicRepositoryNamesNoPrivateInfrastructure(t *testing.T) {
	root := filepath.Join("..", "..")
	// Tracked files only: ignored local files such as .env legitimately hold
	// real identifiers and are never published.
	listing, err := exec.Command("git", "-C", root, "ls-files", "-z").Output()
	if err != nil {
		t.Skipf("cannot list tracked files (no git checkout?): %v", err)
	}
	scanned := 0
	for _, relative := range strings.Split(strings.TrimRight(string(listing), "\x00"), "\x00") {
		name := filepath.Base(relative)
		switch strings.ToLower(filepath.Ext(name)) {
		case ".md", ".go", ".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".txt":
		default:
			if !strings.HasSuffix(name, "Dockerfile") && name != "Makefile" {
				continue
			}
		}
		data, err := os.ReadFile(filepath.Join(root, relative))
		if err != nil {
			t.Fatalf("read %s: %v", relative, err)
		}
		scanned++
		for _, check := range privateInfrastructurePatterns {
			for _, match := range check.pattern.FindAllStringSubmatch(string(data), -1) {
				if !check.allowed[match[1]] {
					t.Errorf("%s names %s (%q); use my-project or a substitution, and keep real identifiers in the private runbook", relative, check.what, match[0])
				}
			}
		}
	}
	// An empty listing would pass vacuously.
	if scanned < 100 {
		t.Fatalf("scanned only %d tracked files; the listing is not covering the repository", scanned)
	}
}
