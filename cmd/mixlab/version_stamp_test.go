package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestContainerVersionStamp(t *testing.T) {
	docker, err := os.ReadFile("../../docker/app.Dockerfile")
	if err != nil {
		t.Fatal(err)
	}
	builder := strings.Split(string(docker), "# --- Runtime image ---")[0]
	for _, arg := range []string{"ARG MIXLAB_VERSION=", "ARG VCS_REF="} {
		if !strings.Contains(builder, arg) {
			t.Fatalf("build stage missing %s", arg)
		}
	}
	for _, field := range []string{"buildinfo.Version=${MIXLAB_VERSION}", "buildinfo.Revision=${VCS_REF}"} {
		if !strings.Contains(string(docker), field) {
			t.Fatalf("container missing %s", field)
		}
	}
	bin := filepath.Join(t.TempDir(), "mixlab")
	cmd := exec.Command("go", "build", "-buildvcs=false", "-ldflags", "-X github.com/mrothroc/mixlab/internal/buildinfo.Version=v9.8.7 -X github.com/mrothroc/mixlab/internal/buildinfo.Revision=0123456789abcdef", "-o", bin, ".")
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	out, err := exec.Command(bin, "-version").CombinedOutput()
	if err != nil || !strings.HasPrefix(string(out), "mixlab v9.8.7 (0123456789ab)\n") {
		t.Fatalf("version: %v %s", err, out)
	}
}
