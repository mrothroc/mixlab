package train

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPreparePythonVersionPreflight(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not found")
	}
	python, err = filepath.Abs(python)
	if err != nil {
		t.Fatal(err)
	}
	for _, minor := range []int{9, 10, 14} {
		t.Run(fmt.Sprintf("3.%d", minor), func(t *testing.T) {
			binDir := t.TempDir()
			// Run the actual preflight code with a simulated interpreter version.
			// Stub dependency imports so this test needs no installed packages.
			wrapper := fmt.Sprintf("#!%s\nimport sys\nsys.version_info = (3, %d, 0)\nsys.version = '3.%d.0'\nsys.modules['numpy'] = object()\nsys.modules['tokenizers'] = object()\nexec(sys.argv[2])\n", python, minor, minor)
			if err := os.WriteFile(filepath.Join(binDir, "python3"), []byte(wrapper), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", binDir)
			_, err := preparePython("text")
			if minor >= 10 {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "requires Python 3.10 or newer") ||
				!strings.Contains(err.Error(), "found 3.9.0") || strings.Contains(err.Error(), "pip install") {
				t.Fatalf("old interpreter error=%v", err)
			}
		})
	}
}

func TestPrepareContainerPackagingContract(t *testing.T) {
	read := func(name string) string {
		t.Helper()
		blob, err := os.ReadFile(filepath.Join("..", "docker", name))
		if err != nil {
			t.Fatal(err)
		}
		return string(blob)
	}
	app := read("app.Dockerfile")
	_, runtime, ok := strings.Cut(app, "FROM ${BASE_IMAGE} AS runtime")
	if !ok {
		t.Fatal("missing app runtime stage")
	}
	for _, contract := range []string{
		"python3-venv",
		"COPY requirements-prepare.txt /opt/mixlab/requirements-prepare.txt",
		"python3 -m venv /opt/mixlab/venv",
		`ENV PATH="/opt/mixlab/venv/bin:${PATH}"`,
		"python3 -m pip install --no-cache-dir -r /opt/mixlab/requirements-prepare.txt",
		`org.opencontainers.image.version="${MIXLAB_VERSION}"`,
		`org.opencontainers.image.revision="${VCS_REF}"`,
		"FROM runtime AS prepare-check",
		"USER 10001:10001",
		"RUN python3 /tmp/prepare_smoke.py /tmp/mixlab-prepare-check",
		"FROM runtime AS final\nCOPY --from=prepare-check",
	} {
		if !strings.Contains(runtime, contract) {
			t.Errorf("app runtime missing %q", contract)
		}
	}
	ci := read("cloudbuild-ci.yaml")
	for _, contract := range []string{
		"_RELEASE_VERSION: '${TAG_NAME:-dev}'", "_SOURCE_REVISION: '${COMMIT_SHA:-unknown}'",
		"MIXLAB_VERSION=${_RELEASE_VERSION}", "VCS_REF=${_SOURCE_REVISION}", "dynamicSubstitutions: true",
	} {
		if !strings.Contains(ci, contract) {
			t.Errorf("Cloud Build missing %q", contract)
		}
	}
	if !strings.Contains(read("runpod.Dockerfile"), "-c /opt/mixlab/requirements-prepare.txt") {
		t.Error("RunPod dependencies must respect inherited prepare constraints")
	}
}

func TestPrepareContainerSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("builds an installed CLI")
	}
	python, err := preparePython("text")
	if err != nil {
		t.Skipf("prepare dependencies unavailable: %v", err)
	}
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	binary := filepath.Join(dir, "mixlab")
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-o", binary, "./cmd/mixlab")
	build.Dir = root
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build installed CLI: %v\n%s", err, out)
	}
	cmd := exec.CommandContext(ctx, python, filepath.Join(root, "docker", "prepare_smoke.py"), binary)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("container prepare smoke: %v\n%s", err, out)
	}
}
