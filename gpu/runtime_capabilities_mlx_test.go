//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"runtime"
	"testing"
)

func TestRequiredDistributedBackendIsCompiled(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX GPU backend unavailable")
	}
	info := RuntimeInfo()
	if !info.VersionSupported {
		t.Fatalf("MLX runtime %q is older than required %s", info.Version, MinimumMLXVersion)
	}
	backend := "ring"
	if runtime.GOOS == "linux" {
		backend = "nccl"
	}
	if err := RequireDistributedBackend(backend); err != nil {
		t.Fatal(err)
	}
}
