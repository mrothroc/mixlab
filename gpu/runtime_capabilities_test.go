package gpu

import (
	"strings"
	"testing"
)

func TestRequireDistributedBackendRejectsUnknownBackend(t *testing.T) {
	err := RequireDistributedBackend("auto")
	if err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("RequireDistributedBackend(auto) error = %v, want unsupported-backend error", err)
	}
}

func TestRuntimeInfoIsInternallyConsistent(t *testing.T) {
	info := RuntimeInfo()
	if info.Available && info.Version == "" {
		t.Fatal("available MLX runtime reported an empty version")
	}
	if info.Available && info.Device == "" {
		t.Fatal("available MLX runtime reported an empty device")
	}
	if !info.Available && info.Device != "" {
		t.Fatalf("unavailable MLX runtime reported device %q", info.Device)
	}
}

func TestMLXVersionAtLeast(t *testing.T) {
	tests := []struct {
		version string
		want    bool
	}{
		{version: "0.32.0", want: true},
		{version: "0.32.1", want: true},
		{version: "0.33.0", want: true},
		{version: "1.0.0", want: true},
		{version: "0.32.0.dev20260725+abc", want: true},
		{version: "0.31.2", want: false},
		{version: "0.25.2", want: false},
		{version: "", want: false},
		{version: "unknown", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.version, func(t *testing.T) {
			if got := mlxVersionAtLeast(tt.version, MinimumMLXVersion); got != tt.want {
				t.Fatalf("mlxVersionAtLeast(%q, %q) = %v, want %v", tt.version, MinimumMLXVersion, got, tt.want)
			}
		})
	}
}
