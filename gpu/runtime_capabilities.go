package gpu

import (
	"fmt"
	"strconv"
	"strings"
)

const MinimumMLXVersion = "0.32.0"

// MLXRuntimeInfo describes the linked MLX runtime without initializing a
// distributed process group.
type MLXRuntimeInfo struct {
	Available        bool
	Version          string
	VersionSupported bool
	Device           string
	Ring             bool
	NCCL             bool
}

// RuntimeInfo initializes the ordinary MLX device and reports dependency and
// distributed-backend capabilities. It never creates a distributed group.
func RuntimeInfo() MLXRuntimeInfo {
	available := Available()
	info := MLXRuntimeInfo{
		Available: available,
		Version:   mlxRuntimeVersion(),
		Ring:      mlxDistributedBackendAvailable("ring"),
		NCCL:      mlxDistributedBackendAvailable("nccl"),
	}
	info.VersionSupported = mlxVersionAtLeast(info.Version, MinimumMLXVersion)
	if available {
		info.Device = DeviceName()
	}
	return info
}

// RequireMinimumRuntime verifies the linked MLX library, not only the headers
// used at compile time.
func RequireMinimumRuntime() error {
	version := mlxRuntimeVersion()
	if !mlxVersionAtLeast(version, MinimumMLXVersion) {
		if version == "" {
			version = "unavailable"
		}
		return fmt.Errorf("MLX runtime %s is too old; mixlab requires MLX >= %s", version, MinimumMLXVersion)
	}
	return nil
}

// DistributedBackendAvailable reports whether the linked MLX library was
// built with the requested communication backend.
func DistributedBackendAvailable(backend string) bool {
	switch backend {
	case "ring", "nccl":
		return mlxDistributedBackendAvailable(backend)
	default:
		return false
	}
}

// RequireDistributedBackend returns an actionable error when an MLX build
// cannot support a requested future distributed-training backend.
func RequireDistributedBackend(backend string) error {
	if backend != "ring" && backend != "nccl" {
		return fmt.Errorf("unsupported MLX distributed backend %q (supported: ring, nccl)", backend)
	}
	if !Available() {
		return fmt.Errorf("MLX backend unavailable; rebuild mixlab with MLX >= %s and -tags mlx", MinimumMLXVersion)
	}
	if err := RequireMinimumRuntime(); err != nil {
		return err
	}
	if !mlxDistributedBackendAvailable(backend) {
		return fmt.Errorf(
			"MLX %s on %s lacks the %s distributed backend; install or rebuild MLX with %s support",
			mlxRuntimeVersion(), DeviceName(), backend, backend,
		)
	}
	return nil
}

func mlxVersionAtLeast(version, minimum string) bool {
	got, ok := parseMLXVersion(version)
	if !ok {
		return false
	}
	want, ok := parseMLXVersion(minimum)
	if !ok {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return got[i] > want[i]
		}
	}
	return true
}

func parseMLXVersion(version string) ([3]int, bool) {
	var parsed [3]int
	parts := strings.Split(version, ".")
	if len(parts) < len(parsed) {
		return parsed, false
	}
	for i := range parsed {
		digits := parts[i]
		for j, r := range digits {
			if r < '0' || r > '9' {
				digits = digits[:j]
				break
			}
		}
		if digits == "" {
			return parsed, false
		}
		value, err := strconv.Atoi(digits)
		if err != nil {
			return parsed, false
		}
		parsed[i] = value
	}
	return parsed, true
}
