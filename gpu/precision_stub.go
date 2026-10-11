//go:build !mlx || !cgo || (!darwin && !linux)

package gpu

import "os"

func PrecisionBackend() string { return "unavailable" }

func ConfigurePrecision(dtype string, requested *bool) (PrecisionPolicy, error) {
	return ResolvePrecisionPolicy(dtype, PrecisionBackend(), requested, os.LookupEnv)
}
