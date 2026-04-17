//go:build !mlx || !cgo || (!darwin && !linux)

package train

func mlxAvailable() bool { return false }
