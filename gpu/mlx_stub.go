//go:build (!mlx && cgo && darwin) || (!mlx && cgo && linux)

// This file keeps the package CGO-enabled when the mlx build tag is absent so
// the C++ bridge sources can coexist with the shared Go stub implementation.
package gpu

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -std=c++20 -I.
#cgo darwin CFLAGS: -I/opt/homebrew/opt/mlx/include
#cgo darwin CXXFLAGS: -I/opt/homebrew/opt/mlx/include -I/opt/homebrew/opt/mlx/include/metal_cpp
#cgo darwin LDFLAGS: -L/opt/homebrew/opt/mlx/lib -Wl,-rpath,/opt/homebrew/opt/mlx/lib -lmlx -framework Metal -framework Foundation -framework Accelerate
*/
import "C"
