//go:build mlx && cgo && (darwin || linux)

package train

import "fmt"

func weightIndexByName(shapes []WeightShape, name string) (int, error) {
	for i, shape := range shapes {
		if shape.Name == name {
			return i, nil
		}
	}
	return -1, fmt.Errorf("unknown weight %q", name)
}
