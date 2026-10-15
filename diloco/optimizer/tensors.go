// Package optimizer implements provider-independent, float32 DiLoCo arithmetic.
// It has no allocation, credential, checkpoint, training or storage dependency.
package optimizer

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

const MaxElements = 1 << 28 // Matches the current 1 GiB float32 artifact budget.

// Tensor order is the program/checkpoint weight-layout order, not arrival order
// or map iteration order. Adapters must preserve it when decoding safetensors.
type Tensor struct {
	Name   string    `json:"name"`
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}

func Validate(ts []Tensor) error {
	if len(ts) == 0 || len(ts) > 1<<20 {
		return fmt.Errorf("invalid tensor count")
	}
	seen := make(map[string]bool, len(ts))
	total := 0
	for _, t := range ts {
		if len(t.Name) == 0 || len(t.Name) > 256 || strings.IndexFunc(t.Name, func(r rune) bool { return r < 33 || r > 126 }) >= 0 || seen[t.Name] {
			return fmt.Errorf("invalid/duplicate tensor name %q", t.Name)
		}
		seen[t.Name] = true
		if len(t.Shape) == 0 || len(t.Shape) > 8 {
			return fmt.Errorf("tensor %s: invalid rank", t.Name)
		}
		n := 1
		for _, d := range t.Shape {
			if d <= 0 || d > MaxElements/n {
				return fmt.Errorf("tensor %s: invalid/oversized shape", t.Name)
			}
			n *= d
		}
		if n != len(t.Values) || n > MaxElements-total {
			return fmt.Errorf("tensor %s: length/byte budget mismatch", t.Name)
		}
		total += n
		for _, v := range t.Values {
			if !finite(float64(v)) {
				return fmt.Errorf("tensor %s: non-finite value", t.Name)
			}
		}
	}
	return nil
}

func ValidateMatch(reference, actual []Tensor) error {
	if e := Validate(reference); e != nil {
		return e
	}
	if e := Validate(actual); e != nil {
		return e
	}
	if len(reference) != len(actual) {
		return fmt.Errorf("tensor count mismatch")
	}
	for i, t := range reference {
		if t.Name != actual[i].Name || !slices.Equal(t.Shape, actual[i].Shape) {
			return fmt.Errorf("tensor %d: canonical name/order/shape mismatch", i)
		}
	}
	return nil
}

func ZerosLike(ts []Tensor) ([]Tensor, error) {
	if e := Validate(ts); e != nil {
		return nil, e
	}
	return allocate(ts), nil
}

func Delta(base, local []Tensor) ([]Tensor, error) {
	if e := ValidateMatch(base, local); e != nil {
		return nil, e
	}
	out := allocate(base)
	for i, t := range base {
		for j, v := range t.Values {
			d, e := checked(float64(v) - float64(local[i].Values[j]))
			if e != nil {
				return nil, e
			}
			out[i].Values[j] = d
		}
	}
	return out, nil
}

func allocate(ts []Tensor) []Tensor {
	out := make([]Tensor, len(ts))
	for i, t := range ts {
		out[i] = Tensor{Name: t.Name, Shape: slices.Clone(t.Shape), Values: make([]float32, len(t.Values))}
	}
	return out
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func checked(v float64) (float32, error) {
	if !finite(v) || math.Abs(v) > math.MaxFloat32 {
		return 0, fmt.Errorf("DiLoCo arithmetic overflow/non-finite result")
	}
	return float32(v), nil
}
