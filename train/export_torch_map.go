package train

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
)

type torchStateMapping struct {
	Source    string `json:"source"`
	Target    string `json:"target"`
	Transform string `json:"transform"`
	Axes      []int  `json:"axes,omitempty"`
	Shape     []int  `json:"shape"`
}

type torchStateExclusion struct {
	Source string `json:"source"`
	Reason string `json:"reason"`
}

type torchStateMap struct {
	Format     string                `json:"format"`
	Provenance map[string]string     `json:"provenance,omitempty"`
	Mappings   []torchStateMapping   `json:"mappings"`
	Excluded   []torchStateExclusion `json:"excluded,omitempty"`
}

func loadTorchStateMap(path string) (torchStateMap, error) {
	var m torchStateMap
	f, err := os.Open(path)
	if err != nil {
		return m, err
	}
	defer func() { _ = f.Close() }()
	info, err := f.Stat()
	if err != nil {
		return m, err
	}
	if info.Size() > 16<<20 {
		return m, fmt.Errorf("export map exceeds 16 MiB")
	}
	d := json.NewDecoder(io.LimitReader(f, 16<<20))
	d.DisallowUnknownFields()
	if err = d.Decode(&m); err != nil {
		return m, fmt.Errorf("export map: %w", err)
	}
	if err = d.Decode(new(any)); err != io.EOF {
		return m, fmt.Errorf("export map must contain one JSON object")
	}
	if m.Format != "mixlab.torch_state_map.v1" {
		return m, fmt.Errorf("unsupported export map format %q", m.Format)
	}
	return m, nil
}

func mapTorchState(m torchStateMap, shapes []WeightShape, weights [][]float32) ([]namedFloatTensor, error) {
	if len(shapes) != len(weights) {
		return nil, fmt.Errorf("weight inventory mismatch")
	}
	indices := map[string]int{}
	for i, s := range shapes {
		indices[s.Name] = i
	}
	seen, targets := map[string]bool{}, map[string]bool{}
	claim := func(name string) (int, error) {
		i, ok := indices[name]
		if !ok || seen[name] {
			return 0, fmt.Errorf("unknown or duplicate mapped source %q", name)
		}
		seen[name] = true
		return i, nil
	}
	var out []namedFloatTensor
	for _, entry := range m.Mappings {
		i, err := claim(entry.Source)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(entry.Target) == "" || entry.Target == "__metadata__" || targets[entry.Target] {
			return nil, fmt.Errorf("invalid or duplicate target %q", entry.Target)
		}
		targets[entry.Target] = true
		shape := shapes[i].Shape
		axes := make([]int, len(shape))
		for j := range axes {
			axes[j] = j
		}
		switch entry.Transform {
		case "identity":
			if len(entry.Axes) != 0 {
				return nil, fmt.Errorf("identity cannot specify axes: %s", entry.Source)
			}
		case "transpose":
			if len(entry.Axes) != len(shape) {
				return nil, fmt.Errorf("transpose rank mismatch: %s", entry.Source)
			}
			copy(axes, entry.Axes)
			seenAxes := make([]bool, len(shape))
			for _, a := range axes {
				if a < 0 || a >= len(shape) || seenAxes[a] {
					return nil, fmt.Errorf("invalid axis permutation: %s", entry.Source)
				}
				seenAxes[a] = true
			}
		default:
			return nil, fmt.Errorf("unsupported transform %q", entry.Transform)
		}
		resultShape := make([]int, len(shape))
		for j, a := range axes {
			resultShape[j] = shape[a]
		}
		if !slices.Equal(resultShape, entry.Shape) {
			return nil, fmt.Errorf("target shape mismatch: %s: got %v want %v", entry.Target, resultShape, entry.Shape)
		}
		values := weights[i]
		if len(values) != shapeProduct(shape) {
			return nil, fmt.Errorf("source payload mismatch: %s", entry.Source)
		}
		if entry.Transform == "transpose" {
			values = permuteTorchState(values, shape, axes, resultShape)
		}
		out = append(out, namedFloatTensor{Name: entry.Target, Shape: resultShape, Data: values})
	}
	for _, exclusion := range m.Excluded {
		if strings.TrimSpace(exclusion.Reason) == "" {
			return nil, fmt.Errorf("excluded weight requires reason: %s", exclusion.Source)
		}
		if _, err := claim(exclusion.Source); err != nil {
			return nil, err
		}
	}
	for _, s := range shapes {
		if !seen[s.Name] {
			return nil, fmt.Errorf("unmapped model weight %q (map or explicitly exclude with reason)", s.Name)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("export map contains no tensors")
	}
	return out, nil
}

func permuteTorchState(values []float32, shape, axes, resultShape []int) []float32 {
	strides := make([]int, len(shape))
	stride := 1
	for j := len(shape) - 1; j >= 0; j-- {
		strides[j], stride = stride, stride*shape[j]
	}
	out := make([]float32, len(values))
	for index := range out {
		remainder, source := index, 0
		for j := len(axes) - 1; j >= 0; j-- {
			source += (remainder % resultShape[j]) * strides[axes[j]]
			remainder /= resultShape[j]
		}
		out[index] = values[source]
	}
	return out
}
