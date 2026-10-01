package arch

import "fmt"

func customWeightShapes(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
	heads := spec.Heads
	if heads <= 0 {
		heads = 1
	}
	metas := make([]WeightMeta, len(spec.Weights))
	for j, w := range spec.Weights {
		shape, err := resolveShapeSymbol(w.Shape, D, heads, T, B, V)
		if err != nil {
			return nil, fmt.Errorf("custom block %q weight %q: %w", spec.Name, w.Name, err)
		}
		metas[j] = WeightMeta{Name: w.Name, Shape: shape}
		if err = gridWeightInit(w, &metas[j], nil); err != nil {
			return nil, err
		}
	}
	return metas, nil
}
