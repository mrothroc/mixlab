package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

func makeGridInputs(decls []arch.TensorDecl, b *data.GridBatch) ([]gpu.TensorInput, error) {
	if b == nil {
		return nil, fmt.Errorf("grid program requires a grid batch")
	}
	values := map[string][]float32{"grid": b.Inputs, "grid_targets": b.Targets, "grid_loss_mask": b.LossMask}
	inputs := make([]gpu.TensorInput, 0, len(decls))
	for _, d := range decls {
		v, ok := values[d.Name]
		if !ok || len(v) != shapeProduct(d.Shape) {
			return nil, fmt.Errorf("grid input %q: got %d values, expected shape %v", d.Name, len(v), d.Shape)
		}
		inputs = append(inputs, gpu.TensorInput{Name: d.Name, DType: gpu.TensorFloat32, Shape: d.Shape, Data: v})
	}
	return inputs, nil
}
