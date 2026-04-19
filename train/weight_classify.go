package train

import "github.com/mrothroc/mixlab/gpu"

func optimizerWeightMetadata(shapes []WeightShape) []gpu.OptimizerWeightMetadata {
	weights := make([]gpu.OptimizerWeightMetadata, len(shapes))
	for i, shape := range shapes {
		weights[i] = gpu.OptimizerWeightMetadata{
			Name:        shape.Name,
			Shape:       shape.Shape,
			IsNormScale: shape.IsNormScale,
		}
	}
	return weights
}
