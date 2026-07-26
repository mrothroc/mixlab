package train

import "fmt"

func materializeHFExportWeights(cfg, sourceCfg *ArchConfig, shapes []WeightShape, weights [][]float32) ([]WeightShape, [][]float32, error) {
	if cfg == nil {
		return nil, nil, fmt.Errorf("nil config")
	}
	if len(shapes) != len(weights) {
		return nil, nil, fmt.Errorf("weight shape/data count mismatch: shapes=%d weights=%d", len(shapes), len(weights))
	}
	if sourceCfg != nil && sourceCfg.Training.MultiheadEnabled() {
		return materializeMultiheadHFExportWeights(cfg, sourceCfg, shapes, weights)
	}
	if cfg.ClassificationEnabled() {
		return materializeNativeClassificationHFExportWeights(cfg, shapes, weights)
	}
	if hasWeightShapeName(shapes, "head") {
		return shapes, weights, nil
	}
	if !cfg.TieEmbeddings {
		return nil, nil, fmt.Errorf("HF export requires head weight for untied embeddings")
	}
	embedIdx := weightShapeIndex(shapes, "embed")
	if embedIdx < 0 {
		return nil, nil, fmt.Errorf("HF export requires base weight %q", "embed")
	}
	embedShape := shapes[embedIdx].Shape
	if len(embedShape) != 2 || embedShape[0] != cfg.VocabSize || embedShape[1] != cfg.ModelDim {
		return nil, nil, fmt.Errorf("embed shape=%v does not match vocab/model dims [%d,%d]", embedShape, cfg.VocabSize, cfg.ModelDim)
	}
	head := transposeEmbeddingToHead(weights[embedIdx], cfg.VocabSize, cfg.ModelDim)
	outShapes := append([]WeightShape(nil), shapes...)
	outWeights := append([][]float32(nil), weights...)
	outShapes = append(outShapes, WeightShape{Name: "head", Shape: []int{cfg.ModelDim, cfg.VocabSize}})
	outWeights = append(outWeights, head)
	return outShapes, outWeights, nil
}

func materializeNativeClassificationHFExportWeights(cfg *ArchConfig, shapes []WeightShape, weights [][]float32) ([]WeightShape, [][]float32, error) {
	if cfg == nil || cfg.Training.Classification == nil {
		return nil, nil, fmt.Errorf("native classification HF export requires classification config")
	}
	projIdx := weightShapeIndex(shapes, "head_classifier_proj")
	biasIdx := weightShapeIndex(shapes, "head_classifier_bias")
	if projIdx < 0 || biasIdx < 0 {
		return nil, nil, fmt.Errorf("native classification HF export requires classifier projection and bias weights")
	}
	dim := cfg.ModelDim
	labels := cfg.Training.Classification.NumLabels
	if !equalIntShape(shapes[projIdx].Shape, []int{dim, labels}) {
		return nil, nil, fmt.Errorf("classifier projection shape=%v, want [%d,%d]", shapes[projIdx].Shape, dim, labels)
	}
	if !equalIntShape(shapes[biasIdx].Shape, []int{labels}) {
		return nil, nil, fmt.Errorf("classifier bias shape=%v, want [%d]", shapes[biasIdx].Shape, labels)
	}
	outShapes := append([]WeightShape(nil), shapes...)
	outWeights := append([][]float32(nil), weights...)
	outShapes[projIdx].Shape = []int{labels, dim}
	outWeights[projIdx] = transposeMatrix(weights[projIdx], dim, labels)
	if !hasWeightShapeName(outShapes, "head") {
		if !cfg.TieEmbeddings {
			return nil, nil, fmt.Errorf("native classification HF export requires head weight for untied embeddings")
		}
		embedIdx := weightShapeIndex(outShapes, "embed")
		if embedIdx < 0 || !equalIntShape(outShapes[embedIdx].Shape, []int{cfg.VocabSize, dim}) {
			return nil, nil, fmt.Errorf("native classification HF export requires embed shape [%d,%d]", cfg.VocabSize, dim)
		}
		outShapes = append(outShapes, WeightShape{Name: "head", Shape: []int{dim, cfg.VocabSize}})
		outWeights = append(outWeights, transposeEmbeddingToHead(outWeights[embedIdx], cfg.VocabSize, dim))
	}
	return outShapes, outWeights, nil
}

func transposeMatrix(values []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			out[col*rows+row] = values[row*cols+col]
		}
	}
	return out
}

func equalIntShape(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func transposeEmbeddingToHead(embed []float32, vocab, dim int) []float32 {
	head := make([]float32, dim*vocab)
	for v := 0; v < vocab; v++ {
		for d := 0; d < dim; d++ {
			head[d*vocab+v] = embed[v*dim+d]
		}
	}
	return head
}

func hasWeightShapeName(shapes []WeightShape, name string) bool {
	return weightShapeIndex(shapes, name) >= 0
}

func weightShapeIndex(shapes []WeightShape, name string) int {
	for i, shape := range shapes {
		if shape.Name == name {
			return i
		}
	}
	return -1
}
