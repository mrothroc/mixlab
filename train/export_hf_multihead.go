package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
)

// Multihead HF export: extract the configured export head plus the shared trunk
// and materialize a standard single-objective Hugging Face weight set.

func multiheadExportHeadSpec(cfg *ArchConfig) (arch.MultiheadHeadSpec, bool) {
	if cfg == nil || !cfg.Training.MultiheadEnabled() {
		return arch.MultiheadHeadSpec{}, false
	}
	name := cfg.Training.ExportHead
	for _, head := range cfg.Training.Heads {
		if head.Name == name {
			return head, true
		}
	}
	return arch.MultiheadHeadSpec{}, false
}

func materializeMultiheadHFExportWeights(exportCfg, sourceCfg *ArchConfig, sourceShapes []WeightShape, sourceWeights [][]float32) ([]WeightShape, [][]float32, error) {
	head, ok := multiheadExportHeadSpec(sourceCfg)
	if !ok {
		return nil, nil, fmt.Errorf("HF export requires a configured multihead export_head")
	}
	if head.Objective == arch.ObjectiveBlockDiffusion {
		return nil, nil, unsupportedHFExport("training.export_head", "block_diffusion heads are native-only")
	}
	if head.LayerAggregation == arch.LayerAggregationDWA {
		return nil, nil, unsupportedHFExport("training.export_head.layer_aggregation", "multihead DWA is a head-level aggregation and is not represented in the current HF template")
	}
	if !head.FinalNorm {
		return nil, nil, unsupportedHFExport("training.export_head.final_norm", "HF export requires the scorer head final norm in v1")
	}
	exportShapes, err := computeWeightShapes(exportCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("compute export weight shapes: %w", err)
	}
	sourceQueues := make(map[string][]int, len(sourceShapes))
	for i, shape := range sourceShapes {
		sourceQueues[shape.Name] = append(sourceQueues[shape.Name], i)
	}
	take := func(name string, want []int) ([]float32, error) {
		queue := sourceQueues[name]
		for len(queue) > 0 {
			idx := queue[0]
			queue = queue[1:]
			sourceQueues[name] = queue
			if intSlicesEqual(sourceShapes[idx].Shape, want) {
				return sourceWeights[idx], nil
			}
		}
		return nil, fmt.Errorf("HF export requires source multihead weight %q shape=%v", name, want)
	}
	outWeights := make([][]float32, 0, len(exportShapes)+1)
	for _, shape := range exportShapes {
		sourceName := shape.Name
		switch shape.Name {
		case "final_norm":
			sourceName = "head_" + head.Name + "_final_norm_scale"
		case "final_norm_scale":
			sourceName = "head_" + head.Name + "_final_norm_scale"
		case "final_norm_bias":
			sourceName = "head_" + head.Name + "_final_norm_bias"
		case arch.MLMHeadDenseWeightName:
			sourceName = "head_" + head.Name + "_mlm_dense"
		case arch.MLMHeadDenseBiasName:
			sourceName = "head_" + head.Name + "_mlm_dense_bias"
		case arch.MLMHeadOutputBiasName:
			sourceName = "head_" + head.Name + "_mlm_output_bias"
		case "head":
			sourceName = "head_" + head.Name + "_proj"
		}
		weight, err := take(sourceName, shape.Shape)
		if err != nil {
			return nil, nil, err
		}
		outWeights = append(outWeights, append([]float32(nil), weight...))
	}
	outShapes := append([]WeightShape(nil), exportShapes...)
	if !hasWeightShapeName(outShapes, "head") {
		embedIdx := weightShapeIndex(outShapes, "embed")
		if embedIdx < 0 {
			return nil, nil, fmt.Errorf("HF export requires base weight %q", "embed")
		}
		embedShape := outShapes[embedIdx].Shape
		if len(embedShape) != 2 || embedShape[0] != exportCfg.VocabSize || embedShape[1] != exportCfg.ModelDim {
			return nil, nil, fmt.Errorf("embed shape=%v does not match vocab/model dims [%d,%d]", embedShape, exportCfg.VocabSize, exportCfg.ModelDim)
		}
		outShapes = append(outShapes, WeightShape{Name: "head", Shape: []int{exportCfg.ModelDim, exportCfg.VocabSize}})
		outWeights = append(outWeights, transposeEmbeddingToHead(outWeights[embedIdx], exportCfg.VocabSize, exportCfg.ModelDim))
	}
	return outShapes, outWeights, nil
}

func intSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
