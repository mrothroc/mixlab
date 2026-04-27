package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	archpkg "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/train"
)

func main() {
	configPath := flag.String("config", "./experiments/gated_deltanet_test/gated_deltanet_compare.json", "path to mixlab config")
	trainPattern := flag.String("train", "data/example/train_*.bin", "training shard glob")
	flag.Parse()

	cfg, err := train.LoadArchConfig(*configPath)
	if err != nil {
		log.Fatalf("LoadArchConfig(%q): %v", *configPath, err)
	}

	err = train.RunArch(*configPath, *trainPattern, train.TrainOptions{
		OptimizerOverride: func(spec gpu.TrainerOptimizerSpec, shapes []train.WeightShape) (gpu.TrainerOptimizerSpec, error) {
			return forceGatedDeltaNetMatricesToAdamW(cfg, spec, shapes)
		},
	})
	if err != nil {
		log.Fatal(err)
	}
}

func forceGatedDeltaNetMatricesToAdamW(cfg *train.ArchConfig, spec gpu.TrainerOptimizerSpec, shapes []train.WeightShape) (gpu.TrainerOptimizerSpec, error) {
	if cfg == nil {
		return spec, fmt.Errorf("nil config")
	}

	blockCounts := make([]int, len(cfg.Blocks))
	totalBlockWeights := 0
	for i, block := range cfg.Blocks {
		n, err := archpkg.BlockWeightCount(block, cfg.BlockScales, cfg.ResidMix)
		if err != nil {
			return spec, fmt.Errorf("block %d (%s) weight count: %w", i, block.Type, err)
		}
		blockCounts[i] = n
		totalBlockWeights += n
	}

	if totalBlockWeights > len(shapes) {
		return spec, fmt.Errorf("block weights=%d exceed total shapes=%d", totalBlockWeights, len(shapes))
	}
	prefixWeights := len(shapes) - totalBlockWeights

	adamwGroupIndex := -1
	for i, group := range spec.Groups {
		if group.Kind == gpu.OptimizerAdamW &&
			group.LR == cfg.Training.MatrixLR &&
			group.Beta1 == cfg.Training.MuonMomentum &&
			group.Beta2 == cfg.Training.Beta2 &&
			group.Epsilon == cfg.Training.Epsilon &&
			group.WeightDecay == cfg.Training.MatrixWeightDecay {
			adamwGroupIndex = i
			break
		}
	}
	if adamwGroupIndex < 0 {
		spec.Groups = append(spec.Groups, gpu.OptimizerGroup{
			Kind:         gpu.OptimizerAdamW,
			LR:           cfg.Training.MatrixLR,
			Beta1:        cfg.Training.MuonMomentum,
			Beta2:        cfg.Training.Beta2,
			Epsilon:      cfg.Training.Epsilon,
			WeightDecay:  cfg.Training.MatrixWeightDecay,
			BackendSteps: 0,
			Nesterov:     false,
		})
		adamwGroupIndex = len(spec.Groups) - 1
	}

	fmt.Println("== GatedDeltaNet Optimizer Audit ==")
	fmt.Printf("config=%s matrix_lr=%g matrix_wd=%g matrix_default=%q\n",
		cfg.Name, cfg.Training.MatrixLR, cfg.Training.MatrixWeightDecay, matrixOptimizerName(cfg))

	shapeIdx := prefixWeights
	switched := 0
	for blockIdx, block := range cfg.Blocks {
		count := blockCounts[blockIdx]
		start := shapeIdx
		end := shapeIdx + count
		fmt.Printf("block[%d] type=%s weights=[%d,%d)\n", blockIdx, block.Type, start, end)
		for i := start; i < end; i++ {
			beforeGroup := spec.Groups[spec.Weights[i].GroupIndex]
			beforeName := optimizerKindName(beforeGroup.Kind)
			if strings.EqualFold(strings.TrimSpace(block.Type), "gated_deltanet") && len(shapes[i].Shape) == 2 {
				spec.Weights[i].GroupIndex = adamwGroupIndex
				switched++
			}
			afterGroup := spec.Groups[spec.Weights[i].GroupIndex]
			fmt.Printf("  w[%d] %-12s shape=%v %s -> %s\n",
				i, shapes[i].Name, shapes[i].Shape, beforeName, optimizerKindName(afterGroup.Kind))
		}
		shapeIdx = end
	}
	fmt.Printf("switched_gdn_matrix_weights=%d\n", switched)

	return spec, nil
}

func optimizerKindName(kind gpu.OptimizerKind) string {
	switch kind {
	case gpu.OptimizerAdamW:
		return "adamw"
	case gpu.OptimizerMuon:
		return "muon"
	default:
		return fmt.Sprintf("unknown(%d)", kind)
	}
}

func matrixOptimizerName(cfg *train.ArchConfig) string {
	if strings.EqualFold(strings.TrimSpace(cfg.Training.Optimizer), "adamw") {
		return "adamw"
	}
	return "muon"
}
