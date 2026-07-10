//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"log"

	"github.com/mrothroc/mixlab/gpu"
)

func buildTrainerOptimizerSpec(cfg *ArchConfig, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
	if cfg == nil {
		return gpu.TrainerOptimizerSpec{}, fmt.Errorf("nil config")
	}
	if cfg.TieEmbeddings && !cfg.MTPUntieEnabled() && cfg.Training.HeadLR != cfg.Training.EmbedLR {
		log.Printf("warning: tie_embeddings=true ignores head_lr; using embed_lr for shared embedding/head weight")
	}
	muonNesterov := true
	if cfg.Training.MuonNesterov != nil {
		muonNesterov = *cfg.Training.MuonNesterov
	}
	matrixOptimizerName := matrixOptimizer(cfg)
	embedOptimizerName := "adamw"
	headOptimizerName := "adamw"
	scalarOptimizerName := "adamw"
	embedBeta1 := cfg.Training.Beta1
	embedBeta2 := cfg.Training.Beta2
	embedEps := cfg.Training.Epsilon
	headBeta1 := cfg.Training.Beta1
	headBeta2 := cfg.Training.Beta2
	headEps := cfg.Training.Epsilon
	scalarBeta1 := cfg.Training.Beta1
	scalarBeta2 := cfg.Training.Beta2
	scalarEps := cfg.Training.Epsilon
	matrixBeta1 := cfg.Training.MuonMomentum
	matrixBeta2 := cfg.Training.Beta2
	matrixEps := cfg.Training.Epsilon
	if cfg.Training.Optimizer == "lamb" {
		embedOptimizerName = "lamb"
		headOptimizerName = "lamb"
		scalarOptimizerName = "lamb"
		matrixOptimizerName = "lamb"
		embedBeta1 = cfg.Training.LAMBBeta1
		embedBeta2 = cfg.Training.LAMBBeta2
		embedEps = cfg.Training.LAMBEps
		headBeta1 = cfg.Training.LAMBBeta1
		headBeta2 = cfg.Training.LAMBBeta2
		headEps = cfg.Training.LAMBEps
		scalarBeta1 = cfg.Training.LAMBBeta1
		scalarBeta2 = cfg.Training.LAMBBeta2
		scalarEps = cfg.Training.LAMBEps
		matrixBeta1 = cfg.Training.LAMBBeta1
		matrixBeta2 = cfg.Training.LAMBBeta2
		matrixEps = cfg.Training.LAMBEps
	}
	cautiousWeightDecay := cfg.Training.CautiousWeightDecay
	cautiousWeightDecayActivationStep := cfg.Training.EffectiveCautiousWeightDecayActivationStep()
	wmeta := make([]gpu.OptimizerWeightMetadata, len(shapes))
	for i, s := range shapes {
		wmeta[i] = gpu.OptimizerWeightMetadata{
			Name:        s.Name,
			Shape:       s.Shape,
			IsNormScale: s.IsNormScale,
		}
	}
	return gpu.BuildTrainerOptimizerSpec(gpu.TrainerOptimizerConfig{
		Weights: wmeta,
		Embed: gpu.OptimizerSettings{
			Name:                              embedOptimizerName,
			LR:                                cfg.Training.EmbedLR,
			Beta1:                             embedBeta1,
			Beta2:                             embedBeta2,
			Epsilon:                           embedEps,
			WeightDecay:                       cfg.Training.EmbedWeightDecay,
			LAMBTrustRatioCap:                 cfg.Training.LAMBTrustRatioCap,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
		},
		Head: gpu.OptimizerSettings{
			Name:                              headOptimizerName,
			LR:                                cfg.Training.HeadLR,
			Beta1:                             headBeta1,
			Beta2:                             headBeta2,
			Epsilon:                           headEps,
			WeightDecay:                       cfg.Training.HeadWeightDecay,
			LAMBTrustRatioCap:                 cfg.Training.LAMBTrustRatioCap,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
		},
		Scalar: gpu.OptimizerSettings{
			Name:                              scalarOptimizerName,
			LR:                                cfg.Training.ScalarLR,
			Beta1:                             scalarBeta1,
			Beta2:                             scalarBeta2,
			Epsilon:                           scalarEps,
			WeightDecay:                       cfg.Training.ScalarWeightDecay,
			LAMBTrustRatioCap:                 cfg.Training.LAMBTrustRatioCap,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
		},
		Matrix: gpu.OptimizerSettings{
			Name:                              matrixOptimizerName,
			LR:                                cfg.Training.MatrixLR,
			Beta1:                             matrixBeta1,
			Beta2:                             matrixBeta2,
			Epsilon:                           matrixEps,
			WeightDecay:                       cfg.Training.MatrixWeightDecay,
			LAMBTrustRatioCap:                 cfg.Training.LAMBTrustRatioCap,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
			BackendSteps:                      cfg.Training.MuonBackendSteps,
			NewtonSchulzVariant:               cfg.Training.NewtonSchulzVariant,
			Nesterov:                          muonNesterov,
			MuonNormalization:                 matrixMuonNormalization(matrixOptimizerName),
			RowNormalize:                      matrixOptimizerName == "muon_eq_r",
		},
		MaxGradNorm:   cfg.Training.GradClip,
		DefaultBaseLR: float32(cfg.Training.LR),
	})
}

func matrixOptimizer(cfg *ArchConfig) string {
	switch cfg.Training.Optimizer {
	case "adamw":
		return "adamw"
	case "muon_eq_r":
		return "muon_eq_r"
	case "normuon":
		return "normuon"
	case "lamb":
		return "lamb"
	default:
		return "muon"
	}
}

func matrixMuonNormalization(name string) gpu.MuonNormalization {
	switch name {
	case "muon_eq_r":
		return gpu.MuonNormalizationRowL2
	case "normuon":
		return gpu.MuonNormalizationNorMuon
	default:
		return gpu.MuonNormalizationNone
	}
}
