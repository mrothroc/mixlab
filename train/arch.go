package train

import (
	"github.com/mrothroc/mixlab/arch"
)

type ArchConfig = arch.ArchConfig
type BlockSpec = arch.BlockSpec
type CustomOpSpec = arch.CustomOpSpec
type CustomWeightSpec = arch.CustomWeightSpec
type EvalSpec = arch.EvalSpec
type OpSpec = arch.OpSpec
type Program = arch.Program
type TrainingPhase = arch.TrainingPhase
type TrainingSpec = arch.TrainingSpec
type WeightSpec = arch.WeightSpec

var DefaultTrainingSpec = arch.DefaultTrainingSpec
var DefaultEvalSpec = arch.DefaultEvalSpec
var DefaultLegalChunkSGDEvalSpec = arch.DefaultLegalChunkSGDEvalSpec
var LoadArchConfig = arch.LoadArchConfig
var ParseArchConfig = arch.ParseArchConfig

func BuildIRProgramFromConfig(cfg *ArchConfig) (*arch.Program, error) {
	return arch.BuildIRProgramFromConfig(cfg)
}

func CountIRWeightsFromConfig(cfg *ArchConfig) (int, error) {
	return arch.CountIRWeightsFromConfig(cfg)
}

func effectiveShuffleChunkTokens(cfg *ArchConfig) int {
	if cfg.Training.ShuffleChunkTokens > 0 {
		return cfg.Training.ShuffleChunkTokens
	}
	return cfg.SeqLen
}
