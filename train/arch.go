package train

import (
	"github.com/mrothroc/mixlab/arch"
)

type ArchConfig = arch.ArchConfig
type BlockSpec = arch.BlockSpec
type CustomOpSpec = arch.CustomOpSpec
type CustomWeightSpec = arch.CustomWeightSpec
type OpSpec = arch.OpSpec
type TrainingSpec = arch.TrainingSpec
type WeightSpec = arch.WeightSpec

var DefaultTrainingSpec = arch.DefaultTrainingSpec
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
