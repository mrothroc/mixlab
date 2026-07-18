package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

// These helpers resolve the IR output names score-ebm reads for each scoring
// mode (native energy head vs MLM/MNTP span-PLL head).

func energyHeadLogitsOutputName(cfg *ArchConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("nil config")
	}
	for _, head := range cfg.Training.Heads {
		if head.Objective == arch.ObjectiveEnergy {
			return "head_" + head.Name + "_logits", nil
		}
	}
	return "", fmt.Errorf("score-ebm requires a multihead objective=%q head", arch.ObjectiveEnergy)
}

func energyHeadTokenOutputName(cfg *ArchConfig) string {
	if cfg == nil {
		return ""
	}
	for _, head := range cfg.Training.Heads {
		if head.Objective == arch.ObjectiveEnergy {
			return "head_" + head.Name + "_token_energy"
		}
	}
	return ""
}

func mlmSpanPLLScoreOutputName(cfg *ArchConfig) (string, error) {
	headName, err := mlmSpanPLLScoreHeadName(cfg)
	if err != nil {
		return "", err
	}
	return "head_" + headName + "_minimal_pair_scores", nil
}

func mlmSpanPLLScoreLogitsOutputName(cfg *ArchConfig) (string, error) {
	headName, err := mlmSpanPLLScoreHeadName(cfg)
	if err != nil {
		return "", err
	}
	return "head_" + headName + "_logits", nil
}

func mlmSpanPLLScoreHeadName(cfg *ArchConfig) (string, error) {
	if cfg == nil || cfg.Training.MinimalPair == nil || !cfg.Training.MinimalPair.UsesMLMSpanPLL() {
		return "", fmt.Errorf("score-ebm requires training.minimal_pair.score_source=%q or a native energy head", arch.MinimalPairScoreMLMPLL)
	}
	headName := strings.TrimSpace(cfg.Training.MinimalPair.ScoreHead)
	if headName == "" {
		headName = strings.TrimSpace(cfg.Training.ExportHead)
	}
	for _, head := range cfg.Training.Heads {
		if head.Name != headName {
			continue
		}
		if head.Objective != arch.ObjectiveMLM && head.Objective != arch.ObjectiveMNTP {
			return "", fmt.Errorf("score-ebm training.minimal_pair.score_head=%q must select an mlm or mntp head", headName)
		}
		return head.Name, nil
	}
	return "", fmt.Errorf("score-ebm training.minimal_pair.score_head=%q does not match any training head", headName)
}
