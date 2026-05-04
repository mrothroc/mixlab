package arch

import (
	"fmt"
	"strings"
)

func validateEvalSpec(cfg *ArchConfig, source string) error {
	if cfg.Eval == nil {
		return nil
	}

	eval := cfg.Eval
	eval.TTTMode = strings.ToLower(strings.TrimSpace(eval.TTTMode))
	if eval.TTTMode == "" {
		eval.TTTMode = "none"
	}
	switch eval.TTTMode {
	case "none":
		return nil
	case "legal_chunk_sgd":
	default:
		return fmt.Errorf("config %q has invalid eval.ttt_mode=%q (must be \"none\" or \"legal_chunk_sgd\")", source, eval.TTTMode)
	}

	d := DefaultLegalChunkSGDEvalSpec()
	if eval.ChunkTokens == 0 {
		eval.ChunkTokens = d.ChunkTokens
	}
	if eval.TTTEpochs == 0 {
		eval.TTTEpochs = d.TTTEpochs
	}
	if eval.TTTLR == 0 {
		eval.TTTLR = d.TTTLR
	}
	if eval.TTTMomentum == nil {
		eval.TTTMomentum = d.TTTMomentum
	}
	eval.TTTLRSchedule = strings.ToLower(strings.TrimSpace(eval.TTTLRSchedule))
	if eval.TTTLRSchedule == "" {
		eval.TTTLRSchedule = d.TTTLRSchedule
	}

	if eval.ChunkTokens <= 0 {
		return fmt.Errorf("config %q has invalid eval.chunk_tokens=%d (must be > 0)", source, eval.ChunkTokens)
	}
	if eval.ChunkTokens < cfg.Training.BatchTokens {
		return fmt.Errorf("config %q has eval.chunk_tokens=%d smaller than training.batch_tokens=%d", source, eval.ChunkTokens, cfg.Training.BatchTokens)
	}
	if eval.ChunkTokens%cfg.Training.BatchTokens != 0 {
		return fmt.Errorf("config %q has eval.chunk_tokens=%d not divisible by training.batch_tokens=%d", source, eval.ChunkTokens, cfg.Training.BatchTokens)
	}
	if eval.TTTEpochs <= 0 {
		return fmt.Errorf("config %q has invalid eval.ttt_epochs=%d (must be > 0)", source, eval.TTTEpochs)
	}
	if eval.TTTLR <= 0 {
		return fmt.Errorf("config %q has invalid eval.ttt_lr=%g (must be > 0)", source, eval.TTTLR)
	}
	momentum := eval.EffectiveTTTMomentum()
	if momentum < 0 || momentum >= 1 {
		return fmt.Errorf("config %q has invalid eval.ttt_momentum=%g (must be in [0,1))", source, momentum)
	}
	switch eval.TTTLRSchedule {
	case "cosine", "constant":
	default:
		return fmt.Errorf("config %q has invalid eval.ttt_lr_schedule=%q (must be \"cosine\" or \"constant\")", source, eval.TTTLRSchedule)
	}
	return nil
}
