package train

import "github.com/mrothroc/mixlab/arch"

func diffusionLogitsOutputName(cfg *ArchConfig) string {
	if cfg != nil && cfg.Training.MultiheadEnabled() && cfg.Training.DiffusionHead != "" {
		return "head_" + cfg.Training.DiffusionHead + "_logits"
	}
	return "logits"
}

func expandBatchForMultiheadDiffusion(cfg *ArchConfig, batch objectiveBatch, rawBatchSize, seqLen int) objectiveBatch {
	if cfg == nil || !cfg.Training.MultiheadEnabled() {
		return batch
	}
	headCount := len(cfg.Training.Heads)
	if headCount == 0 || rawBatchSize <= 0 || seqLen <= 0 {
		return batch
	}
	rawNeed := rawBatchSize * seqLen
	totalNeed := rawNeed * headCount
	out := objectiveBatch{
		x:                   make([]int, totalNeed),
		y:                   make([]int, totalNeed),
		lossMask:            make([]float32, totalNeed),
		unmaskedX:           make([]int, totalNeed),
		diffusionBlockStart: make([]int32, rawBatchSize*headCount),
		diffusionBlockEnd:   make([]int32, rawBatchSize*headCount),
		diffusionTimestep:   make([]float32, rawBatchSize*headCount),
		batchSizeOverride:   rawBatchSize * headCount,
	}
	for headIdx, head := range cfg.Training.Heads {
		tokenOffset := headIdx * rawNeed
		rowOffset := headIdx * rawBatchSize
		copy(out.x[tokenOffset:tokenOffset+rawNeed], batch.x[:rawNeed])
		copy(out.y[tokenOffset:tokenOffset+rawNeed], batch.y[:rawNeed])
		if len(batch.unmaskedX) >= rawNeed {
			copy(out.unmaskedX[tokenOffset:tokenOffset+rawNeed], batch.unmaskedX[:rawNeed])
		} else {
			copy(out.unmaskedX[tokenOffset:tokenOffset+rawNeed], batch.x[:rawNeed])
		}
		copyDiffusionUtilityLossMask(out.lossMask[tokenOffset:tokenOffset+rawNeed], batch.lossMask, head.Objective)
		switch head.Objective {
		case arch.ObjectiveBlockDiffusion:
			if len(batch.diffusionBlockStart) >= rawBatchSize {
				copy(out.diffusionBlockStart[rowOffset:rowOffset+rawBatchSize], batch.diffusionBlockStart[:rawBatchSize])
			}
			if len(batch.diffusionBlockEnd) >= rawBatchSize {
				copy(out.diffusionBlockEnd[rowOffset:rowOffset+rawBatchSize], batch.diffusionBlockEnd[:rawBatchSize])
			}
			if len(batch.diffusionTimestep) >= rawBatchSize {
				copy(out.diffusionTimestep[rowOffset:rowOffset+rawBatchSize], batch.diffusionTimestep[:rawBatchSize])
			}
		case arch.ObjectiveMLM, arch.ObjectiveMNTP:
			for row := 0; row < rawBatchSize; row++ {
				out.diffusionBlockStart[rowOffset+row] = 0
				out.diffusionBlockEnd[rowOffset+row] = int32(seqLen)
			}
		default:
			for row := 0; row < rawBatchSize; row++ {
				out.diffusionBlockStart[rowOffset+row] = 0
				out.diffusionBlockEnd[rowOffset+row] = 0
			}
		}
	}
	return out
}

func copyDiffusionUtilityLossMask(dst []float32, src []float32, objective string) {
	switch objective {
	case arch.ObjectiveBlockDiffusion, arch.ObjectiveMLM, arch.ObjectiveMNTP:
		if len(src) >= len(dst) {
			copy(dst, src[:len(dst)])
			return
		}
	}
	for i := range dst {
		dst[i] = 1
	}
}

func countPositiveMask(mask []float32) int {
	n := 0
	for _, v := range mask {
		if v > 0 {
			n++
		}
	}
	return n
}
