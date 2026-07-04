package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
)

func prepareMultiheadBatch(cfg *ArchConfig, batch trainBatch, step, need, seqLen int) (objectiveBatch, error) {
	if cfg == nil {
		return objectiveBatch{}, fmt.Errorf("nil config")
	}
	if !cfg.Training.MultiheadEnabled() {
		return objectiveBatch{}, fmt.Errorf("prepareMultiheadBatch requires training.objective=%q", arch.ObjectiveMultihead)
	}
	if need%seqLen != 0 {
		return objectiveBatch{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", need, seqLen)
	}
	rawBatchSize := need / seqLen
	headCount := len(cfg.Training.Heads)
	if headCount == 0 {
		return objectiveBatch{}, fmt.Errorf("multihead objective has no heads")
	}
	totalRows := rawBatchSize * headCount
	if minimalPairUsesMLMSpanPLL(cfg) {
		totalRows += rawBatchSize
	}
	totalNeed := totalRows * seqLen
	out := objectiveBatch{
		x:                   make([]int, totalNeed),
		y:                   make([]int, totalNeed),
		lossMask:            make([]float32, totalNeed),
		unmaskedX:           make([]int, totalNeed),
		diffusionBlockStart: make([]int32, totalRows),
		diffusionBlockEnd:   make([]int32, totalRows),
		diffusionTimestep:   make([]float32, totalRows),
		batchSizeOverride:   totalRows,
	}
	if cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.UsesDifferingSpanEnergy() {
		out.energySpanMask = make([]float32, totalNeed)
	}
	for headIdx, head := range cfg.Training.Heads {
		headCfg := *cfg
		headCfg.Training.Objective = head.Objective
		headCfg.Training.Diffusion = head.Diffusion
		prepared, err := prepareSingleMultiheadView(&headCfg, batch, step, need, seqLen, head.Objective)
		if err != nil {
			return objectiveBatch{}, fmt.Errorf("prepare multihead head %q: %w", head.Name, err)
		}
		tokenOffset := headIdx * need
		rowOffset := headIdx * rawBatchSize
		copy(out.x[tokenOffset:tokenOffset+need], prepared.x[:need])
		copy(out.y[tokenOffset:tokenOffset+need], prepared.y[:need])
		if len(prepared.unmaskedX) >= need {
			copy(out.unmaskedX[tokenOffset:tokenOffset+need], prepared.unmaskedX[:need])
		} else {
			copy(out.unmaskedX[tokenOffset:tokenOffset+need], batch.x[:need])
		}
		fillHeadLossMask(out.lossMask[tokenOffset:tokenOffset+need], prepared.lossMask, head.Objective)
		fillHeadDiffusionBoundaries(out.diffusionBlockStart[rowOffset:rowOffset+rawBatchSize], out.diffusionBlockEnd[rowOffset:rowOffset+rawBatchSize], prepared, head.Objective, seqLen)
		fillHeadDiffusionTimestep(out.diffusionTimestep[rowOffset:rowOffset+rawBatchSize], prepared, head.Objective, seqLen)
	}
	return out, nil
}

func prepareSingleMultiheadView(cfg *ArchConfig, batch trainBatch, step, need, seqLen int, objective string) (objectiveBatch, error) {
	switch objective {
	case arch.ObjectiveMLM:
		return prepareMLMBatch(cfg, batch, step, need)
	case arch.ObjectiveMNTP:
		return prepareMNTPBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveBlockDiffusion:
		return prepareBlockDiffusionBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveRTD:
		return prepareRTDBatch(batch, need)
	case arch.ObjectiveEnergy:
		return prepareEnergyPlaceholderBatch(batch, need)
	case arch.ObjectiveCausal:
		return objectiveBatch{x: batch.x[:need], y: batch.y[:need], unmaskedX: batch.x[:need]}, nil
	default:
		return objectiveBatch{}, fmt.Errorf("unsupported multihead objective %q", objective)
	}
}

func fillHeadLossMask(dst []float32, src []float32, objective string) {
	if len(src) >= len(dst) {
		copy(dst, src[:len(dst)])
		return
	}
	for i := range dst {
		dst[i] = 1
	}
	if objective == arch.ObjectiveMNTP {
		return
	}
}

func fillHeadDiffusionBoundaries(starts, ends []int32, prepared objectiveBatch, objective string, seqLen int) {
	switch objective {
	case arch.ObjectiveBlockDiffusion:
		copy(starts, prepared.diffusionBlockStart[:len(starts)])
		copy(ends, prepared.diffusionBlockEnd[:len(ends)])
	case arch.ObjectiveMLM, arch.ObjectiveMNTP, arch.ObjectiveRTD, arch.ObjectiveEnergy:
		for i := range starts {
			starts[i] = 0
			ends[i] = int32(seqLen)
		}
	default:
		for i := range starts {
			starts[i] = 0
			ends[i] = 0
		}
	}
}

func prepareRTDBatch(batch trainBatch, need int) (objectiveBatch, error) {
	x := append([]int(nil), batch.x[:need]...)
	y := make([]int, need)
	lossMask := make([]float32, need)
	for i := 0; i < need; i++ {
		y[i] = 1
		lossMask[i] = 1
	}
	return objectiveBatch{x: x, y: y, lossMask: lossMask, unmaskedX: append([]int(nil), batch.x[:need]...)}, nil
}

func prepareEnergyPlaceholderBatch(batch trainBatch, need int) (objectiveBatch, error) {
	x := append([]int(nil), batch.x[:need]...)
	y := make([]int, need)
	lossMask := make([]float32, need)
	return objectiveBatch{x: x, y: y, lossMask: lossMask, unmaskedX: append([]int(nil), batch.x[:need]...)}, nil
}

func fillHeadDiffusionTimestep(dst []float32, prepared objectiveBatch, objective string, seqLen int) {
	if objective != arch.ObjectiveBlockDiffusion || len(prepared.lossMask) == 0 {
		clear(dst)
		return
	}
	for b := range dst {
		start := int(prepared.diffusionBlockStart[b])
		end := int(prepared.diffusionBlockEnd[b])
		if start < 0 || end <= start || end > seqLen {
			dst[b] = 0
			continue
		}
		rowStart := b * seqLen
		selected := 0
		for pos := start; pos < end; pos++ {
			if prepared.lossMask[rowStart+pos] > 0 {
				selected++
			}
		}
		dst[b] = float32(selected) / float32(end-start)
	}
}
