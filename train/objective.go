package train

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

type objectiveBatch struct {
	x               []int
	y               []int
	lossMask        []float32
	attentionCausal []int32
	segmentIDs      []int32
	maskedLossMask  []float32
	teacherProbs    []float32
	unmaskedX       []int
	data2vecTargets []float32
	data2vecMask    []float32
}

func objectiveForStep(spec TrainingSpec, step int) string {
	spec.ApplyDefaults()
	switch spec.EffectiveObjective() {
	case arch.ObjectiveMLM:
		return arch.ObjectiveMLM
	case arch.ObjectiveMNTP:
		return arch.ObjectiveMNTP
	case arch.ObjectiveHybrid:
		if spec.EffectiveHybridMixGranularity() == arch.HybridMixGranularityExample {
			if spec.HybridCLMFraction >= 1 {
				return arch.ObjectiveCausal
			}
			if spec.HybridCLMFraction <= 0 {
				return spec.EffectiveHybridSecondaryObjective()
			}
			return arch.ObjectiveHybridExample
		}
		rng := deterministicObjectiveRNG(spec.Seed, step, 0x9e3779b97f4a7c15)
		if rng.Float64() < spec.HybridCLMFraction {
			return arch.ObjectiveCausal
		}
		return spec.EffectiveHybridSecondaryObjective()
	default:
		return arch.ObjectiveCausal
	}
}

func prepareObjectiveBatch(cfg *ArchConfig, batch trainBatch, step int, objective string) (objectiveBatch, error) {
	return prepareObjectiveBatchWithSeqLen(cfg, batch, step, objective, cfg.SeqLen)
}

func prepareObjectiveBatchWithSeqLen(cfg *ArchConfig, batch trainBatch, step int, objective string, seqLen int) (objectiveBatch, error) {
	if cfg == nil {
		return objectiveBatch{}, fmt.Errorf("nil config")
	}
	if batch.err != nil {
		return objectiveBatch{}, batch.err
	}
	need := cfg.Training.BatchTokens
	if need <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid batch_tokens=%d", need)
	}
	if len(batch.x) < need || len(batch.y) < need {
		return objectiveBatch{}, fmt.Errorf("input size mismatch: tokens=%d targets=%d need=%d", len(batch.x), len(batch.y), need)
	}
	var prepared objectiveBatch
	var err error
	switch canonicalObjective(objective) {
	case arch.ObjectiveMLM:
		prepared, err = prepareMLMBatch(cfg, batch, step, need)
	case arch.ObjectiveMNTP:
		prepared, err = prepareMNTPBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveHybridExample:
		prepared, err = prepareHybridExampleBatch(cfg, batch, step, need, seqLen)
	default:
		prepared = objectiveBatch{x: batch.x, y: batch.y, unmaskedX: batch.x[:need]}
	}
	if err != nil {
		return objectiveBatch{}, err
	}
	if cfg.Training.AttentionSegmentMaskEnabled() {
		if err := attachSegmentIDs(cfg, &prepared, need, seqLen); err != nil {
			return objectiveBatch{}, err
		}
	}
	return prepared, nil
}

func canonicalObjective(objective string) string {
	switch strings.ToLower(strings.TrimSpace(objective)) {
	case arch.ObjectiveMLM:
		return arch.ObjectiveMLM
	case arch.ObjectiveMNTP:
		return arch.ObjectiveMNTP
	case arch.ObjectiveHybridExample:
		return arch.ObjectiveHybridExample
	case arch.ObjectiveHybrid:
		return arch.ObjectiveHybrid
	default:
		return arch.ObjectiveCausal
	}
}

func prepareMLMBatch(cfg *ArchConfig, batch trainBatch, step, need int) (objectiveBatch, error) {
	if cfg.VocabSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid vocab_size=%d", cfg.VocabSize)
	}
	x := append([]int(nil), batch.x[:need]...)
	y := append([]int(nil), batch.x[:need]...)
	unmasked := append([]int(nil), batch.x[:need]...)
	lossMask := make([]float32, need)
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0x243f6a8885a308d3)
	maskProb := cfg.Training.EffectiveMLMMaskProbForStep(step)
	selected := selectMaskPositions(rng, lossMask, maskProb, need)
	if selected == 0 && maskProb > 0 && need > 0 {
		pos := rng.Intn(need)
		lossMask[pos] = 1
		selected = 1
	}
	if selected > 0 {
		replaceSelectedMLMTokens(cfg, rng, x, y, lossMask)
	}
	return objectiveBatch{x: x, y: y, lossMask: lossMask, unmaskedX: unmasked}, nil
}

func prepareMNTPBatch(cfg *ArchConfig, batch trainBatch, step, need, seqLen int) (objectiveBatch, error) {
	if seqLen <= 1 {
		return objectiveBatch{}, fmt.Errorf("mntp objective requires seq_len > 1")
	}
	if need%seqLen != 0 {
		return objectiveBatch{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", need, seqLen)
	}
	x := append([]int(nil), batch.x[:need]...)
	y := append([]int(nil), batch.y[:need]...)
	unmasked := append([]int(nil), batch.x[:need]...)
	lossMask := make([]float32, need)
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0x13198a2e03707344)
	maskProb := cfg.Training.EffectiveMLMMaskProbForStep(step)
	selected := 0
	eligible := 0
	for i := 0; i < need; i++ {
		if i%seqLen == seqLen-1 {
			continue
		}
		eligible++
		if rng.Float64() >= maskProb {
			continue
		}
		lossMask[i] = 1
		x[i+1] = cfg.Training.MLMMaskTokenID
		selected++
	}
	if selected == 0 && maskProb > 0 && eligible > 0 {
		slot := rng.Intn(eligible)
		for i := 0; i < need; i++ {
			if i%seqLen == seqLen-1 {
				continue
			}
			if slot == 0 {
				lossMask[i] = 1
				x[i+1] = cfg.Training.MLMMaskTokenID
				break
			}
			slot--
		}
	}
	return objectiveBatch{x: x, y: y, lossMask: lossMask, unmaskedX: unmasked}, nil
}

func prepareHybridExampleBatch(cfg *ArchConfig, batch trainBatch, step, need, seqLen int) (objectiveBatch, error) {
	if cfg.VocabSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid vocab_size=%d", cfg.VocabSize)
	}
	if seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid seq_len=%d", seqLen)
	}
	if cfg.Training.EffectiveHybridSecondaryObjective() == arch.ObjectiveMNTP && seqLen <= 1 {
		return objectiveBatch{}, fmt.Errorf("mntp objective requires seq_len > 1")
	}
	if need%seqLen != 0 {
		return objectiveBatch{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", need, seqLen)
	}
	batchSize := need / seqLen
	x := append([]int(nil), batch.x[:need]...)
	y := append([]int(nil), batch.y[:need]...)
	unmasked := append([]int(nil), batch.x[:need]...)
	lossMask := make([]float32, need)
	maskedLossMask := make([]float32, need)
	attentionCausal := make([]int32, batchSize)
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0xd1b54a32d192ed03)
	maskProb := cfg.Training.EffectiveMLMMaskProbForStep(step)
	secondary := cfg.Training.EffectiveHybridSecondaryObjective()
	for b := 0; b < batchSize; b++ {
		start := b * seqLen
		end := start + seqLen
		if rng.Float64() < cfg.Training.HybridCLMFraction {
			attentionCausal[b] = 1
			for i := start; i < end; i++ {
				lossMask[i] = 1
			}
			continue
		}
		switch secondary {
		case arch.ObjectiveMLM:
			for i := start; i < end; i++ {
				y[i] = batch.x[i]
			}
			selected := selectMaskPositions(rng, maskedLossMask[start:end], maskProb, seqLen)
			if selected == 0 && maskProb > 0 {
				pos := start + rng.Intn(seqLen)
				maskedLossMask[pos] = 1
				selected = 1
			}
			if selected > 0 {
				copy(lossMask[start:end], maskedLossMask[start:end])
				replaceSelectedMLMTokens(cfg, rng, x[start:end], y[start:end], maskedLossMask[start:end])
			}
		case arch.ObjectiveMNTP:
			selected := 0
			eligible := seqLen - 1
			for pos := 0; pos < seqLen-1; pos++ {
				if rng.Float64() >= maskProb {
					continue
				}
				i := start + pos
				lossMask[i] = 1
				maskedLossMask[i] = 1
				x[i+1] = cfg.Training.MLMMaskTokenID
				selected++
			}
			if selected == 0 && maskProb > 0 && eligible > 0 {
				i := start + rng.Intn(eligible)
				lossMask[i] = 1
				maskedLossMask[i] = 1
				x[i+1] = cfg.Training.MLMMaskTokenID
			}
		default:
			return objectiveBatch{}, fmt.Errorf("unsupported hybrid secondary objective %q", secondary)
		}
	}
	return objectiveBatch{
		x:               x,
		y:               y,
		lossMask:        lossMask,
		attentionCausal: attentionCausal,
		maskedLossMask:  maskedLossMask,
		unmaskedX:       unmasked,
	}, nil
}

func selectMaskPositions(rng *rand.Rand, lossMask []float32, prob float64, n int) int {
	selected := 0
	for i := 0; i < n; i++ {
		if rng.Float64() < prob {
			lossMask[i] = 1
			selected++
		}
	}
	return selected
}

func replaceSelectedMLMTokens(cfg *ArchConfig, rng *rand.Rand, x, originals []int, lossMask []float32) {
	maskProb := cfg.Training.MLMMaskTokenProb
	randomProb := cfg.Training.MLMRandomTokenProb
	for i, active := range lossMask {
		if active <= 0 {
			continue
		}
		r := rng.Float64()
		switch {
		case r < maskProb:
			x[i] = cfg.Training.MLMMaskTokenID
		case r < maskProb+randomProb:
			x[i] = rng.Intn(cfg.VocabSize)
		default:
			x[i] = originals[i]
		}
	}
}

func attachSegmentIDs(cfg *ArchConfig, batch *objectiveBatch, need, seqLen int) error {
	if batch == nil {
		return fmt.Errorf("nil objective batch")
	}
	if seqLen <= 0 || need%seqLen != 0 {
		return fmt.Errorf("invalid segment id batch shape: tokens=%d seq_len=%d", need, seqLen)
	}
	source := batch.unmaskedX
	if len(source) < need {
		source = batch.x
	}
	if len(source) < need {
		return fmt.Errorf("segment id source size=%d need=%d", len(source), need)
	}
	batch.segmentIDs = deriveSegmentIDs(source[:need], need, seqLen, cfg.Training.AttentionSegmentBoundaryTokenID)
	return nil
}

func deriveSegmentIDs(tokens []int, need, seqLen, boundary int) []int32 {
	ids := make([]int32, need)
	for rowStart := 0; rowStart < need; rowStart += seqLen {
		segment := int32(0)
		for pos := 0; pos < seqLen; pos++ {
			i := rowStart + pos
			if tokens[i] == boundary {
				segment++
			}
			ids[i] = segment
		}
	}
	return ids
}

func deterministicObjectiveRNG(seed int64, step int, salt uint64) *rand.Rand {
	mixed := splitmix64(uint64(seed) ^ uint64(step+1)*0x9e3779b97f4a7c15 ^ salt)
	return rand.New(rand.NewSource(int64(mixed)))
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
