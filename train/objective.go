package train

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

type objectiveBatch struct {
	x                     []int
	y                     []int
	batchSizeOverride     int
	lossMask              []float32
	wordStructTargets     []int
	wordStructLossMask    []float32
	invarianceLossMask    []float32
	pllMarginLossMask     []float32
	energySpanMask        []float32
	attentionCausal       []int32
	segmentIDs            []int32
	maskedLossMask        []float32
	teacherProbs          []float32
	unmaskedX             []int
	data2vecTargets       []float32
	data2vecMask          []float32
	diffusionBlockStart   []int32
	diffusionBlockEnd     []int32
	diffusionTimestep     []float32
	rtdGeneratorX         []int
	rtdGeneratorPositions []int32
	rtdGeneratorY         []int
	rtdGeneratorLossMask  []float32
	tttInnerLRScale       []float32
	classificationLabels  []int32
	classificationMask    []float32
	classificationPos     []int32
	mlmMaskStats          mlmMaskStats
}

type mlmMaskStats struct {
	Unit                    string
	TargetProb              float64
	EligibleTokens          int
	BudgetTokens            int
	SelectedTokens          int
	CandidateGroups         int
	SelectedGroups          int
	SelectedGroupTokenTotal int
	MaxSelectedGroupSize    int
	UnderfilledRows         int
}

func (s *mlmMaskStats) add(other mlmMaskStats) {
	if s.Unit == "" {
		s.Unit = other.Unit
		s.TargetProb = other.TargetProb
	}
	s.EligibleTokens += other.EligibleTokens
	s.BudgetTokens += other.BudgetTokens
	s.SelectedTokens += other.SelectedTokens
	s.CandidateGroups += other.CandidateGroups
	s.SelectedGroups += other.SelectedGroups
	s.SelectedGroupTokenTotal += other.SelectedGroupTokenTotal
	if other.MaxSelectedGroupSize > s.MaxSelectedGroupSize {
		s.MaxSelectedGroupSize = other.MaxSelectedGroupSize
	}
	s.UnderfilledRows += other.UnderfilledRows
}

func objectiveForStep(spec TrainingSpec, step int) string {
	spec.ApplyDefaults()
	switch spec.EffectiveObjective() {
	case arch.ObjectiveMLM:
		return arch.ObjectiveMLM
	case arch.ObjectiveMNTP:
		return arch.ObjectiveMNTP
	case arch.ObjectiveBlockDiffusion:
		return arch.ObjectiveBlockDiffusion
	case arch.ObjectiveMultihead:
		return arch.ObjectiveMultihead
	case arch.ObjectiveClassification:
		return arch.ObjectiveClassification
	case arch.ObjectiveHybrid:
		causalFraction := spec.EffectiveHybridCLMFractionForStep(step)
		if spec.EffectiveHybridMixGranularity() == arch.HybridMixGranularityExample {
			if causalFraction >= 1 {
				return arch.ObjectiveCausal
			}
			if causalFraction <= 0 {
				return spec.EffectiveHybridSecondaryObjective()
			}
			return arch.ObjectiveHybridExample
		}
		rng := deterministicObjectiveRNG(spec.Seed, step, 0x9e3779b97f4a7c15)
		if rng.Float64() < causalFraction {
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
	var err error
	batch, err = maybeApplyReverseComplement(cfg, batch, step, need)
	if err != nil {
		return objectiveBatch{}, err
	}
	var prepared objectiveBatch
	switch canonicalObjective(objective) {
	case arch.ObjectiveMLM:
		prepared, err = prepareMLMBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveMNTP:
		prepared, err = prepareMNTPBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveBlockDiffusion:
		prepared, err = prepareBlockDiffusionBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveMultihead:
		prepared, err = prepareMultiheadBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveHybridExample:
		prepared, err = prepareHybridExampleBatch(cfg, batch, step, need, seqLen)
	case arch.ObjectiveClassification:
		prepared, err = prepareClassificationBatch(cfg, batch, need, seqLen)
	default:
		prepared = objectiveBatch{x: batch.x, y: batch.y, unmaskedX: batch.x[:need]}
		if len(batch.lossMask) >= need {
			prepared.lossMask = append([]float32(nil), batch.lossMask[:need]...)
		}
	}
	if err != nil {
		return objectiveBatch{}, err
	}
	if err := maybeApplyWordStructuralObjective(cfg, &prepared, step, canonicalObjective(objective), need, seqLen); err != nil {
		return objectiveBatch{}, err
	}
	if cfg.Training.ExampleFramingEnabled() && canonicalObjective(objective) == arch.ObjectiveCausal {
		prepared.lossMask = data.FramedCausalLossMask(need, seqLen)
	}
	if len(batch.segmentIDs) >= need {
		prepared.segmentIDs = append([]int32(nil), batch.segmentIDs[:need]...)
	}
	if cfg.Training.AttentionSegmentMaskEnabled() {
		if err := attachSegmentIDs(cfg, &prepared, need, seqLen); err != nil {
			return objectiveBatch{}, err
		}
	}
	prepared.tttInnerLRScale = arch.TTTMLPInnerLRScalesForStep(cfg.Blocks, step)
	return prepared, nil
}

func canonicalObjective(objective string) string {
	switch strings.ToLower(strings.TrimSpace(objective)) {
	case arch.ObjectiveMLM:
		return arch.ObjectiveMLM
	case arch.ObjectiveMNTP:
		return arch.ObjectiveMNTP
	case arch.ObjectiveBlockDiffusion:
		return arch.ObjectiveBlockDiffusion
	case arch.ObjectiveMultihead:
		return arch.ObjectiveMultihead
	case arch.ObjectiveHybridExample:
		return arch.ObjectiveHybridExample
	case arch.ObjectiveHybrid:
		return arch.ObjectiveHybrid
	case arch.ObjectiveClassification:
		return arch.ObjectiveClassification
	default:
		return arch.ObjectiveCausal
	}
}

func prepareClassificationBatch(cfg *ArchConfig, batch trainBatch, need, seqLen int) (objectiveBatch, error) {
	if cfg == nil || !cfg.ClassificationEnabled() {
		return objectiveBatch{}, fmt.Errorf("classification batch requires training.objective=%q", arch.ObjectiveClassification)
	}
	if seqLen <= 0 || need%seqLen != 0 {
		return objectiveBatch{}, fmt.Errorf("classification batch_tokens=%d must be divisible by seq_len=%d", need, seqLen)
	}
	batchSize := need / seqLen
	if len(batch.labels) < batchSize {
		return objectiveBatch{}, fmt.Errorf("classification batch has %d labels, need %d", len(batch.labels), batchSize)
	}
	if len(batch.validMask) < need {
		return objectiveBatch{}, fmt.Errorf("classification batch valid mask has %d entries, need %d", len(batch.validMask), need)
	}
	labels := append([]int32(nil), batch.labels[:batchSize]...)
	for row, label := range labels {
		if label < 0 || int(label) >= cfg.Training.Classification.NumLabels {
			return objectiveBatch{}, fmt.Errorf(
				"classification batch row %d label=%d outside [0,%d)",
				row, label, cfg.Training.Classification.NumLabels,
			)
		}
	}
	validMask := append([]float32(nil), batch.validMask[:need]...)
	positions := make([]int32, batchSize)
	for row := 0; row < batchSize; row++ {
		last := -1
		for pos := 0; pos < seqLen; pos++ {
			if validMask[row*seqLen+pos] > 0 {
				last = pos
			}
		}
		if last < 0 {
			return objectiveBatch{}, fmt.Errorf("classification batch row %d has no valid tokens", row)
		}
		positions[row] = int32(row*seqLen + last)
	}
	return objectiveBatch{
		x:                    batch.x[:need],
		y:                    batch.y[:need],
		unmaskedX:            batch.x[:need],
		classificationLabels: labels,
		classificationMask:   validMask,
		classificationPos:    positions,
	}, nil
}

func prepareMLMBatch(cfg *ArchConfig, batch trainBatch, step, need, seqLen int) (objectiveBatch, error) {
	if cfg.VocabSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid vocab_size=%d", cfg.VocabSize)
	}
	x := append([]int(nil), batch.x[:need]...)
	y := append([]int(nil), batch.x[:need]...)
	unmasked := append([]int(nil), batch.x[:need]...)
	lossMask := make([]float32, need)
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0x243f6a8885a308d3)
	maskProb := cfg.Training.EffectiveMLMMaskProbForStep(step)
	unit := cfg.Training.EffectiveMLMMaskUnitForStep(step)
	selected := 0
	stats := mlmMaskStats{Unit: unit, TargetProb: maskProb}
	if unit == arch.MLMMaskUnitWholeWord {
		var err error
		stats, err = selectWholeWordMaskPositions(rng, batch.x[:need], lossMask, maskProb, seqLen, cfg.Training.MLMWordStart, cfg.Training.MLMMaskEligible)
		if err != nil {
			return objectiveBatch{}, err
		}
		selected = stats.SelectedTokens
	} else {
		// Keep the legacy token selector and fallback in exactly this order.
		selected = selectMaskPositionsForBatch(rng, lossMask, maskProb, need, batch.maskEligible)
		if selected == 0 && maskProb > 0 && need > 0 {
			if len(batch.maskEligible) >= need {
				candidates := make([]int, 0, need)
				for i, active := range batch.maskEligible[:need] {
					if active != 0 {
						candidates = append(candidates, i)
					}
				}
				if len(candidates) > 0 {
					lossMask[candidates[rng.Intn(len(candidates))]] = 1
					selected = 1
				}
			} else {
				pos := rng.Intn(need)
				lossMask[pos] = 1
				selected = 1
			}
		}
		eligibleCount := need
		if len(batch.maskEligible) >= need {
			eligibleCount = 0
			for _, active := range batch.maskEligible[:need] {
				if active != 0 {
					eligibleCount++
				}
			}
		}
		stats = tokenMLMMaskStats(maskProb, eligibleCount, selected)
	}
	if selected > 0 {
		replaceSelectedMLMTokens(cfg, rng, x, y, lossMask)
	}
	return objectiveBatch{x: x, y: y, lossMask: lossMask, unmaskedX: unmasked, mlmMaskStats: stats}, nil
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
		if len(batch.maskEligible) >= need && batch.maskEligible[i+1] == 0 {
			continue
		}
		if len(batch.lossMask) >= need && batch.lossMask[i] <= 0 {
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
			if len(batch.maskEligible) >= need && batch.maskEligible[i+1] == 0 {
				continue
			}
			if len(batch.lossMask) >= need && batch.lossMask[i] <= 0 {
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

func prepareBlockDiffusionBatch(cfg *ArchConfig, batch trainBatch, step, need, seqLen int) (objectiveBatch, error) {
	if cfg.VocabSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid vocab_size=%d", cfg.VocabSize)
	}
	if cfg.Training.MLMMaskTokenID < 0 || cfg.Training.MLMMaskTokenID >= cfg.VocabSize {
		return objectiveBatch{}, fmt.Errorf("invalid mlm_mask_token_id=%d for vocab_size=%d", cfg.Training.MLMMaskTokenID, cfg.VocabSize)
	}
	diffusion, err := diffusionSpecForObjectiveBatch(cfg, seqLen)
	if err != nil {
		return objectiveBatch{}, err
	}
	if need%seqLen != 0 {
		return objectiveBatch{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", need, seqLen)
	}
	batchSize := need / seqLen
	blockSize := diffusion.BlockSize
	numBlocks := seqLen / blockSize
	x := append([]int(nil), batch.x[:need]...)
	y := append([]int(nil), batch.x[:need]...)
	unmasked := append([]int(nil), batch.x[:need]...)
	lossMask := make([]float32, need)
	diffusionBlockStart := make([]int32, batchSize)
	diffusionBlockEnd := make([]int32, batchSize)
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0xa4093822299f31d0)
	for b := 0; b < batchSize; b++ {
		rowStart := b * seqLen
		blockStart := rng.Intn(numBlocks) * blockSize
		blockEnd := blockStart + blockSize
		diffusionBlockStart[b] = int32(blockStart)
		diffusionBlockEnd[b] = int32(blockEnd)

		maskFraction := diffusion.MinMaskFraction
		if diffusion.MaxMaskFraction > diffusion.MinMaskFraction {
			maskFraction += rng.Float64() * (diffusion.MaxMaskFraction - diffusion.MinMaskFraction)
		}
		globalStart := rowStart + blockStart
		globalEnd := rowStart + blockEnd
		selected := selectMaskPositions(rng, lossMask[globalStart:globalEnd], maskFraction, blockSize)
		if selected == 0 && diffusion.MaxMaskFraction > 0 && blockSize > 0 {
			pos := globalStart + rng.Intn(blockSize)
			lossMask[pos] = 1
		}
		for i := globalStart; i < globalEnd; i++ {
			if lossMask[i] > 0 {
				x[i] = cfg.Training.MLMMaskTokenID
			}
		}
	}
	return objectiveBatch{
		x:                   x,
		y:                   y,
		lossMask:            lossMask,
		unmaskedX:           unmasked,
		diffusionBlockStart: diffusionBlockStart,
		diffusionBlockEnd:   diffusionBlockEnd,
	}, nil
}

func diffusionSpecForObjectiveBatch(cfg *ArchConfig, seqLen int) (arch.DiffusionSpec, error) {
	if seqLen <= 0 {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid seq_len=%d", seqLen)
	}
	var diffusion arch.DiffusionSpec
	if cfg.Training.Diffusion != nil {
		diffusion = *cfg.Training.Diffusion
	}
	if diffusion.BlockSize == 0 {
		diffusion.BlockSize = defaultDiffusionBlockSizeForBatch(seqLen)
	}
	if diffusion.MaxMaskFraction == 0 {
		diffusion.MaxMaskFraction = 1
	}
	if diffusion.BlockSize <= 0 {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid diffusion block_size=%d", diffusion.BlockSize)
	}
	if diffusion.BlockSize > seqLen {
		return arch.DiffusionSpec{}, fmt.Errorf("diffusion block_size=%d must be <= seq_len=%d", diffusion.BlockSize, seqLen)
	}
	if seqLen%diffusion.BlockSize != 0 {
		return arch.DiffusionSpec{}, fmt.Errorf("diffusion block_size=%d must divide seq_len=%d", diffusion.BlockSize, seqLen)
	}
	if math.IsNaN(diffusion.MinMaskFraction) || diffusion.MinMaskFraction < 0 || diffusion.MinMaskFraction > 1 {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid diffusion min_mask_fraction=%g", diffusion.MinMaskFraction)
	}
	if math.IsNaN(diffusion.MaxMaskFraction) || diffusion.MaxMaskFraction <= 0 || diffusion.MaxMaskFraction > 1 {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid diffusion max_mask_fraction=%g", diffusion.MaxMaskFraction)
	}
	if diffusion.MinMaskFraction > diffusion.MaxMaskFraction {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid diffusion mask fraction range [%g,%g]", diffusion.MinMaskFraction, diffusion.MaxMaskFraction)
	}
	return diffusion, nil
}

func defaultDiffusionBlockSizeForBatch(seqLen int) int {
	if seqLen <= 0 {
		return 0
	}
	if seqLen >= 16 && seqLen%16 == 0 {
		return 16
	}
	return seqLen
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
	causalFraction := cfg.Training.EffectiveHybridCLMFractionForStep(step)
	secondary := cfg.Training.EffectiveHybridSecondaryObjective()
	unit := cfg.Training.EffectiveMLMMaskUnitForStep(step)
	maskStats := mlmMaskStats{Unit: unit, TargetProb: maskProb}
	for b := 0; b < batchSize; b++ {
		start := b * seqLen
		end := start + seqLen
		if rng.Float64() < causalFraction {
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
			selected := 0
			if unit == arch.MLMMaskUnitWholeWord {
				rowStats, err := selectWholeWordMaskPositions(rng, batch.x[start:end], maskedLossMask[start:end], maskProb, seqLen, cfg.Training.MLMWordStart, cfg.Training.MLMMaskEligible)
				if err != nil {
					return objectiveBatch{}, err
				}
				selected = rowStats.SelectedTokens
				maskStats.add(rowStats)
			} else {
				selected = selectMaskPositions(rng, maskedLossMask[start:end], maskProb, seqLen)
				if selected == 0 && maskProb > 0 {
					pos := start + rng.Intn(seqLen)
					maskedLossMask[pos] = 1
					selected = 1
				}
				maskStats.add(tokenMLMMaskStats(maskProb, seqLen, selected))
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
		mlmMaskStats:    maskStats,
	}, nil
}

type mlmWordGroup struct {
	positions []int
}

func selectWholeWordMaskPositions(rng *rand.Rand, tokens []int, lossMask []float32, prob float64, seqLen int, wordStart, eligible []uint8) (mlmMaskStats, error) {
	stats := mlmMaskStats{Unit: arch.MLMMaskUnitWholeWord, TargetProb: prob}
	if seqLen <= 0 || len(tokens)%seqLen != 0 || len(lossMask) < len(tokens) {
		return stats, fmt.Errorf("invalid whole-word MLM batch shape: tokens=%d mask=%d seq_len=%d", len(tokens), len(lossMask), seqLen)
	}
	if len(wordStart) == 0 || len(wordStart) != len(eligible) {
		return stats, fmt.Errorf("whole-word MLM tokenizer metadata is not configured")
	}
	for rowStart := 0; rowStart < len(tokens); rowStart += seqLen {
		groups := make([]mlmWordGroup, 0, seqLen)
		active := -1
		for pos := 0; pos < seqLen; pos++ {
			idx := rowStart + pos
			token := tokens[idx]
			if token < 0 || token >= len(eligible) {
				return stats, fmt.Errorf("whole-word MLM token id %d at position %d is outside tokenizer metadata [0,%d)", token, idx, len(eligible))
			}
			if eligible[token] == 0 {
				active = -1
				continue
			}
			stats.EligibleTokens++
			if active < 0 || wordStart[token] != 0 {
				groups = append(groups, mlmWordGroup{})
				active = len(groups) - 1
			}
			groups[active].positions = append(groups[active].positions, idx)
		}
		stats.CandidateGroups += len(groups)
		eligibleCount := 0
		for _, group := range groups {
			eligibleCount += len(group.positions)
		}
		budget := int(math.Round(prob * float64(eligibleCount)))
		if prob > 0 && eligibleCount > 0 && budget < 1 {
			budget = 1
		}
		if budget > eligibleCount {
			budget = eligibleCount
		}
		stats.BudgetTokens += budget
		order := make([]int, len(groups))
		for i := range order {
			order[i] = i
		}
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		rowSelected := 0
		for _, groupIdx := range order {
			group := groups[groupIdx]
			groupSize := len(group.positions)
			if rowSelected+groupSize > budget {
				continue
			}
			for _, idx := range group.positions {
				lossMask[idx] = 1
			}
			rowSelected += groupSize
			stats.SelectedGroups++
			stats.SelectedGroupTokenTotal += groupSize
			if groupSize > stats.MaxSelectedGroupSize {
				stats.MaxSelectedGroupSize = groupSize
			}
			if rowSelected == budget {
				break
			}
		}
		stats.SelectedTokens += rowSelected
		if rowSelected < budget {
			stats.UnderfilledRows++
		}
	}
	return stats, nil
}

func tokenMLMMaskStats(prob float64, considered, selected int) mlmMaskStats {
	return mlmMaskStats{
		Unit:                    arch.MLMMaskUnitToken,
		TargetProb:              prob,
		EligibleTokens:          considered,
		BudgetTokens:            int(math.Round(prob * float64(considered))),
		SelectedTokens:          selected,
		CandidateGroups:         considered,
		SelectedGroups:          selected,
		SelectedGroupTokenTotal: selected,
		MaxSelectedGroupSize:    boolInt(selected > 0),
	}
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
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
			x[i] = randomEligibleToken(cfg, rng)
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
	if len(batch.segmentIDs) >= need {
		return nil
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
	return rand.New(rand.NewSource(int64(deterministicObjectiveSeed(seed, step, salt))))
}

func deterministicObjectiveSeed(seed int64, step int, salt uint64) uint64 {
	return splitmix64(uint64(seed) ^ uint64(step+1)*0x9e3779b97f4a7c15 ^ salt)
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
