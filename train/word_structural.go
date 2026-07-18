package train

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/mrothroc/mixlab/arch"
)

type wordStructuralSpan struct {
	start int
}

const wordStructuralRNGSalt uint64 = 0x6a09e667f3bcc909

func maybeApplyWordStructuralObjective(cfg *ArchConfig, batch *objectiveBatch, step int, objective string, need, seqLen int) error {
	if cfg == nil || batch == nil || !cfg.Training.WordStructuralActiveForConcreteObjective(objective) {
		return nil
	}
	if seqLen <= 0 || need%seqLen != 0 {
		return fmt.Errorf("invalid word-structural batch shape: tokens=%d seq_len=%d", need, seqLen)
	}
	ensureWordStructuralBuffers(batch, need)
	source := wordStructuralSource(batch, need)
	copy(batch.wordStructTargets[:need], source[:need])
	clear(batch.wordStructLossMask[:need])
	var segmentIDs []int32
	if cfg.Training.AttentionSegmentMaskEnabled() {
		segmentIDs = deriveSegmentIDs(source[:need], need, seqLen, cfg.Training.AttentionSegmentBoundaryTokenID)
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, wordStructuralRNGSalt)
	mode := canonicalObjective(objective)
	if mode == arch.ObjectiveHybridExample {
		mode = cfg.Training.EffectiveHybridSecondaryObjective()
	}
	rows := need / seqLen
	for row := 0; row < rows; row++ {
		if canonicalObjective(objective) == arch.ObjectiveHybridExample && len(batch.attentionCausal) > row && batch.attentionCausal[row] > 0 {
			continue
		}
		applyWordStructuralRow(cfg, batch, source, segmentIDs, rng, mode, row*seqLen, seqLen)
	}
	return nil
}

func maybeApplyWordStructuralMultihead(cfg *ArchConfig, batch *objectiveBatch, step, rawRows, seqLen int) error {
	if cfg == nil || batch == nil || !cfg.Training.WordStructuralActive() || !cfg.Training.MultiheadEnabled() {
		return nil
	}
	if rawRows <= 0 || seqLen <= 0 {
		return fmt.Errorf("invalid multihead word-structural shape: rows=%d seq_len=%d", rawRows, seqLen)
	}
	need := len(batch.x)
	if len(batch.y) < need {
		return fmt.Errorf("invalid multihead word-structural batch: targets=%d tokens=%d", len(batch.y), need)
	}
	ensureWordStructuralBuffers(batch, need)
	source := wordStructuralSource(batch, need)
	copy(batch.wordStructTargets[:need], source[:need])
	clear(batch.wordStructLossMask[:need])
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, wordStructuralRNGSalt)
	for headIdx, head := range cfg.Training.Heads {
		if !cfg.Training.WordStructuralHeadSelected(head.Name) {
			continue
		}
		mode := canonicalObjective(head.Objective)
		if mode != arch.ObjectiveMLM && mode != arch.ObjectiveMNTP {
			continue
		}
		rowBase := headIdx * rawRows
		for row := 0; row < rawRows; row++ {
			applyWordStructuralRow(cfg, batch, source, nil, rng, mode, (rowBase+row)*seqLen, seqLen)
		}
	}
	return nil
}

func ensureWordStructuralBuffers(batch *objectiveBatch, need int) {
	if len(batch.wordStructTargets) < need {
		batch.wordStructTargets = make([]int, need)
	}
	if len(batch.wordStructLossMask) < need {
		batch.wordStructLossMask = make([]float32, need)
	}
}

func wordStructuralSource(batch *objectiveBatch, need int) []int {
	if len(batch.unmaskedX) >= need {
		return batch.unmaskedX[:need]
	}
	return batch.x[:need]
}

func applyWordStructuralRow(cfg *ArchConfig, batch *objectiveBatch, source []int, segmentIDs []int32, rng *rand.Rand, mode string, rowStart, seqLen int) {
	spec := cfg.Training.WordStructuralObjective
	if spec == nil || !spec.Enabled || rowStart < 0 || rowStart+seqLen > len(batch.x) || rowStart+seqLen > len(source) {
		return
	}
	eligiblePositions := 0
	for pos := 0; pos < seqLen; pos++ {
		if wordStructuralEligibleAt(cfg, batch, source, mode, rowStart, pos) {
			eligiblePositions++
		}
	}
	if eligiblePositions == 0 {
		return
	}
	targetSpans := int(math.Ceil(spec.Fraction * float64(eligiblePositions) / float64(spec.Span)))
	if targetSpans <= 0 {
		return
	}
	candidates := wordStructuralCandidateSpans(cfg, batch, source, segmentIDs, mode, rowStart, seqLen)
	if len(candidates) == 0 {
		return
	}
	rng.Shuffle(len(candidates), func(i, j int) {
		candidates[i], candidates[j] = candidates[j], candidates[i]
	})
	occupied := make([]bool, seqLen)
	selected := 0
	for _, span := range candidates {
		if selected >= targetSpans {
			break
		}
		local := span.start - rowStart
		overlap := false
		for i := 0; i < spec.Span; i++ {
			if occupied[local+i] {
				overlap = true
				break
			}
		}
		if overlap {
			continue
		}
		if applyWordStructuralPermutation(batch, source, rng, span.start, spec.Span) {
			for i := 0; i < spec.Span; i++ {
				occupied[local+i] = true
				batch.wordStructLossMask[span.start+i] = 1
			}
			selected++
		}
	}
}

func wordStructuralCandidateSpans(cfg *ArchConfig, batch *objectiveBatch, source []int, segmentIDs []int32, mode string, rowStart, seqLen int) []wordStructuralSpan {
	spec := cfg.Training.WordStructuralObjective
	if spec == nil || spec.Span > seqLen {
		return nil
	}
	var out []wordStructuralSpan
	for pos := 0; pos <= seqLen-spec.Span; pos++ {
		start := rowStart + pos
		segment := int32(0)
		if len(segmentIDs) > start {
			segment = segmentIDs[start]
		}
		distinct := false
		first := source[start]
		valid := true
		for i := 0; i < spec.Span; i++ {
			idx := start + i
			if !wordStructuralEligibleAt(cfg, batch, source, mode, rowStart, pos+i) {
				valid = false
				break
			}
			if len(segmentIDs) > idx && segmentIDs[idx] != segment {
				valid = false
				break
			}
			if source[idx] != first {
				distinct = true
			}
		}
		if valid && distinct {
			out = append(out, wordStructuralSpan{start: start})
		}
	}
	return out
}

func wordStructuralEligibleAt(cfg *ArchConfig, batch *objectiveBatch, source []int, mode string, rowStart, pos int) bool {
	idx := rowStart + pos
	if idx < 0 || idx >= len(source) || idx >= len(batch.x) {
		return false
	}
	if wordStructuralSkipToken(cfg, source[idx]) {
		return false
	}
	if len(batch.lossMask) > idx && batch.lossMask[idx] > 0 {
		return false
	}
	if mode == arch.ObjectiveMNTP && pos > 0 {
		prev := idx - 1
		if len(batch.lossMask) > prev && batch.lossMask[prev] > 0 {
			return false
		}
	}
	return true
}

func wordStructuralSkipToken(cfg *ArchConfig, token int) bool {
	if cfg == nil || cfg.Training.WordStructuralObjective == nil {
		return false
	}
	for _, skip := range cfg.Training.WordStructuralObjective.SkipTokenIDs {
		if token == skip {
			return true
		}
	}
	return false
}

func applyWordStructuralPermutation(batch *objectiveBatch, source []int, rng *rand.Rand, start, span int) bool {
	perm := make([]int, span)
	for i := range perm {
		perm[i] = i
	}
	for tries := 0; tries < 32; tries++ {
		rng.Shuffle(span, func(i, j int) {
			perm[i], perm[j] = perm[j], perm[i]
		})
		if wordStructuralPermutationChangesValues(source[start:start+span], perm) {
			applyWordStructuralPerm(batch, source, start, perm)
			return true
		}
	}
	for shift := 1; shift < span; shift++ {
		for i := range perm {
			perm[i] = (i + shift) % span
		}
		if wordStructuralPermutationChangesValues(source[start:start+span], perm) {
			applyWordStructuralPerm(batch, source, start, perm)
			return true
		}
	}
	return false
}

func wordStructuralPermutationChangesValues(tokens []int, perm []int) bool {
	if len(tokens) != len(perm) {
		return false
	}
	for i, p := range perm {
		if p < 0 || p >= len(tokens) {
			return false
		}
		if p != i && tokens[p] != tokens[i] {
			return true
		}
	}
	return false
}

func applyWordStructuralPerm(batch *objectiveBatch, source []int, start int, perm []int) {
	orig := append([]int(nil), batch.x[start:start+len(perm)]...)
	for i, p := range perm {
		batch.x[start+i] = orig[p]
		batch.wordStructTargets[start+i] = source[start+i]
	}
}
