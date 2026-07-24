package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
)

const reverseComplementRNGSalt uint64 = 0xbb67ae8584caa73b

func maybeApplyReverseComplement(cfg *ArchConfig, batch trainBatch, step, need int) (trainBatch, error) {
	if cfg == nil || (!cfg.Training.DatasetSequencePacking && !cfg.Training.DatasetRecordFraming && !cfg.Training.DatasetNucleotideStream) ||
		cfg.Training.ReverseComplementProb == 0 || batch.disableAugmentation {
		return batch, nil
	}
	if cfg.Training.DatasetNucleotideAlphabet != "dna" {
		return trainBatch{}, fmt.Errorf("reverse-complement augmentation requires a DNA dataset")
	}
	if len(cfg.Training.DatasetNucleotideComplement) != cfg.VocabSize {
		return trainBatch{}, fmt.Errorf("DNA complement lookup has %d entries, want vocab_size=%d", len(cfg.Training.DatasetNucleotideComplement), cfg.VocabSize)
	}
	if cfg.Training.DatasetNucleotideStream {
		return applyStreamReverseComplement(cfg, batch, step, need)
	}
	if len(batch.x) < need || len(batch.segmentIDs) < need || len(batch.maskEligible) < need {
		return trainBatch{}, fmt.Errorf("reverse-complement batch metadata is incomplete: x=%d y=%d segments=%d eligible=%d need=%d",
			len(batch.x), len(batch.y), len(batch.segmentIDs), len(batch.maskEligible), need)
	}
	out := batch
	out.x = append([]int(nil), batch.x...)
	if len(batch.y) >= need {
		out.y = append([]int(nil), batch.y...)
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, reverseComplementRNGSalt)
	for rowStart := 0; rowStart < need; rowStart += cfg.SeqLen {
		rowEnd := rowStart + cfg.SeqLen
		if rowEnd > need {
			rowEnd = need
		}
		for start := rowStart; start < rowEnd; {
			segment := batch.segmentIDs[start]
			end := start + 1
			for end < rowEnd && batch.segmentIDs[end] == segment {
				end++
			}
			if rng.Float64() < cfg.Training.ReverseComplementProb {
				positions := make([]int, 0, end-start)
				for i := start; i < end; i++ {
					if batch.maskEligible[i] != 0 {
						positions = append(positions, i)
					}
				}
				for left, right := 0, len(positions)-1; left <= right; left, right = left+1, right-1 {
					leftToken := batch.x[positions[right]]
					rightToken := batch.x[positions[left]]
					out.x[positions[left]] = cfg.Training.DatasetNucleotideComplement[leftToken]
					out.x[positions[right]] = cfg.Training.DatasetNucleotideComplement[rightToken]
				}
			}
			start = end
		}
	}
	for i := 0; i < need && len(out.y) >= need; i++ {
		if len(batch.lossMask) > i && batch.lossMask[i] > 0 && i+1 < need &&
			i/cfg.SeqLen == (i+1)/cfg.SeqLen && batch.segmentIDs[i] == batch.segmentIDs[i+1] {
			out.y[i] = out.x[i+1]
		}
	}
	return out, nil
}

func applyStreamReverseComplement(cfg *ArchConfig, batch trainBatch, step, need int) (trainBatch, error) {
	if len(batch.x) < need {
		return trainBatch{}, fmt.Errorf("reverse-complement stream batch has %d input tokens, need %d", len(batch.x), need)
	}
	if len(cfg.Training.DatasetTokenEligible) != cfg.VocabSize {
		return trainBatch{}, fmt.Errorf("DNA biological-token lookup has %d entries, want vocab_size=%d", len(cfg.Training.DatasetTokenEligible), cfg.VocabSize)
	}
	out := batch
	out.x = append([]int(nil), batch.x...)
	if len(batch.y) >= need {
		out.y = append([]int(nil), batch.y...)
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, reverseComplementRNGSalt)
	for rowStart := 0; rowStart < need; rowStart += cfg.SeqLen {
		rowEnd := rowStart + cfg.SeqLen
		if rowEnd > need {
			rowEnd = need
		}
		if rng.Float64() >= cfg.Training.ReverseComplementProb {
			continue
		}
		for start := rowStart; start < rowEnd; {
			for start < rowEnd && !nucleotideStreamTokenEligible(cfg, batch.x[start]) {
				start++
			}
			end := start
			for end < rowEnd && nucleotideStreamTokenEligible(cfg, batch.x[end]) {
				end++
			}
			for left, right := start, end-1; left <= right; left, right = left+1, right-1 {
				leftToken, rightToken := batch.x[left], batch.x[right]
				out.x[left] = cfg.Training.DatasetNucleotideComplement[rightToken]
				out.x[right] = cfg.Training.DatasetNucleotideComplement[leftToken]
			}
			start = end
		}
	}
	if len(out.y) >= need {
		for rowStart := 0; rowStart < need; rowStart += cfg.SeqLen {
			rowEnd := rowStart + cfg.SeqLen
			if rowEnd > need {
				rowEnd = need
			}
			for i := rowStart; i+1 < rowEnd; i++ {
				if len(batch.lossMask) < need || batch.lossMask[i] > 0 {
					out.y[i] = out.x[i+1]
				}
			}
		}
	}
	return out, nil
}

func nucleotideStreamTokenEligible(cfg *ArchConfig, token int) bool {
	return token >= 0 && token < len(cfg.Training.DatasetTokenEligible) &&
		cfg.Training.DatasetTokenEligible[token] != 0
}

func attachRCEquivariantInputs(cfg *ArchConfig, batch trainBatch, prepared *objectiveBatch, need, seqLen int) error {
	if cfg == nil || !cfg.RCEquivarianceEnabled() {
		return nil
	}
	if prepared == nil {
		return fmt.Errorf("RC-equivariant batch preparation requires an objective batch")
	}
	if cfg.Training.DatasetNucleotideAlphabet != "dna" {
		return fmt.Errorf("rc_equivariant=true requires a DNA dataset")
	}
	if len(cfg.Training.DatasetNucleotideComplement) != cfg.VocabSize {
		return fmt.Errorf("DNA complement lookup has %d entries, want vocab_size=%d", len(cfg.Training.DatasetNucleotideComplement), cfg.VocabSize)
	}
	if seqLen <= 0 || need <= 0 || need%seqLen != 0 || len(prepared.x) < need {
		return fmt.Errorf("invalid RC-equivariant batch shape: tokens=%d need=%d seq_len=%d", len(prepared.x), need, seqLen)
	}
	if len(batch.maskEligible) < need {
		return fmt.Errorf("RC-equivariant batch is missing biological-token eligibility: got=%d need=%d", len(batch.maskEligible), need)
	}

	rcTokens := append([]int(nil), prepared.x[:need]...)
	alignment := make([]int32, need)
	for i := range alignment {
		alignment[i] = int32(i)
	}
	for rowStart := 0; rowStart < need; rowStart += seqLen {
		rowEnd := rowStart + seqLen
		for segmentStart := rowStart; segmentStart < rowEnd; {
			segmentEnd := rowEnd
			if len(batch.segmentIDs) >= need {
				segment := batch.segmentIDs[segmentStart]
				segmentEnd = segmentStart + 1
				for segmentEnd < rowEnd && batch.segmentIDs[segmentEnd] == segment {
					segmentEnd++
				}
			}
			positions := make([]int, 0, segmentEnd-segmentStart)
			for pos := segmentStart; pos < segmentEnd; pos++ {
				if batch.maskEligible[pos] != 0 {
					positions = append(positions, pos)
				}
			}
			for left, right := 0, len(positions)-1; left <= right; left, right = left+1, right-1 {
				dstLeft, srcRight := positions[left], positions[right]
				dstRight, srcLeft := positions[right], positions[left]
				rightToken := prepared.x[srcRight]
				leftToken := prepared.x[srcLeft]
				if rightToken < 0 || rightToken >= len(cfg.Training.DatasetNucleotideComplement) ||
					leftToken < 0 || leftToken >= len(cfg.Training.DatasetNucleotideComplement) {
					return fmt.Errorf("RC-equivariant token id outside complement lookup: left=%d right=%d", leftToken, rightToken)
				}
				rcTokens[dstLeft] = cfg.Training.DatasetNucleotideComplement[rightToken]
				rcTokens[dstRight] = cfg.Training.DatasetNucleotideComplement[leftToken]
				alignment[dstLeft] = int32(srcRight)
				alignment[dstRight] = int32(srcLeft)
			}
			segmentStart = segmentEnd
		}
	}
	prepared.rcTokens = rcTokens
	prepared.rcAlignmentPositions = alignment
	return nil
}

func selectMaskPositionsForBatch(rng interface{ Float64() float64 }, lossMask []float32, prob float64, n int, eligible []uint8) int {
	if len(eligible) < n {
		// Keep legacy RNG and selection behavior byte-identical.
		selected := 0
		for i := 0; i < n; i++ {
			if rng.Float64() < prob {
				lossMask[i] = 1
				selected++
			}
		}
		return selected
	}
	selected := 0
	for i := 0; i < n; i++ {
		if eligible[i] == 0 {
			continue
		}
		if rng.Float64() < prob {
			lossMask[i] = 1
			selected++
		}
	}
	return selected
}

func randomEligibleToken(cfg *ArchConfig, rng interface{ Intn(int) int }) int {
	eligible := cfg.Training.DatasetTokenEligible
	if len(eligible) != cfg.VocabSize {
		return rng.Intn(cfg.VocabSize)
	}
	count := 0
	for _, active := range eligible {
		if active != 0 {
			count++
		}
	}
	if count == 0 {
		return rng.Intn(cfg.VocabSize)
	}
	draw := rng.Intn(count)
	for id, active := range eligible {
		if active == 0 {
			continue
		}
		if draw == 0 {
			return id
		}
		draw--
	}
	return 0
}

func validateNucleotideRuntimeObjective(cfg *ArchConfig) error {
	if cfg == nil || !cfg.Training.DatasetSequencePacking {
		return nil
	}
	if len(cfg.Training.SeqLenSchedule) > 0 {
		return fmt.Errorf("record-oriented sequence packing does not support training.seq_len_schedule in this release")
	}
	if cfg.Training.UsesWholeWordMasking() {
		return fmt.Errorf("nucleotide sequence training does not support whole-word MLM masking")
	}
	switch cfg.Training.EffectiveObjective() {
	case arch.ObjectiveCausal, arch.ObjectiveMLM, arch.ObjectiveMNTP, arch.ObjectiveHybrid:
		return nil
	default:
		return fmt.Errorf("nucleotide sequence packing supports causal, mlm, mntp, or hybrid objectives in this release; got %q", cfg.Training.EffectiveObjective())
	}
}
