package train

import (
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestPrepareWordStructuralMLMDeterministicAndNonOverlapping(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 6
	cfg.Training.BatchTokens = 6
	cfg.Training.Objective = arch.ObjectiveMLM
	cfg.Training.Seed = 123
	cfg.Training.MLMMaskProb = 0
	cfg.Training.WordStructuralObjective = &arch.WordStructuralObjectiveSpec{
		Enabled:      true,
		Fraction:     0.5,
		Span:         3,
		LossWeight:   1,
		SkipTokenIDs: []int{9},
	}
	batch := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6},
		y: []int{2, 3, 4, 5, 6, 7},
	}
	first, err := prepareObjectiveBatch(cfg, batch, 4, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare first MLM word-structural batch: %v", err)
	}
	second, err := prepareObjectiveBatch(cfg, batch, 4, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare second MLM word-structural batch: %v", err)
	}
	if !reflect.DeepEqual(first.x, second.x) || !reflect.DeepEqual(first.wordStructLossMask, second.wordStructLossMask) {
		t.Fatalf("word-structural batch is not deterministic:\nfirst x=%v mask=%v\nsecond x=%v mask=%v", first.x, first.wordStructLossMask, second.x, second.wordStructLossMask)
	}
	if reflect.DeepEqual(first.x, batch.x) {
		t.Fatalf("word-structural x was not shuffled: %v", first.x)
	}
	if !reflect.DeepEqual(first.y, batch.x) {
		t.Fatalf("MLM targets=%v, want originals %v", first.y, batch.x)
	}
	if !reflect.DeepEqual(first.wordStructTargets[:len(batch.x)], batch.x) {
		t.Fatalf("word_struct_targets=%v, want current-position originals %v", first.wordStructTargets, batch.x)
	}
	shuffled := 0
	for i, active := range first.wordStructLossMask {
		if active <= 0 {
			continue
		}
		shuffled++
		if first.lossMask[i] > 0 {
			t.Fatalf("word-structural position %d overlaps primary MLM mask", i)
		}
		if first.wordStructTargets[i] != batch.x[i] {
			t.Fatalf("word-structural target at %d=%d, want original %d", i, first.wordStructTargets[i], batch.x[i])
		}
	}
	if shuffled != 3 {
		t.Fatalf("word-structural shuffled positions=%d, want one trigram", shuffled)
	}
}

func TestPrepareWordStructuralMNTPUsesCurrentPositionTargets(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 6
	cfg.Training.BatchTokens = 6
	cfg.Training.Objective = arch.ObjectiveMNTP
	cfg.Training.Seed = 99
	cfg.Training.MLMMaskProb = 0
	cfg.Training.WordStructuralObjective = &arch.WordStructuralObjectiveSpec{
		Enabled:      true,
		Fraction:     0.5,
		Span:         3,
		LossWeight:   1,
		SkipTokenIDs: []int{9},
	}
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 14, 15},
		y: []int{11, 12, 13, 14, 15, 16},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 2, arch.ObjectiveMNTP)
	if err != nil {
		t.Fatalf("prepare MNTP word-structural batch: %v", err)
	}
	if len(got.wordStructLossMask) != cfg.Training.BatchTokens {
		t.Fatalf("wordStructLossMask length=%d, want %d", len(got.wordStructLossMask), cfg.Training.BatchTokens)
	}
	for i, active := range got.wordStructLossMask {
		if active <= 0 {
			continue
		}
		if got.wordStructTargets[i] != batch.x[i] {
			t.Fatalf("word-structural MNTP target at %d=%d, want current original %d", i, got.wordStructTargets[i], batch.x[i])
		}
		if got.wordStructTargets[i] == batch.y[i] {
			t.Fatalf("word-structural MNTP target at %d followed shifted target convention", i)
		}
	}
}

func TestPrepareWordStructuralMNTPExcludesPrimaryLossAndInputMaskPositions(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 6
	cfg.Training.BatchTokens = 6
	cfg.Training.Objective = arch.ObjectiveMNTP
	cfg.Training.MLMMaskProb = 1
	cfg.Training.WordStructuralObjective = &arch.WordStructuralObjectiveSpec{
		Enabled:      true,
		Fraction:     0.5,
		Span:         3,
		LossWeight:   1,
		SkipTokenIDs: []int{9},
	}
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 14, 15},
		y: []int{11, 12, 13, 14, 15, 16},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 3, arch.ObjectiveMNTP)
	if err != nil {
		t.Fatalf("prepare MNTP word-structural excluded batch: %v", err)
	}
	for i, active := range got.wordStructLossMask {
		if active > 0 {
			t.Fatalf("word-structural selected position %d despite full MNTP primary masking; mask=%v", i, got.wordStructLossMask)
		}
	}
}

func TestPrepareWordStructuralHybridExampleSkipsCausalRows(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 8
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridMixGranularity = arch.HybridMixGranularityExample
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveMLM
	cfg.Training.HybridCLMFraction = 0.5
	cfg.Training.MLMMaskProb = 0
	cfg.Training.WordStructuralObjective = &arch.WordStructuralObjectiveSpec{
		Enabled:      true,
		Fraction:     0.5,
		Span:         2,
		LossWeight:   1,
		SkipTokenIDs: []int{9},
	}
	batch := trainBatch{
		x: []int{1, 2, 3, 4, 10, 11, 12, 13},
		y: []int{2, 3, 4, 5, 11, 12, 13, 14},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveHybridExample)
	if err != nil {
		t.Fatalf("prepare hybrid-example word-structural batch: %v", err)
	}
	if len(got.attentionCausal) == 0 {
		t.Fatal("hybrid-example batch missing attentionCausal rows")
	}
	for row, causal := range got.attentionCausal {
		rowMask := got.wordStructLossMask[row*cfg.SeqLen : (row+1)*cfg.SeqLen]
		if causal > 0 {
			for pos, active := range rowMask {
				if active > 0 {
					t.Fatalf("causal row %d pos %d has word-structural mask=%v", row, pos, rowMask)
				}
			}
		}
	}
}

func TestPrepareWordStructuralDoesNotCrossSegmentBoundary(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 6
	cfg.Training.BatchTokens = 6
	cfg.Training.Objective = arch.ObjectiveMLM
	cfg.Training.MLMMaskProb = 0
	cfg.Training.AttentionSegmentMask = arch.AttentionSegmentMaskBoundaryToken
	cfg.Training.AttentionSegmentBoundaryTokenID = 1
	cfg.Training.WordStructuralObjective = &arch.WordStructuralObjectiveSpec{
		Enabled:      true,
		Fraction:     0.5,
		Span:         3,
		LossWeight:   1,
		SkipTokenIDs: []int{1, 9},
	}
	batch := trainBatch{
		x: []int{2, 3, 1, 4, 5, 6},
		y: []int{3, 1, 4, 5, 6, 7},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 8, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare segmented word-structural batch: %v", err)
	}
	for i, active := range got.wordStructLossMask {
		if active <= 0 {
			continue
		}
		if i < 3 {
			t.Fatalf("word-structural selected position %d across or before boundary; mask=%v", i, got.wordStructLossMask)
		}
	}
}
