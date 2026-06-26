package train

import (
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestPrepareObjectiveBatchMLMTargetsAndReplacementModes(t *testing.T) {
	batch := trainBatch{
		x: []int{1, 2, 3, 4},
		y: []int{2, 3, 4, 5},
	}
	cfg := objectiveTestConfig()
	cfg.Training.MLMMaskProb = 1

	cfg.Training.MLMMaskTokenProb = 1
	cfg.Training.MLMRandomTokenProb = 0
	cfg.Training.MLMKeptUnchangedProb = 0
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare MLM mask-token batch: %v", err)
	}
	if want := []int{9, 9, 9, 9}; !reflect.DeepEqual(got.x, want) {
		t.Fatalf("mask-token x = %v, want %v", got.x, want)
	}
	if !reflect.DeepEqual(got.y, batch.x) {
		t.Fatalf("MLM targets = %v, want original inputs %v", got.y, batch.x)
	}
	if want := []float32{1, 1, 1, 1}; !reflect.DeepEqual(got.lossMask, want) {
		t.Fatalf("lossMask = %v, want %v", got.lossMask, want)
	}

	cfg.Training.MLMMaskTokenProb = 0
	cfg.Training.MLMRandomTokenProb = 0
	cfg.Training.MLMKeptUnchangedProb = 1
	got, err = prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare MLM kept-token batch: %v", err)
	}
	if !reflect.DeepEqual(got.x, batch.x) {
		t.Fatalf("kept-token x = %v, want original inputs %v", got.x, batch.x)
	}

	cfg.Training.MLMMaskTokenProb = 0
	cfg.Training.MLMRandomTokenProb = 1
	cfg.Training.MLMKeptUnchangedProb = 0
	got, err = prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare MLM random-token batch: %v", err)
	}
	for i, tok := range got.x {
		if tok < 0 || tok >= cfg.VocabSize {
			t.Fatalf("random replacement token %d at %d outside vocab size %d", tok, i, cfg.VocabSize)
		}
	}
}

func TestPrepareObjectiveBatchUsesMLMMaskProbSchedule(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.Training.BatchTokens = 8
	cfg.Training.MLMMaskProb = 0
	cfg.Training.MLMMaskProbSchedule = [][]float64{{0, 0}, {1, 1}}
	cfg.Training.MLMMaskTokenProb = 1
	cfg.Training.MLMRandomTokenProb = 0
	cfg.Training.MLMKeptUnchangedProb = 0
	batch := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 5, 6, 7, 8, 9},
	}

	step0, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare step0 MLM batch: %v", err)
	}
	if want := []float32{0, 0, 0, 0, 0, 0, 0, 0}; !reflect.DeepEqual(step0.lossMask, want) {
		t.Fatalf("step0 lossMask = %v, want %v", step0.lossMask, want)
	}
	if !reflect.DeepEqual(step0.x, batch.x) {
		t.Fatalf("step0 x = %v, want unchanged %v", step0.x, batch.x)
	}

	step1, err := prepareObjectiveBatch(cfg, batch, 1, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare step1 MLM batch: %v", err)
	}
	if want := []float32{1, 1, 1, 1, 1, 1, 1, 1}; !reflect.DeepEqual(step1.lossMask, want) {
		t.Fatalf("step1 lossMask = %v, want %v", step1.lossMask, want)
	}
	if want := []int{9, 9, 9, 9, 9, 9, 9, 9}; !reflect.DeepEqual(step1.x, want) {
		t.Fatalf("step1 x = %v, want %v", step1.x, want)
	}
}

func TestPrepareObjectiveBatchMNTPMasksNextInputWithoutLeakage(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 8
	cfg.Training.MLMMaskProb = 1
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 20, 21, 22, 23},
		y: []int{11, 12, 13, 14, 21, 22, 23, 24},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 5, arch.ObjectiveMNTP)
	if err != nil {
		t.Fatalf("prepare MNTP batch: %v", err)
	}
	wantX := []int{10, 9, 9, 9, 20, 9, 9, 9}
	if !reflect.DeepEqual(got.x, wantX) {
		t.Fatalf("MNTP x = %v, want %v", got.x, wantX)
	}
	if !reflect.DeepEqual(got.y, batch.y) {
		t.Fatalf("MNTP targets = %v, want next-token targets %v", got.y, batch.y)
	}
	wantMask := []float32{1, 1, 1, 0, 1, 1, 1, 0}
	if !reflect.DeepEqual(got.lossMask, wantMask) {
		t.Fatalf("MNTP lossMask = %v, want %v", got.lossMask, wantMask)
	}
	for i, active := range got.lossMask {
		if active > 0 && got.x[i+1] == got.y[i] {
			t.Fatalf("MNTP leaked target at loss position %d: x[next]=%d target=%d", i, got.x[i+1], got.y[i])
		}
	}
}

func TestPrepareBlockDiffusionBatch(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 6
	cfg.VocabSize = 64
	cfg.Training.BatchTokens = 12
	cfg.Training.Objective = arch.ObjectiveBlockDiffusion
	cfg.Training.Diffusion = &arch.DiffusionSpec{
		BlockSize:       3,
		MinMaskFraction: 1,
		MaxMaskFraction: 1,
	}
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25},
		y: []int{11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 7, arch.ObjectiveBlockDiffusion)
	if err != nil {
		t.Fatalf("prepare block diffusion batch: %v", err)
	}
	if !reflect.DeepEqual(got.y, batch.x) {
		t.Fatalf("block diffusion targets = %v, want clean inputs %v", got.y, batch.x)
	}
	if !reflect.DeepEqual(got.unmaskedX, batch.x) {
		t.Fatalf("block diffusion unmaskedX = %v, want clean inputs %v", got.unmaskedX, batch.x)
	}
	if len(got.diffusionBlockStart) != 2 || len(got.diffusionBlockEnd) != 2 {
		t.Fatalf("diffusion block vectors have lengths start=%d end=%d, want 2", len(got.diffusionBlockStart), len(got.diffusionBlockEnd))
	}
	for b := 0; b < 2; b++ {
		blockStart := int(got.diffusionBlockStart[b])
		blockEnd := int(got.diffusionBlockEnd[b])
		if blockEnd-blockStart != cfg.Training.Diffusion.BlockSize || blockStart%cfg.Training.Diffusion.BlockSize != 0 || blockStart < 0 || blockEnd > cfg.SeqLen {
			t.Fatalf("row %d diffusion block=[%d,%d), want aligned size-%d block inside seq_len=%d", b, blockStart, blockEnd, cfg.Training.Diffusion.BlockSize, cfg.SeqLen)
		}
		rowStart := b * cfg.SeqLen
		for pos := 0; pos < cfg.SeqLen; pos++ {
			i := rowStart + pos
			inActiveBlock := pos >= blockStart && pos < blockEnd
			wantMask := float32(0)
			wantX := batch.x[i]
			if inActiveBlock {
				wantMask = 1
				wantX = cfg.Training.MLMMaskTokenID
			}
			if got.lossMask[i] != wantMask {
				t.Fatalf("row %d pos %d lossMask=%g, want %g for active block [%d,%d)", b, pos, got.lossMask[i], wantMask, blockStart, blockEnd)
			}
			if got.x[i] != wantX {
				t.Fatalf("row %d pos %d x=%d, want %d for active block [%d,%d)", b, pos, got.x[i], wantX, blockStart, blockEnd)
			}
		}
	}
}

func TestBlockDiffusionObjectiveForStep(t *testing.T) {
	spec := TrainingSpec{
		Objective: arch.ObjectiveBlockDiffusion,
		Seed:      123,
	}
	for step := 0; step < 8; step++ {
		if got := objectiveForStep(spec, step); got != arch.ObjectiveBlockDiffusion {
			t.Fatalf("objectiveForStep step %d = %q, want %q", step, got, arch.ObjectiveBlockDiffusion)
		}
	}
	if got := canonicalObjective(" " + arch.ObjectiveBlockDiffusion + " "); got != arch.ObjectiveBlockDiffusion {
		t.Fatalf("canonicalObjective(block_diffusion) = %q, want %q", got, arch.ObjectiveBlockDiffusion)
	}
}

func TestBlockDiffusionBatchDeterministic(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 8
	cfg.VocabSize = 128
	cfg.Training.BatchTokens = 16
	cfg.Training.Seed = 777
	cfg.Training.MLMMaskTokenID = 99
	cfg.Training.Objective = arch.ObjectiveBlockDiffusion
	cfg.Training.Diffusion = &arch.DiffusionSpec{
		BlockSize:       4,
		MinMaskFraction: 0,
		MaxMaskFraction: 1e-9,
	}
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35, 36, 37},
		y: []int{11, 12, 13, 14, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36, 37, 38},
	}
	first, err := prepareObjectiveBatch(cfg, batch, 12, arch.ObjectiveBlockDiffusion)
	if err != nil {
		t.Fatalf("prepare first block diffusion batch: %v", err)
	}
	second, err := prepareObjectiveBatch(cfg, batch, 12, arch.ObjectiveBlockDiffusion)
	if err != nil {
		t.Fatalf("prepare second block diffusion batch: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("block diffusion batch is not deterministic for fixed seed+step:\nfirst=%+v\nsecond=%+v", first, second)
	}
	for b := 0; b < 2; b++ {
		blockStart := int(first.diffusionBlockStart[b])
		blockEnd := int(first.diffusionBlockEnd[b])
		rowStart := b * cfg.SeqLen
		masked := 0
		for pos := 0; pos < cfg.SeqLen; pos++ {
			i := rowStart + pos
			inActiveBlock := pos >= blockStart && pos < blockEnd
			if first.lossMask[i] > 0 {
				masked++
				if !inActiveBlock {
					t.Fatalf("row %d pos %d is masked outside active block [%d,%d)", b, pos, blockStart, blockEnd)
				}
				if first.x[i] != cfg.Training.MLMMaskTokenID {
					t.Fatalf("row %d pos %d masked x=%d, want mask token %d", b, pos, first.x[i], cfg.Training.MLMMaskTokenID)
				}
				continue
			}
			if first.x[i] != batch.x[i] {
				t.Fatalf("row %d pos %d unmasked x=%d, want clean token %d", b, pos, first.x[i], batch.x[i])
			}
		}
		if masked == 0 {
			t.Fatalf("row %d has no masked token despite max_mask_fraction=%g", b, cfg.Training.Diffusion.MaxMaskFraction)
		}
	}
}

func TestBlockDiffusionObjectiveBatchInputs(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 8
	cfg.VocabSize = 64
	cfg.Training.BatchTokens = 16
	cfg.Training.Seed = 42
	cfg.Training.Objective = arch.ObjectiveBlockDiffusion
	cfg.Training.Diffusion = &arch.DiffusionSpec{
		BlockSize:       4,
		MinMaskFraction: 1,
		MaxMaskFraction: 1,
	}
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27},
		y: []int{11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 3, arch.ObjectiveBlockDiffusion)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch(block_diffusion): %v", err)
	}
	if len(got.diffusionBlockStart) != 2 || len(got.diffusionBlockEnd) != 2 {
		t.Fatalf("diffusion block inputs lengths start=%d end=%d, want 2/2", len(got.diffusionBlockStart), len(got.diffusionBlockEnd))
	}
	if !reflect.DeepEqual(got.y, batch.x) {
		t.Fatalf("block_diffusion targets=%v, want clean inputs %v", got.y, batch.x)
	}
	for b := 0; b < 2; b++ {
		start := int(got.diffusionBlockStart[b])
		end := int(got.diffusionBlockEnd[b])
		if start < 0 || end > cfg.SeqLen || end-start != cfg.Training.Diffusion.BlockSize || start%cfg.Training.Diffusion.BlockSize != 0 {
			t.Fatalf("row %d block=[%d,%d), want aligned size-%d block in seq_len=%d", b, start, end, cfg.Training.Diffusion.BlockSize, cfg.SeqLen)
		}
		for pos := 0; pos < cfg.SeqLen; pos++ {
			i := b*cfg.SeqLen + pos
			wantMask := float32(0)
			if pos >= start && pos < end {
				wantMask = 1
			}
			if got.lossMask[i] != wantMask {
				t.Fatalf("row %d pos %d lossMask=%g, want %g for block [%d,%d)", b, pos, got.lossMask[i], wantMask, start, end)
			}
		}
	}
}

type recordingBlockDiffusionSubmitTrainer struct {
	objectiveCalls int
	plainCalls     int
	batch          objectiveBatch
	batchSize      int
	seqLen         int
	lr             float32
}

func (t *recordingBlockDiffusionSubmitTrainer) TrainStepGPU([]int, []int, int, int, float32) (float32, error) {
	return 0, nil
}

func (t *recordingBlockDiffusionSubmitTrainer) SubmitStepGPU([]int, []int, int, int, float32) error {
	t.plainCalls++
	return nil
}

func (t *recordingBlockDiffusionSubmitTrainer) SubmitObjectiveStepGPU(batch objectiveBatch, batchSize, seqLen int, lr float32) error {
	t.objectiveCalls++
	t.batch = batch
	t.batchSize = batchSize
	t.seqLen = seqLen
	t.lr = lr
	return nil
}

func (t *recordingBlockDiffusionSubmitTrainer) CollectLossGPU() (float32, error) { return 0, nil }
func (t *recordingBlockDiffusionSubmitTrainer) FlushGPU() error                  { return nil }
func (t *recordingBlockDiffusionSubmitTrainer) SetQATGPU(string) error           { return nil }
func (t *recordingBlockDiffusionSubmitTrainer) SetWeightGPU(string, []float32) error {
	return nil
}
func (t *recordingBlockDiffusionSubmitTrainer) EvaluateObjectiveGPU(objectiveBatch, int, int) (float32, error) {
	return 0, nil
}
func (t *recordingBlockDiffusionSubmitTrainer) EvaluateGPU([]int, []int, int, int) (float32, error) {
	return 0, nil
}
func (t *recordingBlockDiffusionSubmitTrainer) EvaluatePerTokenGPU([]int, []int, int, int) ([]float32, error) {
	return nil, nil
}
func (t *recordingBlockDiffusionSubmitTrainer) EvaluateLoRATTTGPU([]int, []int, int, int, int, float32, int) (float32, error) {
	return 0, nil
}
func (t *recordingBlockDiffusionSubmitTrainer) CloseTrainer() {}

func TestBlockDiffusionInputsUseObjectiveSubmitPath(t *testing.T) {
	trainer := &recordingBlockDiffusionSubmitTrainer{}
	batch := objectiveBatch{
		x:                   []int{1, 2, 3, 4},
		y:                   []int{1, 2, 3, 4},
		lossMask:            []float32{0, 1, 0, 0},
		diffusionBlockStart: []int32{1},
		diffusionBlockEnd:   []int32{2},
	}
	if err := submitPreparedStepGPU(trainer, batch, 1, 4, 0.125); err != nil {
		t.Fatalf("submitPreparedStepGPU: %v", err)
	}
	if trainer.objectiveCalls != 1 || trainer.plainCalls != 0 {
		t.Fatalf("calls objective=%d plain=%d, want 1/0", trainer.objectiveCalls, trainer.plainCalls)
	}
	if trainer.batchSize != 1 || trainer.seqLen != 4 || trainer.lr != 0.125 {
		t.Fatalf("submitted shape/lr=(%d,%d,%g), want (1,4,0.125)", trainer.batchSize, trainer.seqLen, trainer.lr)
	}
	if !reflect.DeepEqual(trainer.batch.diffusionBlockStart, batch.diffusionBlockStart) ||
		!reflect.DeepEqual(trainer.batch.diffusionBlockEnd, batch.diffusionBlockEnd) {
		t.Fatalf("submitted diffusion block inputs start=%v end=%v, want start=%v end=%v",
			trainer.batch.diffusionBlockStart, trainer.batch.diffusionBlockEnd, batch.diffusionBlockStart, batch.diffusionBlockEnd)
	}
}

func TestPrepareObjectiveBatchHybridExampleCausalRows(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 8
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridMixGranularity = arch.HybridMixGranularityExample
	cfg.Training.HybridCLMFraction = 1
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveMLM
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 20, 21, 22, 23},
		y: []int{11, 12, 13, 14, 21, 22, 23, 24},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveHybridExample)
	if err != nil {
		t.Fatalf("prepare hybrid example causal batch: %v", err)
	}
	if !reflect.DeepEqual(got.x, batch.x) {
		t.Fatalf("hybrid causal x = %v, want %v", got.x, batch.x)
	}
	if !reflect.DeepEqual(got.y, batch.y) {
		t.Fatalf("hybrid causal y = %v, want %v", got.y, batch.y)
	}
	if want := []float32{1, 1, 1, 1, 1, 1, 1, 1}; !reflect.DeepEqual(got.lossMask, want) {
		t.Fatalf("hybrid causal lossMask = %v, want %v", got.lossMask, want)
	}
	if want := []int32{1, 1}; !reflect.DeepEqual(got.attentionCausal, want) {
		t.Fatalf("hybrid causal attention mask = %v, want %v", got.attentionCausal, want)
	}
}

func TestPrepareObjectiveBatchHybridExampleMaskedRows(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 8
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridMixGranularity = arch.HybridMixGranularityExample
	cfg.Training.HybridCLMFraction = 0
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveMNTP
	cfg.Training.MLMMaskProb = 1
	batch := trainBatch{
		x: []int{10, 11, 12, 13, 20, 21, 22, 23},
		y: []int{11, 12, 13, 14, 21, 22, 23, 24},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveHybridExample)
	if err != nil {
		t.Fatalf("prepare hybrid example masked batch: %v", err)
	}
	wantX := []int{10, 9, 9, 9, 20, 9, 9, 9}
	if !reflect.DeepEqual(got.x, wantX) {
		t.Fatalf("hybrid MNTP x = %v, want %v", got.x, wantX)
	}
	wantMask := []float32{1, 1, 1, 0, 1, 1, 1, 0}
	if !reflect.DeepEqual(got.lossMask, wantMask) {
		t.Fatalf("hybrid MNTP lossMask = %v, want %v", got.lossMask, wantMask)
	}
	if !reflect.DeepEqual(got.maskedLossMask, wantMask) {
		t.Fatalf("hybrid MNTP maskedLossMask = %v, want %v", got.maskedLossMask, wantMask)
	}
	if want := []int32{0, 0}; !reflect.DeepEqual(got.attentionCausal, want) {
		t.Fatalf("hybrid MNTP attention mask = %v, want %v", got.attentionCausal, want)
	}
}

func TestPrepareObjectiveBatchSegmentIDsFromBoundaryToken(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 8
	cfg.Training.AttentionSegmentMask = arch.AttentionSegmentMaskBoundaryToken
	cfg.Training.AttentionSegmentBoundaryTokenID = 1
	batch := trainBatch{
		x: []int{10, 1, 11, 12, 1, 20, 1, 21},
		y: []int{1, 11, 12, 1, 20, 1, 21, 22},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepare causal segment batch: %v", err)
	}
	want := []int32{0, 1, 1, 1, 1, 1, 2, 2}
	if !reflect.DeepEqual(got.segmentIDs, want) {
		t.Fatalf("segmentIDs = %v, want %v", got.segmentIDs, want)
	}
}

func TestPrepareObjectiveBatchSegmentIDsUseUnmaskedMLMTokens(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 4
	cfg.Training.AttentionSegmentMask = arch.AttentionSegmentMaskBoundaryToken
	cfg.Training.AttentionSegmentBoundaryTokenID = 1
	cfg.Training.MLMMaskProb = 1
	cfg.Training.MLMMaskTokenProb = 1
	cfg.Training.MLMRandomTokenProb = 0
	cfg.Training.MLMKeptUnchangedProb = 0
	batch := trainBatch{
		x: []int{1, 10, 1, 11},
		y: []int{10, 1, 11, 12},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare MLM segment batch: %v", err)
	}
	if wantX := []int{9, 9, 9, 9}; !reflect.DeepEqual(got.x, wantX) {
		t.Fatalf("MLM x = %v, want %v", got.x, wantX)
	}
	if want := []int32{1, 1, 2, 2}; !reflect.DeepEqual(got.segmentIDs, want) {
		t.Fatalf("segmentIDs = %v, want %v", got.segmentIDs, want)
	}
}

func TestPrepareObjectiveBatchHybridExampleSegmentIDsAndCausalRows(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 8
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridMixGranularity = arch.HybridMixGranularityExample
	cfg.Training.HybridCLMFraction = 1
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveMLM
	cfg.Training.AttentionSegmentMask = arch.AttentionSegmentMaskBoundaryToken
	cfg.Training.AttentionSegmentBoundaryTokenID = 1
	batch := trainBatch{
		x: []int{1, 10, 11, 12, 20, 1, 21, 22},
		y: []int{10, 11, 12, 20, 1, 21, 22, 23},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveHybridExample)
	if err != nil {
		t.Fatalf("prepare hybrid segment batch: %v", err)
	}
	if want := []int32{1, 1, 1, 1, 0, 1, 1, 1}; !reflect.DeepEqual(got.segmentIDs, want) {
		t.Fatalf("segmentIDs = %v, want %v", got.segmentIDs, want)
	}
	if want := []int32{1, 1}; !reflect.DeepEqual(got.attentionCausal, want) {
		t.Fatalf("attentionCausal = %v, want %v", got.attentionCausal, want)
	}
}

func TestHybridExampleObjectiveSelectionDeterministicAndApproximateFraction(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.SeqLen = 4
	cfg.Training.BatchTokens = 64
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridMixGranularity = arch.HybridMixGranularityExample
	cfg.Training.HybridCLMFraction = 0.0625
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveMLM
	cfg.Training.MLMMaskProb = 0
	batch := trainBatch{
		x: make([]int, cfg.Training.BatchTokens),
		y: make([]int, cfg.Training.BatchTokens),
	}
	for i := range batch.x {
		batch.x[i] = i % cfg.VocabSize
		batch.y[i] = (i + 1) % cfg.VocabSize
	}
	if got := objectiveForStep(cfg.Training, 0); got != arch.ObjectiveHybridExample {
		t.Fatalf("objectiveForStep example granularity = %q, want %q", got, arch.ObjectiveHybridExample)
	}
	first := make([]int32, 0, 512)
	total := 0
	causal := 0
	for step := 0; step < 32; step++ {
		got, err := prepareObjectiveBatch(cfg, batch, step, arch.ObjectiveHybridExample)
		if err != nil {
			t.Fatalf("prepare hybrid example step %d: %v", step, err)
		}
		first = append(first, got.attentionCausal...)
		for _, v := range got.attentionCausal {
			total++
			if v > 0 {
				causal++
			}
		}
	}
	second := make([]int32, 0, len(first))
	for step := 0; step < 32; step++ {
		got, err := prepareObjectiveBatch(cfg, batch, step, arch.ObjectiveHybridExample)
		if err != nil {
			t.Fatalf("prepare hybrid example repeat step %d: %v", step, err)
		}
		second = append(second, got.attentionCausal...)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("hybrid example selection is not deterministic")
	}
	ratio := float64(causal) / float64(total)
	if ratio < 0.03 || ratio > 0.10 {
		t.Fatalf("hybrid example causal ratio = %.4f, want near 0.0625 (causal=%d total=%d)", ratio, causal, total)
	}
}

func TestObjectiveForStepHybridDeterministicAndHonorsFractions(t *testing.T) {
	spec := TrainingSpec{
		Objective:                arch.ObjectiveHybrid,
		Seed:                     123,
		HybridCLMFraction:        1,
		HybridSecondaryObjective: arch.ObjectiveMLM,
		MLMMaskProb:              0.15,
		MLMMaskTokenID:           9,
		MLMMaskTokenProb:         0.8,
		MLMRandomTokenProb:       0.1,
		MLMKeptUnchangedProb:     0.1,
		Steps:                    4,
		LR:                       1e-3,
		BatchTokens:              4,
	}
	for step := 0; step < 8; step++ {
		if got := objectiveForStep(spec, step); got != arch.ObjectiveCausal {
			t.Fatalf("hybrid fraction 1 step %d = %q, want causal", step, got)
		}
	}
	spec = parsedHybridObjectiveSpec(t, 0, arch.ObjectiveMLM)
	for step := 0; step < 8; step++ {
		if got := objectiveForStep(spec, step); got != arch.ObjectiveMLM {
			t.Fatalf("hybrid fraction 0 step %d = %q, want mlm", step, got)
		}
	}

	spec.HybridCLMFraction = 0.5
	first := make([]string, 32)
	second := make([]string, 32)
	for step := range first {
		first[step] = objectiveForStep(spec, step)
		second[step] = objectiveForStep(spec, step)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("hybrid sequence not deterministic: %v vs %v", first, second)
	}
}

func TestObjectiveForStepHybridBlockDiffusionSchedule(t *testing.T) {
	cfg := parseHybridBlockDiffusionObjectiveConfig(t)
	if got := cfg.Training.EffectiveHybridCLMFractionForStep(0); got != 1 {
		t.Fatalf("hybrid causal fraction step0=%g, want 1", got)
	}
	if got := cfg.Training.EffectiveHybridCLMFractionForStep(1); got != 0 {
		t.Fatalf("hybrid causal fraction step1=%g, want 0", got)
	}
	if got := objectiveForStep(cfg.Training, 0); got != arch.ObjectiveCausal {
		t.Fatalf("objectiveForStep step0=%q, want causal", got)
	}
	if got := objectiveForStep(cfg.Training, 1); got != arch.ObjectiveBlockDiffusion {
		t.Fatalf("objectiveForStep step1=%q, want block_diffusion", got)
	}

	batch := trainBatch{
		x: []int{10, 11, 12, 13, 20, 21, 22, 23},
		y: []int{11, 12, 13, 14, 21, 22, 23, 24},
	}
	causal, err := prepareObjectiveBatch(cfg, batch, 0, objectiveForStep(cfg.Training, 0))
	if err != nil {
		t.Fatalf("prepare causal hybrid step: %v", err)
	}
	if causal.diffusionBlockStart != nil || causal.diffusionBlockEnd != nil || causal.lossMask != nil {
		t.Fatalf("causal hybrid step carried diffusion/masked inputs: %+v", causal)
	}
	diffusion, err := prepareObjectiveBatch(cfg, batch, 1, objectiveForStep(cfg.Training, 1))
	if err != nil {
		t.Fatalf("prepare diffusion hybrid step: %v", err)
	}
	if len(diffusion.diffusionBlockStart) != 2 || len(diffusion.diffusionBlockEnd) != 2 {
		t.Fatalf("diffusion block vectors lengths start=%d end=%d, want 2", len(diffusion.diffusionBlockStart), len(diffusion.diffusionBlockEnd))
	}
	if diffusion.lossMask == nil {
		t.Fatal("diffusion hybrid step missing lossMask")
	}
}

func objectiveTestConfig() *ArchConfig {
	return &ArchConfig{
		Name:      "objective_test",
		ModelDim:  8,
		VocabSize: 32,
		SeqLen:    4,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}},
		Training: TrainingSpec{
			Steps:                1,
			LR:                   1e-3,
			BatchTokens:          4,
			Seed:                 42,
			MLMMaskProb:          1,
			MLMMaskTokenID:       9,
			MLMMaskTokenProb:     0.8,
			MLMRandomTokenProb:   0.1,
			MLMKeptUnchangedProb: 0.1,
		},
	}
}

func parsedHybridObjectiveSpec(t *testing.T, causalFraction float64, secondary string) TrainingSpec {
	t.Helper()
	raw := []byte(`{
		"name": "objective_step_test",
		"model_dim": 8,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"batch_tokens": 4,
			"seed": 123,
			"objective": "hybrid",
			"mlm_mask_token_id": 9,
			"hybrid_clm_fraction": ` + formatFloatForJSON(causalFraction) + `,
			"hybrid_secondary_objective": "` + secondary + `"
		}
	}`)
	cfg, err := ParseArchConfig(raw, "objective_step_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg.Training
}

func parseHybridBlockDiffusionObjectiveConfig(t *testing.T) *ArchConfig {
	t.Helper()
	raw := []byte(`{
		"name": "hybrid_block_diffusion_objective_test",
		"model_dim": 8,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"batch_tokens": 8,
			"seed": 123,
			"objective": "hybrid",
			"mlm_mask_token_id": 9,
			"hybrid_secondary_objective": "block_diffusion",
			"hybrid_clm_fraction_schedule": [[0,1],[1,0]],
			"diffusion": {"block_size": 2, "min_mask_fraction": 1, "max_mask_fraction": 1}
		}
	}`)
	cfg, err := ParseArchConfig(raw, "hybrid_block_diffusion_objective_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func formatFloatForJSON(v float64) string {
	if v == 0 {
		return "0"
	}
	if v == 1 {
		return "1"
	}
	return "0.5"
}
