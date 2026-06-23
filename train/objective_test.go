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

func formatFloatForJSON(v float64) string {
	if v == 0 {
		return "0"
	}
	if v == 1 {
		return "1"
	}
	return "0.5"
}
