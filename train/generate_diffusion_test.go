package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestGenerateDiffusionBatchWiring(t *testing.T) {
	batch, err := diffusionGenerationBatch([]int{4, 99, 99}, []int{1, 2}, 1, 3, 5)
	if err != nil {
		t.Fatalf("diffusionGenerationBatch: %v", err)
	}
	if !reflect.DeepEqual(batch.x, []int{4, 99, 99, 0, 0}) {
		t.Fatalf("x = %v", batch.x)
	}
	if !reflect.DeepEqual(batch.y, batch.x) {
		t.Fatalf("y = %v, want x %v", batch.y, batch.x)
	}
	if !reflect.DeepEqual(batch.lossMask, []float32{0, 1, 1, 0, 0}) {
		t.Fatalf("lossMask = %v", batch.lossMask)
	}
	if !reflect.DeepEqual(batch.diffusionBlockStart, []int32{1}) || !reflect.DeepEqual(batch.diffusionBlockEnd, []int32{3}) {
		t.Fatalf("diffusion block start/end = %v/%v, want [1]/[3]", batch.diffusionBlockStart, batch.diffusionBlockEnd)
	}
}

func TestGenerateDiffusionFakeLogitsCommitBlock(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	evaluator := &fakeDiffusionGenerationEvaluator{
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			return diffusionTestLogitsFromLossMask(batch, cfg.VocabSize, 10, 12)
		},
	}

	result, err := generateDiffusionTokens(cfg, evaluator, []int{3}, 2)
	if err != nil {
		t.Fatalf("generateDiffusionTokens: %v", err)
	}
	if want := []int{3, 11, 12}; !reflect.DeepEqual(result.tokens, want) {
		t.Fatalf("tokens = %v, want %v", result.tokens, want)
	}
	if result.blocks != 1 || result.steps != 1 {
		t.Fatalf("blocks/steps = %d/%d, want 1/1", result.blocks, result.steps)
	}
	if len(result.commits) != 2 {
		t.Fatalf("commits = %#v, want 2 commits", result.commits)
	}
	if len(evaluator.batches) != 1 {
		t.Fatalf("EvaluateObjectiveGPU calls = %d, want 1", len(evaluator.batches))
	}
	gotBatch := evaluator.batches[0]
	if !reflect.DeepEqual(gotBatch.x, []int{3, 63, 63, 0, 0, 0}) {
		t.Fatalf("batch x = %v", gotBatch.x)
	}
	if !reflect.DeepEqual(gotBatch.lossMask, []float32{0, 1, 1, 0, 0, 0}) {
		t.Fatalf("batch lossMask = %v", gotBatch.lossMask)
	}
	if !reflect.DeepEqual(evaluator.readShapes, [][]int{{cfg.SeqLen, cfg.VocabSize}}) {
		t.Fatalf("read shapes = %v", evaluator.readShapes)
	}
}

func TestGenerateDiffusionCommitFloorCarriesCommittedTokens(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	cfg.Training.Diffusion.ConfidenceThreshold = 0.99
	cfg.Training.Diffusion.StepsPerBlock = 2
	cfg.Training.Diffusion.CommitFloor = 1
	evaluator := &fakeDiffusionGenerationEvaluator{
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			return diffusionTestLogitsFromLossMask(batch, cfg.VocabSize, 20, 1)
		},
	}

	result, err := generateDiffusionTokens(cfg, evaluator, []int{5}, 2)
	if err != nil {
		t.Fatalf("generateDiffusionTokens: %v", err)
	}
	if want := []int{5, 21, 22}; !reflect.DeepEqual(result.tokens, want) {
		t.Fatalf("tokens = %v, want %v", result.tokens, want)
	}
	if result.steps != 2 {
		t.Fatalf("steps = %d, want 2", result.steps)
	}
	if len(evaluator.batches) != 2 {
		t.Fatalf("EvaluateObjectiveGPU calls = %d, want 2", len(evaluator.batches))
	}
	if !reflect.DeepEqual(evaluator.batches[1].x, []int{5, 21, 63, 0, 0, 0}) {
		t.Fatalf("second batch x = %v, want committed first token carried forward", evaluator.batches[1].x)
	}
	if !reflect.DeepEqual(evaluator.batches[1].lossMask, []float32{0, 0, 1, 0, 0, 0}) {
		t.Fatalf("second batch lossMask = %v", evaluator.batches[1].lossMask)
	}
}

func TestGenerateDiffusionMultipleBlocksAndSeqLenStop(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	evaluator := &fakeDiffusionGenerationEvaluator{
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			return diffusionTestLogitsFromLossMask(batch, cfg.VocabSize, 30, 12)
		},
	}

	result, err := generateDiffusionTokens(cfg, evaluator, []int{1}, 10)
	if err != nil {
		t.Fatalf("generateDiffusionTokens: %v", err)
	}
	if want := []int{1, 31, 32, 33, 34, 35}; !reflect.DeepEqual(result.tokens, want) {
		t.Fatalf("tokens = %v, want %v", result.tokens, want)
	}
	if result.blocks != 3 || result.steps != 3 {
		t.Fatalf("blocks/steps = %d/%d, want 3/3", result.blocks, result.steps)
	}
	if !result.stoppedAtSeqLen {
		t.Fatal("stoppedAtSeqLen = false, want true")
	}
	if len(evaluator.batches) != 3 {
		t.Fatalf("EvaluateObjectiveGPU calls = %d, want 3", len(evaluator.batches))
	}
	last := evaluator.batches[2]
	if !reflect.DeepEqual(last.diffusionBlockStart, []int32{5}) || !reflect.DeepEqual(last.diffusionBlockEnd, []int32{6}) {
		t.Fatalf("last block start/end = %v/%v, want [5]/[6]", last.diffusionBlockStart, last.diffusionBlockEnd)
	}
}

func TestGenerateDiffusionHybridBlockDiffusionConfig(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveBlockDiffusion
	cfg.Training.HybridCLMFraction = 0.5
	evaluator := &fakeDiffusionGenerationEvaluator{
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			return diffusionTestLogitsFromLossMask(batch, cfg.VocabSize, 40, 12)
		},
	}

	result, err := generateDiffusionTokens(cfg, evaluator, []int{2}, 2)
	if err != nil {
		t.Fatalf("generateDiffusionTokens hybrid block diffusion: %v", err)
	}
	if want := []int{2, 41, 42}; !reflect.DeepEqual(result.tokens, want) {
		t.Fatalf("tokens = %v, want %v", result.tokens, want)
	}
}

func TestGenerateDiffusionOptionsOverrideAndTraceJSONL(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	confidence := 0.99
	evaluator := &fakeDiffusionGenerationEvaluator{
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			return diffusionTestLogitsFromLossMask(batch, cfg.VocabSize, 20, 1)
		},
	}

	result, err := generateDiffusionTokensWithOptions(cfg, evaluator, []int{5}, 2, diffusionGenerationRuntimeOptions{
		stepsPerBlock:       2,
		confidenceThreshold: &confidence,
		commitFloor:         1,
	})
	if err != nil {
		t.Fatalf("generateDiffusionTokensWithOptions: %v", err)
	}
	if result.steps != 2 {
		t.Fatalf("steps=%d, want 2", result.steps)
	}
	if len(result.trace) != 2 {
		t.Fatalf("trace len=%d, want 2", len(result.trace))
	}
	if result.trace[0].Forced != 1 || result.trace[0].Committed != 1 || result.trace[0].UnresolvedBefore != 2 {
		t.Fatalf("first trace = %+v, want one forced commit from two unresolved", result.trace[0])
	}

	tracePath := filepath.Join(t.TempDir(), "trace.jsonl")
	if err := writeDiffusionTrace(tracePath, result.trace); err != nil {
		t.Fatalf("writeDiffusionTrace: %v", err)
	}
	data, err := os.ReadFile(tracePath)
	if err != nil {
		t.Fatalf("ReadFile(trace): %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("trace lines=%d, want 2: %q", len(lines), string(data))
	}
	var entry diffusionSamplerTraceEntry
	if err := json.Unmarshal([]byte(lines[0]), &entry); err != nil {
		t.Fatalf("json.Unmarshal trace: %v", err)
	}
	if entry.Block != 0 || entry.Step != 0 || entry.Forced != 1 {
		t.Fatalf("first trace JSON = %+v", entry)
	}
}

func TestGenerateDiffusionRejectsNonDiffusionConfig(t *testing.T) {
	tests := []struct {
		name      string
		objective string
		secondary string
	}{
		{name: "causal", objective: arch.ObjectiveCausal},
		{name: "ordinary hybrid", objective: arch.ObjectiveHybrid, secondary: arch.ObjectiveMNTP},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testGenerateDiffusionConfig()
			cfg.Training.Objective = tt.objective
			cfg.Training.HybridSecondaryObjective = tt.secondary
			_, err := generateDiffusionTokens(cfg, &fakeDiffusionGenerationEvaluator{}, []int{1}, 1)
			if err == nil {
				t.Fatal("generateDiffusionTokens with non-diffusion objective succeeded, want error")
			}
		})
	}
}

func TestApplyDiffusionGenerationOverridesBounds(t *testing.T) {
	base := func() *arch.DiffusionSpec {
		return &arch.DiffusionSpec{BlockSize: 4, StepsPerBlock: 4, ConfidenceThreshold: 0.8, CommitFloor: 1}
	}
	bad := 1.5
	cases := []struct {
		name string
		opts diffusionGenerationRuntimeOptions
	}{
		{name: "confidence above 1", opts: diffusionGenerationRuntimeOptions{confidenceThreshold: &bad}},
		{name: "commit_floor above block_size", opts: diffusionGenerationRuntimeOptions{commitFloor: 5}},
		{name: "negative steps_per_block", opts: diffusionGenerationRuntimeOptions{stepsPerBlock: -1}},
		{name: "negative temperature", opts: diffusionGenerationRuntimeOptions{temperature: -0.1}},
		{name: "negative top_k", opts: diffusionGenerationRuntimeOptions{topK: -1}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if err := applyDiffusionGenerationOverrides(base(), tt.opts); err == nil {
				t.Fatalf("applyDiffusionGenerationOverrides(%s) succeeded, want error", tt.name)
			}
		})
	}

	spec := base()
	good := 0.5
	if err := applyDiffusionGenerationOverrides(spec, diffusionGenerationRuntimeOptions{
		stepsPerBlock: 7, confidenceThreshold: &good, commitFloor: 3,
	}); err != nil {
		t.Fatalf("applyDiffusionGenerationOverrides(valid): %v", err)
	}
	if spec.StepsPerBlock != 7 || spec.ConfidenceThreshold != 0.5 || spec.CommitFloor != 3 {
		t.Fatalf("overrides not applied: %+v", spec)
	}
}

func testGenerateDiffusionConfig() *ArchConfig {
	return &ArchConfig{
		Name:      "generate_diffusion_test",
		ModelDim:  8,
		VocabSize: 64,
		SeqLen:    6,
		Training: TrainingSpec{
			Objective:        arch.ObjectiveBlockDiffusion,
			BatchTokens:      6,
			MLMMaskTokenID:   63,
			MLMMaskTokenProb: 1,
			Diffusion: &arch.DiffusionSpec{
				BlockSize:           2,
				StepsPerBlock:       2,
				MaxMaskFraction:     1,
				ConfidenceThreshold: 0.9,
				CommitFloor:         1,
			},
		},
	}
}

type fakeDiffusionGenerationEvaluator struct {
	batches        []objectiveBatch
	batchSizes     []int
	seqLens        []int
	readShapes     [][]int
	logitsForBatch func(batch objectiveBatch, call int) []float32
	readCalls      int
}

func (f *fakeDiffusionGenerationEvaluator) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	f.batches = append(f.batches, cloneDiffusionObjectiveBatch(batch))
	f.batchSizes = append(f.batchSizes, batchSize)
	f.seqLens = append(f.seqLens, seqLen)
	return 0, nil
}

func (f *fakeDiffusionGenerationEvaluator) ReadOutput(name string, shape []int) ([]float32, error) {
	if name != "logits" {
		return nil, fmt.Errorf("unexpected output %q", name)
	}
	f.readShapes = append(f.readShapes, append([]int(nil), shape...))
	if f.logitsForBatch == nil {
		return nil, fmt.Errorf("no fake logits configured")
	}
	if len(f.batches) == 0 {
		return nil, fmt.Errorf("ReadOutput before EvaluateObjectiveGPU")
	}
	logits := f.logitsForBatch(f.batches[len(f.batches)-1], f.readCalls)
	f.readCalls++
	return append([]float32(nil), logits...), nil
}

func cloneDiffusionObjectiveBatch(batch objectiveBatch) objectiveBatch {
	return objectiveBatch{
		x:                   append([]int(nil), batch.x...),
		y:                   append([]int(nil), batch.y...),
		lossMask:            append([]float32(nil), batch.lossMask...),
		unmaskedX:           append([]int(nil), batch.unmaskedX...),
		diffusionBlockStart: append([]int32(nil), batch.diffusionBlockStart...),
		diffusionBlockEnd:   append([]int32(nil), batch.diffusionBlockEnd...),
	}
}

func diffusionTestLogitsFromLossMask(batch objectiveBatch, vocab, offset int, high float32) []float32 {
	logits := make([]float32, len(batch.x)*vocab)
	for pos, mask := range batch.lossMask {
		if mask <= 0 {
			continue
		}
		token := offset + pos
		if token >= vocab {
			token = vocab - 1
		}
		logits[pos*vocab+token] = high
	}
	return logits
}
