package train

import (
	"reflect"
	"testing"
)

func TestPrepareMultiheadBatchExpandsRowsAndBoundaries(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2, "min_mask_fraction": 1.0, "max_mask_fraction": 1.0}}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	batch, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	if batch.batchSizeOverride != 4 {
		t.Fatalf("batchSizeOverride=%d, want 4", batch.batchSizeOverride)
	}
	if len(batch.x) != 16 || len(batch.y) != 16 || len(batch.lossMask) != 16 {
		t.Fatalf("expanded lengths x/y/mask=%d/%d/%d, want 16", len(batch.x), len(batch.y), len(batch.lossMask))
	}
	if !reflect.DeepEqual(batch.diffusionBlockStart[:2], []int32{0, 0}) || !reflect.DeepEqual(batch.diffusionBlockEnd[:2], []int32{4, 4}) {
		t.Fatalf("scorer boundaries=%v/%v, want full bidirectional rows", batch.diffusionBlockStart[:2], batch.diffusionBlockEnd[:2])
	}
	for _, v := range batch.diffusionTimestep[:2] {
		if v != 0 {
			t.Fatalf("scorer timestep=%v, want zeros", batch.diffusionTimestep[:2])
		}
	}
	for row := 2; row < 4; row++ {
		start, end := batch.diffusionBlockStart[row], batch.diffusionBlockEnd[row]
		if end <= start || end-start > 2 {
			t.Fatalf("denoiser row %d block=[%d,%d), want active block length <=2", row, start, end)
		}
		if batch.diffusionTimestep[row] <= 0 || batch.diffusionTimestep[row] > 1 {
			t.Fatalf("denoiser row %d timestep=%g, want (0,1]", row, batch.diffusionTimestep[row])
		}
	}
}

func TestExpandBatchForMultiheadDiffusionUsesDenoiserRows(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2}}
		]
	}`)
	base, err := diffusionScoringBatch([]int{3, 4, 5, 6}, []int{1, 3}, 4, 31, 2, 2)
	if err != nil {
		t.Fatalf("diffusionScoringBatch: %v", err)
	}
	expanded := expandBatchForMultiheadDiffusion(cfg, base, 2, 4)
	if expanded.batchSizeOverride != 4 {
		t.Fatalf("batchSizeOverride=%d, want 4", expanded.batchSizeOverride)
	}
	if len(expanded.x) != 16 {
		t.Fatalf("expanded x len=%d, want 16", len(expanded.x))
	}
	if !reflect.DeepEqual(expanded.diffusionBlockStart, []int32{0, 0, 0, 2}) ||
		!reflect.DeepEqual(expanded.diffusionBlockEnd, []int32{4, 4, 2, 4}) {
		t.Fatalf("expanded boundaries=%v/%v", expanded.diffusionBlockStart, expanded.diffusionBlockEnd)
	}
	if diffusionLogitsOutputName(cfg) != "head_denoiser_logits" {
		t.Fatalf("diffusion output=%q, want head_denoiser_logits", diffusionLogitsOutputName(cfg))
	}
}

func TestScoreDiffusionMultiheadReadsDenoiserOutput(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2}}
		]
	}`)
	eval := &fakeDiffusionGenerationEvaluator{
		expectedOutput: "head_denoiser_logits",
		logitsForBatch: scoreDiffusionTestLogits(cfg.SeqLen, cfg.VocabSize),
	}
	got, err := scoreDiffusionTokens(cfg, eval, []int{4, 5, 6, 7}, 1, 2)
	if err != nil {
		t.Fatalf("scoreDiffusionTokens: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("scores len=%d, want 3", len(got))
	}
	if !reflect.DeepEqual(eval.readNames, []string{"head_denoiser_logits", "head_denoiser_logits"}) {
		t.Fatalf("read names=%v", eval.readNames)
	}
	if !reflect.DeepEqual(eval.batchSizes, []int{4, 4}) {
		t.Fatalf("batch sizes=%v, want expanded [4 4]", eval.batchSizes)
	}
}

func TestGenerateDiffusionMultiheadReadsDenoiserOutput(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 4,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2, "steps_per_block": 1, "commit_floor": 1}}
		]
	}`)
	eval := &fakeDiffusionGenerationEvaluator{
		expectedOutput: "head_denoiser_logits",
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			return diffusionTestLogitsFromLossMask(batch, cfg.VocabSize, 10, 12)
		},
	}
	_, err := generateDiffusionTokens(cfg, eval, []int{3}, 2)
	if err != nil {
		t.Fatalf("generateDiffusionTokens: %v", err)
	}
	if !reflect.DeepEqual(eval.readNames, []string{"head_denoiser_logits"}) {
		t.Fatalf("read names=%v", eval.readNames)
	}
	if !reflect.DeepEqual(eval.batchSizes, []int{2}) {
		t.Fatalf("batch sizes=%v, want expanded [2]", eval.batchSizes)
	}
}

func parseTrainMultiheadConfig(t *testing.T, body string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name": "train_multihead_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		`+body+`
	}`), "train_multihead_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}
