package train

import (
	"math"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
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

func TestRTDGeneratorCorruptionForMNTP(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": "tied", "generator_head": "scorer", "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	probe, err := prepareRTDGeneratorProbeBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareRTDGeneratorProbeBatch: %v", err)
	}
	logits := make([]float32, cfg.Training.BatchTokens*cfg.VocabSize)
	for i := range logits {
		logits[i] = -100
	}
	for row := 0; row < cfg.Training.BatchTokens; row++ {
		sample := 9
		if row == 0 {
			sample = raw.x[1] // Same-token sample should stay labeled original.
		}
		logits[row*cfg.VocabSize+sample] = 100
	}
	if err := applyRTDGeneratorCorruption(cfg, raw, 3, cfg.SeqLen, probe, logits, &prepared); err != nil {
		t.Fatalf("applyRTDGeneratorCorruption: %v", err)
	}
	rtdOffset := cfg.Training.BatchTokens
	if !reflect.DeepEqual(prepared.diffusionBlockStart[2:], []int32{0, 0}) || !reflect.DeepEqual(prepared.diffusionBlockEnd[2:], []int32{4, 4}) {
		t.Fatalf("RTD boundaries=%v/%v, want full bidirectional", prepared.diffusionBlockStart[2:], prepared.diffusionBlockEnd[2:])
	}
	if prepared.x[rtdOffset] != 1 || prepared.y[rtdOffset] != 1 {
		t.Fatalf("position 0 x/y=%d/%d, want original label", prepared.x[rtdOffset], prepared.y[rtdOffset])
	}
	if prepared.x[rtdOffset+1] != 2 || prepared.y[rtdOffset+1] != 1 {
		t.Fatalf("same-token replacement x/y=%d/%d, want original label", prepared.x[rtdOffset+1], prepared.y[rtdOffset+1])
	}
	for _, pos := range []int{2, 3, 5, 6, 7} {
		if prepared.x[rtdOffset+pos] != 9 || prepared.y[rtdOffset+pos] != 0 {
			t.Fatalf("replacement pos %d x/y=%d/%d, want 9/0", pos, prepared.x[rtdOffset+pos], prepared.y[rtdOffset+pos])
		}
	}
	for i := 0; i < cfg.Training.BatchTokens; i++ {
		if prepared.lossMask[rtdOffset+i] != 1 {
			t.Fatalf("RTD loss mask[%d]=%g, want 1", i, prepared.lossMask[rtdOffset+i])
		}
	}
}

func TestRTDGeneratorSampledCorruptionForMNTP(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": "tied", "generator_head": "scorer", "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	probe, err := prepareRTDGeneratorProbeBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareRTDGeneratorProbeBatch: %v", err)
	}
	samples := []int{raw.x[1], 9, 9, 9, 9, 9, 9, 9}
	if err := applyRTDGeneratorSampledCorruption(cfg, raw, cfg.SeqLen, probe, samples, &prepared); err != nil {
		t.Fatalf("applyRTDGeneratorSampledCorruption: %v", err)
	}
	rtdOffset := cfg.Training.BatchTokens
	if prepared.x[rtdOffset] != 1 || prepared.y[rtdOffset] != 1 {
		t.Fatalf("position 0 x/y=%d/%d, want original label", prepared.x[rtdOffset], prepared.y[rtdOffset])
	}
	if prepared.x[rtdOffset+1] != 2 || prepared.y[rtdOffset+1] != 1 {
		t.Fatalf("same-token replacement x/y=%d/%d, want original label", prepared.x[rtdOffset+1], prepared.y[rtdOffset+1])
	}
	for _, pos := range []int{2, 3, 5, 6, 7} {
		if prepared.x[rtdOffset+pos] != 9 || prepared.y[rtdOffset+pos] != 0 {
			t.Fatalf("replacement pos %d x/y=%d/%d, want 9/0", pos, prepared.x[rtdOffset+pos], prepared.y[rtdOffset+pos])
		}
	}
}

type rtdSamplerOnlyTrainer struct {
	samples    []int
	calls      int
	batch      objectiveBatch
	batchSize  int
	seqLen     int
	outputName string
	rows       int
	vocab      int
}

func (t *rtdSamplerOnlyTrainer) SampleObjectiveOutputCategoricalGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, _ float64, _ uint64) ([]int, error) {
	t.calls++
	t.batch = cloneObjectiveBatchForRTDTest(batch)
	t.batchSize = batchSize
	t.seqLen = seqLen
	t.outputName = outputName
	t.rows = rows
	t.vocab = vocab
	return append([]int(nil), t.samples...), nil
}

func (t *rtdSamplerOnlyTrainer) TrainStepGPU([]int, []int, int, int, float32) (float32, error) {
	return 0, nil
}
func (t *rtdSamplerOnlyTrainer) SubmitStepGPU([]int, []int, int, int, float32) error { return nil }
func (t *rtdSamplerOnlyTrainer) CollectLossGPU() (float32, error)                    { return 0, nil }
func (t *rtdSamplerOnlyTrainer) FlushGPU() error                                     { return nil }
func (t *rtdSamplerOnlyTrainer) SetQATGPU(string) error                              { return nil }
func (t *rtdSamplerOnlyTrainer) SetWeightGPU(string, []float32) error                { return nil }
func (t *rtdSamplerOnlyTrainer) EvaluateObjectiveGPU(objectiveBatch, int, int) (float32, error) {
	return 0, nil
}
func (t *rtdSamplerOnlyTrainer) EvaluateGPU([]int, []int, int, int) (float32, error) {
	return 0, nil
}
func (t *rtdSamplerOnlyTrainer) EvaluatePerTokenGPU([]int, []int, int, int) ([]float32, error) {
	return nil, nil
}
func (t *rtdSamplerOnlyTrainer) EvaluateLoRATTTGPU([]int, []int, int, int, int, float32, int) (float32, error) {
	return 0, nil
}
func (t *rtdSamplerOnlyTrainer) CloseTrainer() {}

type rtdDualSamplerTrainer struct {
	rtdSamplerOnlyTrainer
	eagerSamples []int
	eagerCalls   int
}

func (t *rtdDualSamplerTrainer) SampleObjectiveOutputCategoricalEagerGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, _ float64, _ uint64) ([]int, error) {
	t.eagerCalls++
	t.batch = cloneObjectiveBatchForRTDTest(batch)
	t.batchSize = batchSize
	t.seqLen = seqLen
	t.outputName = outputName
	t.rows = rows
	t.vocab = vocab
	return append([]int(nil), t.eagerSamples...), nil
}

func cloneObjectiveBatchForRTDTest(batch objectiveBatch) objectiveBatch {
	return objectiveBatch{
		x:                    append([]int(nil), batch.x...),
		y:                    append([]int(nil), batch.y...),
		lossMask:             append([]float32(nil), batch.lossMask...),
		diffusionBlockStart:  append([]int32(nil), batch.diffusionBlockStart...),
		diffusionBlockEnd:    append([]int32(nil), batch.diffusionBlockEnd...),
		rtdGeneratorX:        append([]int(nil), batch.rtdGeneratorX...),
		rtdGeneratorY:        append([]int(nil), batch.rtdGeneratorY...),
		rtdGeneratorLossMask: append([]float32(nil), batch.rtdGeneratorLossMask...),
	}
}

func TestRTDCorruptionPrefersEagerVectorizedSampler(t *testing.T) {
	t.Setenv("MIXLAB_RTD_COMPILED_GENERATOR_SAMPLER", "")
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": {"type": "dedicated", "model_dim": 8, "layers": 1, "heads": 2}, "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	trainer := &rtdDualSamplerTrainer{
		rtdSamplerOnlyTrainer: rtdSamplerOnlyTrainer{samples: []int{7, 7, 7, 7, 7, 7, 7, 7}},
		eagerSamples:          []int{raw.x[0], 9, 9, 9, 9, 9, 9, 9},
	}
	got, err := maybeAttachRTDCorruption(trainer, cfg, raw, 3, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("maybeAttachRTDCorruption: %v", err)
	}
	if trainer.eagerCalls != 1 || trainer.calls != 0 {
		t.Fatalf("sampler calls eager=%d compiled=%d, want eager only", trainer.eagerCalls, trainer.calls)
	}
	rtdOffset := cfg.Training.BatchTokens
	if got.x[rtdOffset+1] != 9 || got.y[rtdOffset+1] != 0 {
		t.Fatalf("RTD corruption came from wrong sampler x/y=%d/%d, want eager sample 9/0", got.x[rtdOffset+1], got.y[rtdOffset+1])
	}
}

func TestRTDCorruptionCanOptIntoCompiledSampler(t *testing.T) {
	t.Setenv("MIXLAB_RTD_COMPILED_GENERATOR_SAMPLER", "1")
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": {"type": "dedicated", "model_dim": 8, "layers": 1, "heads": 2}, "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	trainer := &rtdDualSamplerTrainer{
		rtdSamplerOnlyTrainer: rtdSamplerOnlyTrainer{samples: []int{raw.x[0], 7, 7, 7, 7, 7, 7, 7}},
		eagerSamples:          []int{9, 9, 9, 9, 9, 9, 9, 9},
	}
	got, err := maybeAttachRTDCorruption(trainer, cfg, raw, 3, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("maybeAttachRTDCorruption: %v", err)
	}
	if trainer.calls != 1 || trainer.eagerCalls != 0 {
		t.Fatalf("sampler calls eager=%d compiled=%d, want compiled only", trainer.eagerCalls, trainer.calls)
	}
	rtdOffset := cfg.Training.BatchTokens
	if got.x[rtdOffset+1] != 7 || got.y[rtdOffset+1] != 0 {
		t.Fatalf("RTD corruption came from wrong sampler x/y=%d/%d, want compiled sample 7/0", got.x[rtdOffset+1], got.y[rtdOffset+1])
	}
}

func TestRTDCorruptionSamplesFromActiveProgramWithoutSwitcher(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": "tied", "generator_head": "scorer", "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	trainer := &rtdSamplerOnlyTrainer{samples: []int{raw.x[1], 9, 9, 9, 9, 9, 9, 9}}
	got, err := maybeAttachRTDCorruption(trainer, cfg, raw, 3, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("maybeAttachRTDCorruption: %v", err)
	}
	if trainer.calls != 1 {
		t.Fatalf("sampler calls=%d, want 1", trainer.calls)
	}
	if trainer.outputName != "head_scorer_logits" {
		t.Fatalf("sample output=%q, want head_scorer_logits", trainer.outputName)
	}
	if trainer.rows != cfg.Training.BatchTokens || trainer.vocab != cfg.VocabSize {
		t.Fatalf("sample shape rows=%d vocab=%d, want %d/%d", trainer.rows, trainer.vocab, cfg.Training.BatchTokens, cfg.VocabSize)
	}
	rtdOffset := cfg.Training.BatchTokens
	if got.x[rtdOffset+1] != 2 || got.y[rtdOffset+1] != 1 {
		t.Fatalf("same-token replacement x/y=%d/%d, want original", got.x[rtdOffset+1], got.y[rtdOffset+1])
	}
	if got.x[rtdOffset+2] != 9 || got.y[rtdOffset+2] != 0 {
		t.Fatalf("replacement x/y=%d/%d, want 9/0", got.x[rtdOffset+2], got.y[rtdOffset+2])
	}
}

func TestRTDDedicatedGeneratorCorruptionAndInputs(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": {"type": "dedicated", "model_dim": 8, "layers": 1, "heads": 2}, "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	probe, err := prepareRTDDedicatedGeneratorProbeBatch(cfg, raw, 3, cfg.Training.BatchTokens)
	if err != nil {
		t.Fatalf("prepareRTDDedicatedGeneratorProbeBatch: %v", err)
	}
	attachDedicatedGeneratorInputs(&prepared, probe)
	if !reflect.DeepEqual(prepared.rtdGeneratorX, probe.x) ||
		!reflect.DeepEqual(prepared.rtdGeneratorY, probe.y) ||
		!reflect.DeepEqual(prepared.rtdGeneratorLossMask, probe.lossMask) {
		t.Fatalf("dedicated generator inputs not attached")
	}
	logits := make([]float32, cfg.Training.BatchTokens*cfg.VocabSize)
	for i := range logits {
		logits[i] = -100
	}
	for row := 0; row < cfg.Training.BatchTokens; row++ {
		sample := 9
		if row == 0 {
			sample = raw.x[0] // Same-token sample should stay labeled original.
		}
		logits[row*cfg.VocabSize+sample] = 100
	}
	if err := applyRTDDedicatedGeneratorCorruption(cfg, raw, 3, cfg.SeqLen, probe, logits, &prepared); err != nil {
		t.Fatalf("applyRTDDedicatedGeneratorCorruption: %v", err)
	}
	rtdOffset := cfg.Training.BatchTokens
	if prepared.x[rtdOffset] != 1 || prepared.y[rtdOffset] != 1 {
		t.Fatalf("same-token replacement x/y=%d/%d, want original label", prepared.x[rtdOffset], prepared.y[rtdOffset])
	}
	for pos := 1; pos < cfg.Training.BatchTokens; pos++ {
		if prepared.x[rtdOffset+pos] != 9 || prepared.y[rtdOffset+pos] != 0 {
			t.Fatalf("replacement pos %d x/y=%d/%d, want 9/0", pos, prepared.x[rtdOffset+pos], prepared.y[rtdOffset+pos])
		}
	}
	for i := 0; i < cfg.Training.BatchTokens; i++ {
		if prepared.lossMask[rtdOffset+i] != 1 {
			t.Fatalf("RTD loss mask[%d]=%g, want 1", i, prepared.lossMask[rtdOffset+i])
		}
	}
}

func TestRTDDedicatedCorruptionSamplesFromActiveProgramWithoutSwitcher(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": {"type": "dedicated", "model_dim": 8, "layers": 1, "heads": 2}, "mask_prob": 1.0, "sample_temperature": 1.0},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 9, 6, 7, 8, 9},
	}
	prepared, err := prepareMultiheadBatch(cfg, raw, 3, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareMultiheadBatch: %v", err)
	}
	trainer := &rtdSamplerOnlyTrainer{samples: []int{raw.x[0], 9, 9, 9, 9, 9, 9, 9}}
	got, err := maybeAttachRTDCorruption(trainer, cfg, raw, 3, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("maybeAttachRTDCorruption: %v", err)
	}
	if trainer.calls != 1 {
		t.Fatalf("sampler calls=%d, want 1", trainer.calls)
	}
	if trainer.outputName != arch.RTDGeneratorLogitsName {
		t.Fatalf("sample output=%q, want %s", trainer.outputName, arch.RTDGeneratorLogitsName)
	}
	if len(trainer.batch.rtdGeneratorX) != cfg.Training.BatchTokens ||
		len(trainer.batch.rtdGeneratorY) != cfg.Training.BatchTokens ||
		len(trainer.batch.rtdGeneratorLossMask) != cfg.Training.BatchTokens {
		t.Fatalf("dedicated generator inputs lengths x=%d y=%d mask=%d, want %d",
			len(trainer.batch.rtdGeneratorX), len(trainer.batch.rtdGeneratorY), len(trainer.batch.rtdGeneratorLossMask), cfg.Training.BatchTokens)
	}
	rtdOffset := cfg.Training.BatchTokens
	if got.x[rtdOffset] != 1 || got.y[rtdOffset] != 1 {
		t.Fatalf("same-token replacement x/y=%d/%d, want original", got.x[rtdOffset], got.y[rtdOffset])
	}
	if got.x[rtdOffset+1] != 9 || got.y[rtdOffset+1] != 0 {
		t.Fatalf("replacement x/y=%d/%d, want 9/0", got.x[rtdOffset+1], got.y[rtdOffset+1])
	}
}

func TestRTDSampleTokenFromLogitsHandlesNonFiniteRows(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		u      float64
		want   int
	}{
		{
			name:   "all nan falls back to uniform",
			logits: []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN())},
			u:      0.75,
			want:   3,
		},
		{
			name:   "all negative infinity falls back to uniform",
			logits: []float32{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))},
			u:      0.25,
			want:   1,
		},
		{
			name:   "finite logits ignore nan",
			logits: []float32{float32(math.NaN()), -1, 3, float32(math.NaN())},
			u:      0.99,
			want:   2,
		},
		{
			name:   "positive infinity dominates",
			logits: []float32{1, float32(math.Inf(1)), 0, float32(math.Inf(1))},
			u:      0.75,
			want:   3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := sampleTokenFromLogits(tt.logits, 1.0, tt.u)
			if err != nil {
				t.Fatalf("sampleTokenFromLogits: %v", err)
			}
			if got != tt.want {
				t.Fatalf("sampleTokenFromLogits=%d, want %d", got, tt.want)
			}
		})
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

func TestScoreElectraReadsDetectorOutput(t *testing.T) {
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"rtd": {"generator": "tied", "generator_head": "scorer"},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	eval := &fakeDiffusionGenerationEvaluator{
		expectedOutput: "head_detector_logits",
		logitsForBatch: func(batch objectiveBatch, call int) []float32 {
			out := make([]float32, 8)
			for i := range out {
				out[i] = float32(i - 3)
			}
			return out
		},
	}
	got, err := scoreElectraTokens(cfg, eval, []int{4, 5, 6, 7}, 1, 2)
	if err != nil {
		t.Fatalf("scoreElectraTokens: %v", err)
	}
	want := []float64{logSigmoid(-2), logSigmoid(-1), logSigmoid(0)}
	if len(got) != len(want) {
		t.Fatalf("scores len=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Fatalf("score[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if !reflect.DeepEqual(eval.readNames, []string{"head_detector_logits"}) {
		t.Fatalf("read names=%v", eval.readNames)
	}
	if !reflect.DeepEqual(eval.batchSizes, []int{4}) {
		t.Fatalf("batch sizes=%v, want expanded [4]", eval.batchSizes)
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
