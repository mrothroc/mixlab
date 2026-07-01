package train

import (
	"fmt"
	"math"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

const rtdGeneratorSalt uint64 = 0x452821e638d01377

func rtdActive(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.MultiheadEnabled() && cfg.Training.RTD != nil
}

// maybeAttachRTDCorruption applies RTD generator corruption when RTD is active.
// It samples generator logits from the active multihead program instead of
// switching to a separate generator program, preserving the MLX compiled
// training-step cache across RTD steps.
func maybeAttachRTDCorruption(
	trainer GPUTrainer,
	cfg *ArchConfig,
	batch trainBatch,
	step int,
	prepared objectiveBatch,
	batchSize, seqLen int,
	objective string,
) (objectiveBatch, error) {
	if !rtdActive(cfg) || objective != arch.ObjectiveMultihead {
		return prepared, nil
	}
	var err error
	if cfg.Training.RTD.DedicatedGeneratorEnabled() {
		prepared, err = attachRTDDedicatedGeneratorCorruption(trainer, cfg, batch, step, prepared, batchSize, seqLen)
	} else {
		prepared, err = attachRTDTiedGeneratorCorruption(trainer, cfg, batch, step, prepared, batchSize, seqLen)
	}
	if err != nil {
		return objectiveBatch{}, fmt.Errorf("prepare step %d RTD corruption: %w", step, err)
	}
	return prepared, nil
}

func multiheadHeadIndexByName(cfg *ArchConfig, name string) int {
	if cfg == nil {
		return -1
	}
	for i, head := range cfg.Training.Heads {
		if head.Name == name {
			return i
		}
	}
	return -1
}

func multiheadRTDHeadIndex(cfg *ArchConfig) int {
	if cfg == nil {
		return -1
	}
	for i, head := range cfg.Training.Heads {
		if head.Objective == arch.ObjectiveRTD {
			return i
		}
	}
	return -1
}

func rtdHeadLogitsOutputName(cfg *ArchConfig) (string, error) {
	idx := multiheadRTDHeadIndex(cfg)
	if idx < 0 {
		return "", fmt.Errorf("config has no multihead objective=%q head", arch.ObjectiveRTD)
	}
	return "head_" + cfg.Training.Heads[idx].Name + "_logits", nil
}

func rtdGeneratorLogitsOutputName(cfg *ArchConfig) (string, error) {
	if cfg == nil || cfg.Training.RTD == nil {
		return "", fmt.Errorf("training.rtd is not configured")
	}
	if strings.TrimSpace(cfg.Training.RTD.GeneratorHead) == "" {
		return "", fmt.Errorf("training.rtd.generator_head is empty")
	}
	return "head_" + cfg.Training.RTD.GeneratorHead + "_logits", nil
}

func prepareRTDGeneratorProbeBatch(cfg *ArchConfig, batch trainBatch, step, need, seqLen int) (objectiveBatch, error) {
	if !rtdActive(cfg) {
		return objectiveBatch{}, fmt.Errorf("training.rtd is not active")
	}
	probeCfg := *cfg
	probeCfg.Training.MLMMaskProb = cfg.Training.RTD.MaskProb
	probeCfg.Training.MLMMaskProbSchedule = nil
	return prepareMultiheadBatch(&probeCfg, batch, step, need, seqLen)
}

func prepareRTDDedicatedGeneratorProbeBatch(cfg *ArchConfig, batch trainBatch, step, need int) (objectiveBatch, error) {
	if !rtdActive(cfg) || !cfg.Training.RTD.DedicatedGeneratorEnabled() {
		return objectiveBatch{}, fmt.Errorf("dedicated training.rtd generator is not active")
	}
	probeCfg := *cfg
	probeCfg.Training.MLMMaskProb = cfg.Training.RTD.MaskProb
	probeCfg.Training.MLMMaskProbSchedule = nil
	return prepareMLMBatch(&probeCfg, batch, step, need)
}

func attachDedicatedGeneratorInputs(prepared *objectiveBatch, probe objectiveBatch) {
	if prepared == nil {
		return
	}
	prepared.rtdGeneratorX = append(prepared.rtdGeneratorX[:0], probe.x...)
	prepared.rtdGeneratorY = append(prepared.rtdGeneratorY[:0], probe.y...)
	prepared.rtdGeneratorLossMask = append(prepared.rtdGeneratorLossMask[:0], probe.lossMask...)
}

func attachRTDTiedGeneratorCorruption(
	trainer GPUTrainer,
	cfg *ArchConfig,
	batch trainBatch,
	step int,
	prepared objectiveBatch,
	batchSize, seqLen int,
) (objectiveBatch, error) {
	if !rtdActive(cfg) {
		return prepared, nil
	}
	need := cfg.Training.BatchTokens
	probe, err := prepareRTDGeneratorProbeBatch(cfg, batch, step, need, seqLen)
	if err != nil {
		return objectiveBatch{}, err
	}
	outputName, err := rtdGeneratorLogitsOutputName(cfg)
	if err != nil {
		return objectiveBatch{}, err
	}
	samples, sampledOnDevice, err := sampleRTDGeneratorReplacements(trainer, probe, batchSize, seqLen, outputName, need, cfg.VocabSize, cfg.Training.RTD.SampleTemperature, deterministicObjectiveSeed(cfg.Training.Seed, step, rtdGeneratorSalt))
	if err != nil {
		return objectiveBatch{}, fmt.Errorf("sample RTD generator replacements: %w", err)
	}
	if sampledOnDevice {
		if err := applyRTDGeneratorSampledCorruption(cfg, batch, seqLen, probe, samples, &prepared); err != nil {
			return objectiveBatch{}, err
		}
		return prepared, nil
	}
	logits, err := evaluateRTDGeneratorLogits(trainer, probe, batchSize, seqLen, outputName, []int{need, cfg.VocabSize})
	if err != nil {
		return objectiveBatch{}, fmt.Errorf("evaluate RTD generator logits: %w", err)
	}
	if err := applyRTDGeneratorCorruption(cfg, batch, step, seqLen, probe, logits, &prepared); err != nil {
		return objectiveBatch{}, err
	}
	return prepared, nil
}

func attachRTDDedicatedGeneratorCorruption(
	trainer GPUTrainer,
	cfg *ArchConfig,
	batch trainBatch,
	step int,
	prepared objectiveBatch,
	batchSize, seqLen int,
) (objectiveBatch, error) {
	if !rtdActive(cfg) {
		return prepared, nil
	}
	need := cfg.Training.BatchTokens
	probe, err := prepareRTDDedicatedGeneratorProbeBatch(cfg, batch, step, need)
	if err != nil {
		return objectiveBatch{}, err
	}
	attachDedicatedGeneratorInputs(&prepared, probe)
	samples, sampledOnDevice, err := sampleRTDGeneratorReplacements(trainer, prepared, batchSize, seqLen, arch.RTDGeneratorLogitsName, need, cfg.VocabSize, cfg.Training.RTD.SampleTemperature, deterministicObjectiveSeed(cfg.Training.Seed, step, rtdGeneratorSalt^0xd1b54a32d192ed03))
	if err != nil {
		return objectiveBatch{}, fmt.Errorf("sample RTD dedicated generator replacements: %w", err)
	}
	if sampledOnDevice {
		if err := applyRTDDedicatedGeneratorSampledCorruption(cfg, batch, seqLen, probe, samples, &prepared); err != nil {
			return objectiveBatch{}, err
		}
		return prepared, nil
	}
	logits, err := evaluateRTDGeneratorLogits(trainer, prepared, batchSize, seqLen, arch.RTDGeneratorLogitsName, []int{need, cfg.VocabSize})
	if err != nil {
		return objectiveBatch{}, fmt.Errorf("evaluate RTD dedicated generator logits: %w", err)
	}
	if err := applyRTDDedicatedGeneratorCorruption(cfg, batch, step, seqLen, probe, logits, &prepared); err != nil {
		return objectiveBatch{}, err
	}
	return prepared, nil
}

func sampleRTDGeneratorReplacements(trainer GPUTrainer, batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, bool, error) {
	if !envTruthy("MIXLAB_RTD_COMPILED_GENERATOR_SAMPLER") {
		if sampler, ok := trainer.(gpuObjectiveCategoricalEagerSampler); ok {
			samples, err := sampler.SampleObjectiveOutputCategoricalEagerGPU(batch, batchSize, seqLen, outputName, rows, vocab, temperature, seed)
			if err != nil {
				return nil, true, err
			}
			if err := validateRTDGeneratorSamples(samples, rows, vocab); err != nil {
				return nil, true, err
			}
			return samples, true, nil
		}
	}
	sampler, ok := trainer.(gpuObjectiveCategoricalSampler)
	if !ok {
		return nil, false, nil
	}
	samples, err := sampler.SampleObjectiveOutputCategoricalGPU(batch, batchSize, seqLen, outputName, rows, vocab, temperature, seed)
	if err != nil {
		return nil, true, err
	}
	if err := validateRTDGeneratorSamples(samples, rows, vocab); err != nil {
		return nil, true, err
	}
	return samples, true, nil
}

func validateRTDGeneratorSamples(samples []int, rows, vocab int) error {
	if len(samples) != rows {
		return fmt.Errorf("RTD generator sampled %d rows, want %d", len(samples), rows)
	}
	for i, sample := range samples {
		if sample < 0 || sample >= vocab {
			return fmt.Errorf("RTD generator sampled token %d at row %d outside vocab [0,%d)", sample, i, vocab)
		}
	}
	return nil
}

func evaluateRTDGeneratorLogits(trainer GPUTrainer, batch objectiveBatch, batchSize, seqLen int, outputName string, shape []int) ([]float32, error) {
	if outputName == "" {
		return nil, fmt.Errorf("RTD generator output name is empty")
	}
	if _, err := trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen); err != nil {
		return nil, err
	}
	out, err := readTrainerOutput(trainer, outputName, shape)
	if err != nil {
		return nil, err
	}
	if err := validateRTDGeneratorLogitsShape(outputName, out, shape); err != nil {
		return nil, err
	}
	return out, nil
}

func validateRTDGeneratorLogitsShape(outputName string, logits []float32, shape []int) error {
	want := 1
	for _, dim := range shape {
		if dim <= 0 {
			return fmt.Errorf("invalid RTD generator output %q shape %v", outputName, shape)
		}
		want *= dim
	}
	if len(logits) != want {
		return fmt.Errorf("RTD generator output %q length=%d, want %d for shape %v", outputName, len(logits), want, shape)
	}
	return nil
}

func applyRTDGeneratorCorruption(cfg *ArchConfig, batch trainBatch, step, seqLen int, probe objectiveBatch, logits []float32, prepared *objectiveBatch) error {
	if cfg == nil || cfg.Training.RTD == nil || prepared == nil {
		return fmt.Errorf("invalid RTD corruption inputs")
	}
	need := cfg.Training.BatchTokens
	if need <= 0 || seqLen <= 0 || need%seqLen != 0 {
		return fmt.Errorf("invalid RTD batch shape need=%d seq_len=%d", need, seqLen)
	}
	if len(batch.x) < need {
		return fmt.Errorf("RTD raw batch has %d tokens, need %d", len(batch.x), need)
	}
	if len(logits) != need*cfg.VocabSize {
		return fmt.Errorf("RTD generator logits length=%d, want %d", len(logits), need*cfg.VocabSize)
	}
	generatorIdx := multiheadHeadIndexByName(cfg, cfg.Training.RTD.GeneratorHead)
	rtdIdx := multiheadRTDHeadIndex(cfg)
	if generatorIdx < 0 || rtdIdx < 0 {
		return fmt.Errorf("RTD generator/head indices not found")
	}
	generatorHead := cfg.Training.Heads[generatorIdx]
	generatorOffset := generatorIdx * need
	rtdOffset := rtdIdx * need
	if len(probe.lossMask) < generatorOffset+need || len(prepared.x) < rtdOffset+need || len(prepared.y) < rtdOffset+need || len(prepared.lossMask) < rtdOffset+need {
		return fmt.Errorf("RTD expanded batch shape mismatch")
	}
	copy(prepared.x[rtdOffset:rtdOffset+need], batch.x[:need])
	for i := 0; i < need; i++ {
		prepared.y[rtdOffset+i] = 1
		prepared.lossMask[rtdOffset+i] = 1
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, rtdGeneratorSalt)
	for pos := 0; pos < need; pos++ {
		if probe.lossMask[generatorOffset+pos] <= 0 {
			continue
		}
		replacementPos := pos
		if generatorHead.Objective == arch.ObjectiveMNTP {
			if pos%seqLen == seqLen-1 {
				continue
			}
			replacementPos = pos + 1
		}
		row := logits[pos*cfg.VocabSize : (pos+1)*cfg.VocabSize]
		sampled, err := sampleTokenFromLogits(row, cfg.Training.RTD.SampleTemperature, rng.Float64())
		if err != nil {
			return fmt.Errorf("sample RTD replacement at position %d: %w", pos, err)
		}
		outPos := rtdOffset + replacementPos
		prepared.x[outPos] = sampled
		if sampled != batch.x[replacementPos] {
			prepared.y[outPos] = 0
		}
	}
	return nil
}

func applyRTDGeneratorSampledCorruption(cfg *ArchConfig, batch trainBatch, seqLen int, probe objectiveBatch, samples []int, prepared *objectiveBatch) error {
	if cfg == nil || cfg.Training.RTD == nil || prepared == nil {
		return fmt.Errorf("invalid RTD sampled corruption inputs")
	}
	need := cfg.Training.BatchTokens
	if need <= 0 || seqLen <= 0 || need%seqLen != 0 {
		return fmt.Errorf("invalid RTD batch shape need=%d seq_len=%d", need, seqLen)
	}
	if len(batch.x) < need {
		return fmt.Errorf("RTD raw batch has %d tokens, need %d", len(batch.x), need)
	}
	if len(samples) < need {
		return fmt.Errorf("RTD generator samples length=%d, want at least %d", len(samples), need)
	}
	generatorIdx := multiheadHeadIndexByName(cfg, cfg.Training.RTD.GeneratorHead)
	rtdIdx := multiheadRTDHeadIndex(cfg)
	if generatorIdx < 0 || rtdIdx < 0 {
		return fmt.Errorf("RTD generator/head indices not found")
	}
	generatorHead := cfg.Training.Heads[generatorIdx]
	generatorOffset := generatorIdx * need
	rtdOffset := rtdIdx * need
	if len(probe.lossMask) < generatorOffset+need || len(prepared.x) < rtdOffset+need || len(prepared.y) < rtdOffset+need || len(prepared.lossMask) < rtdOffset+need {
		return fmt.Errorf("RTD expanded batch shape mismatch")
	}
	copy(prepared.x[rtdOffset:rtdOffset+need], batch.x[:need])
	for i := 0; i < need; i++ {
		prepared.y[rtdOffset+i] = 1
		prepared.lossMask[rtdOffset+i] = 1
	}
	for pos := 0; pos < need; pos++ {
		if probe.lossMask[generatorOffset+pos] <= 0 {
			continue
		}
		replacementPos := pos
		if generatorHead.Objective == arch.ObjectiveMNTP {
			if pos%seqLen == seqLen-1 {
				continue
			}
			replacementPos = pos + 1
		}
		sampled := samples[pos]
		if sampled < 0 || sampled >= cfg.VocabSize {
			return fmt.Errorf("RTD generator sampled token %d at position %d outside vocab [0,%d)", sampled, pos, cfg.VocabSize)
		}
		outPos := rtdOffset + replacementPos
		prepared.x[outPos] = sampled
		if sampled != batch.x[replacementPos] {
			prepared.y[outPos] = 0
		}
	}
	return nil
}

func applyRTDDedicatedGeneratorCorruption(cfg *ArchConfig, batch trainBatch, step, seqLen int, probe objectiveBatch, logits []float32, prepared *objectiveBatch) error {
	if cfg == nil || cfg.Training.RTD == nil || prepared == nil {
		return fmt.Errorf("invalid dedicated RTD corruption inputs")
	}
	need := cfg.Training.BatchTokens
	if need <= 0 || seqLen <= 0 || need%seqLen != 0 {
		return fmt.Errorf("invalid RTD batch shape need=%d seq_len=%d", need, seqLen)
	}
	if len(batch.x) < need {
		return fmt.Errorf("RTD raw batch has %d tokens, need %d", len(batch.x), need)
	}
	if len(probe.lossMask) < need {
		return fmt.Errorf("RTD dedicated generator probe loss mask has %d entries, need %d", len(probe.lossMask), need)
	}
	if len(logits) != need*cfg.VocabSize {
		return fmt.Errorf("RTD dedicated generator logits length=%d, want %d", len(logits), need*cfg.VocabSize)
	}
	rtdIdx := multiheadRTDHeadIndex(cfg)
	if rtdIdx < 0 {
		return fmt.Errorf("RTD head index not found")
	}
	rtdOffset := rtdIdx * need
	if len(prepared.x) < rtdOffset+need || len(prepared.y) < rtdOffset+need || len(prepared.lossMask) < rtdOffset+need {
		return fmt.Errorf("RTD expanded batch shape mismatch")
	}
	copy(prepared.x[rtdOffset:rtdOffset+need], batch.x[:need])
	for i := 0; i < need; i++ {
		prepared.y[rtdOffset+i] = 1
		prepared.lossMask[rtdOffset+i] = 1
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, rtdGeneratorSalt^0xd1b54a32d192ed03)
	for pos := 0; pos < need; pos++ {
		if probe.lossMask[pos] <= 0 {
			continue
		}
		row := logits[pos*cfg.VocabSize : (pos+1)*cfg.VocabSize]
		sampled, err := sampleTokenFromLogits(row, cfg.Training.RTD.SampleTemperature, rng.Float64())
		if err != nil {
			return fmt.Errorf("sample dedicated RTD replacement at position %d: %w", pos, err)
		}
		outPos := rtdOffset + pos
		prepared.x[outPos] = sampled
		if sampled != batch.x[pos] {
			prepared.y[outPos] = 0
		}
	}
	return nil
}

func applyRTDDedicatedGeneratorSampledCorruption(cfg *ArchConfig, batch trainBatch, seqLen int, probe objectiveBatch, samples []int, prepared *objectiveBatch) error {
	if cfg == nil || cfg.Training.RTD == nil || prepared == nil {
		return fmt.Errorf("invalid dedicated RTD sampled corruption inputs")
	}
	need := cfg.Training.BatchTokens
	if need <= 0 || seqLen <= 0 || need%seqLen != 0 {
		return fmt.Errorf("invalid RTD batch shape need=%d seq_len=%d", need, seqLen)
	}
	if len(batch.x) < need {
		return fmt.Errorf("RTD raw batch has %d tokens, need %d", len(batch.x), need)
	}
	if len(probe.lossMask) < need {
		return fmt.Errorf("RTD dedicated generator probe loss mask has %d entries, need %d", len(probe.lossMask), need)
	}
	if len(samples) < need {
		return fmt.Errorf("RTD dedicated generator samples length=%d, want at least %d", len(samples), need)
	}
	rtdIdx := multiheadRTDHeadIndex(cfg)
	if rtdIdx < 0 {
		return fmt.Errorf("RTD head index not found")
	}
	rtdOffset := rtdIdx * need
	if len(prepared.x) < rtdOffset+need || len(prepared.y) < rtdOffset+need || len(prepared.lossMask) < rtdOffset+need {
		return fmt.Errorf("RTD expanded batch shape mismatch")
	}
	copy(prepared.x[rtdOffset:rtdOffset+need], batch.x[:need])
	for i := 0; i < need; i++ {
		prepared.y[rtdOffset+i] = 1
		prepared.lossMask[rtdOffset+i] = 1
	}
	for pos := 0; pos < need; pos++ {
		if probe.lossMask[pos] <= 0 {
			continue
		}
		sampled := samples[pos]
		if sampled < 0 || sampled >= cfg.VocabSize {
			return fmt.Errorf("RTD dedicated generator sampled token %d at position %d outside vocab [0,%d)", sampled, pos, cfg.VocabSize)
		}
		outPos := rtdOffset + pos
		prepared.x[outPos] = sampled
		if sampled != batch.x[pos] {
			prepared.y[outPos] = 0
		}
	}
	return nil
}

func sampleTokenFromLogits(logits []float32, temperature float64, u float64) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty logits")
	}
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return 0, fmt.Errorf("invalid temperature=%g", temperature)
	}
	positiveInf := make([]int, 0)
	for i, v := range logits {
		if math.IsInf(float64(v), 1) {
			positiveInf = append(positiveInf, i)
		}
	}
	if len(positiveInf) > 0 {
		return positiveInf[uniformSampleIndex(len(positiveInf), u)], nil
	}
	maxVal := math.Inf(-1)
	for _, v := range logits {
		scaled := float64(v) / temperature
		if !math.IsNaN(scaled) && !math.IsInf(scaled, 0) && scaled > maxVal {
			maxVal = scaled
		}
	}
	if math.IsInf(maxVal, -1) {
		return uniformSampleIndex(len(logits), u), nil
	}
	total := 0.0
	for _, v := range logits {
		scaled := float64(v)/temperature - maxVal
		if math.IsNaN(scaled) || math.IsInf(scaled, 0) {
			continue
		}
		total += math.Exp(scaled)
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return uniformSampleIndex(len(logits), u), nil
	}
	if u < 0 {
		u = 0
	}
	if u >= 1 {
		u = math.Nextafter(1, 0)
	}
	threshold := u * total
	cumulative := 0.0
	lastFinite := -1
	for i, v := range logits {
		scaled := float64(v)/temperature - maxVal
		if math.IsNaN(scaled) || math.IsInf(scaled, 0) {
			continue
		}
		lastFinite = i
		cumulative += math.Exp(scaled)
		if cumulative >= threshold {
			return i, nil
		}
	}
	if lastFinite >= 0 {
		return lastFinite, nil
	}
	return uniformSampleIndex(len(logits), u), nil
}

func uniformSampleIndex(n int, u float64) int {
	if n <= 1 {
		return 0
	}
	if u < 0 || math.IsNaN(u) {
		u = 0
	}
	if u >= 1 {
		u = math.Nextafter(1, 0)
	}
	idx := int(u * float64(n))
	if idx < 0 {
		return 0
	}
	if idx >= n {
		return n - 1
	}
	return idx
}
