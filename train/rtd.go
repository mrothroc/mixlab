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

// rtdGeneratorPrograms builds and caches the dropout-free forward program used
// to sample RTD generator replacement tokens, keyed by the same
// training program cache key as the main step programs.
type rtdGeneratorPrograms struct {
	cfg                       *ArchConfig
	recurrencePhasesScheduled bool
	cache                     map[trainingProgramCacheKey]*arch.Program
}

func newRTDGeneratorPrograms(cfg *ArchConfig, recurrencePhasesScheduled bool) *rtdGeneratorPrograms {
	return &rtdGeneratorPrograms{
		cfg:                       cfg,
		recurrencePhasesScheduled: recurrencePhasesScheduled,
		cache:                     make(map[trainingProgramCacheKey]*arch.Program),
	}
}

func (g *rtdGeneratorPrograms) programForKey(key trainingProgramCacheKey) (*arch.Program, error) {
	if cached := g.cache[key]; cached != nil {
		return cached, nil
	}
	programCfg := g.cfg
	if key.seqLen > 0 && key.seqLen != g.cfg.SeqLen {
		clone := *g.cfg
		clone.SeqLen = key.seqLen
		programCfg = &clone
	}
	state := TrainingProgramState{
		RecurrenceActive: key.recurrenceOn,
		HeadUntied:       key.headUntied,
		MTPAuxInactive:   !key.mtpAuxOn,
		Objective:        key.objective,
		DropoutInactive:  true,
	}
	var built *arch.Program
	var buildErr error
	if g.recurrencePhasesScheduled {
		built, buildErr = arch.BuildTrainingIRProgramForRecurrencePhaseFromConfig(programCfg, key.recurrencePhase, state)
	} else {
		built, buildErr = BuildTrainingIRProgramFromConfig(programCfg, state)
	}
	if buildErr != nil {
		return nil, buildErr
	}
	g.cache[key] = built
	return built, nil
}

// maybeAttachRTDCorruption resolves the generator program for the step's cache
// key and applies tied-generator RTD corruption when RTD is active. It is a
// no-op (returns prepared unchanged) for non-RTD multihead objectives.
func maybeAttachRTDCorruption(
	trainer GPUTrainer,
	cfg *ArchConfig,
	batch trainBatch,
	step int,
	prepared objectiveBatch,
	batchSize, seqLen int,
	objective string,
	key trainingProgramCacheKey,
	generatorProgramForKey func(trainingProgramCacheKey) (*arch.Program, error),
	restoreProg *arch.Program,
) (objectiveBatch, error) {
	if !rtdActive(cfg) || objective != arch.ObjectiveMultihead {
		return prepared, nil
	}
	rtdProg, err := generatorProgramForKey(key)
	if err != nil {
		return objectiveBatch{}, fmt.Errorf("build step %d RTD generator program: %w", step, err)
	}
	if cfg.Training.RTD.DedicatedGeneratorEnabled() {
		prepared, err = attachRTDDedicatedGeneratorCorruption(trainer, cfg, batch, step, prepared, batchSize, seqLen, rtdProg, restoreProg)
	} else {
		prepared, err = attachRTDTiedGeneratorCorruption(trainer, cfg, batch, step, prepared, batchSize, seqLen, rtdProg, restoreProg)
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

func multiheadHeadIndexByObjective(cfg *ArchConfig, objective string) int {
	if cfg == nil {
		return -1
	}
	for i, head := range cfg.Training.Heads {
		if head.Objective == objective {
			return i
		}
	}
	return -1
}

func rtdHeadLogitsOutputName(cfg *ArchConfig) (string, error) {
	idx := multiheadHeadIndexByObjective(cfg, arch.ObjectiveRTD)
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
	generatorProg, restoreProg *arch.Program,
) (objectiveBatch, error) {
	if !rtdActive(cfg) {
		return prepared, nil
	}
	switcher, ok := trainer.(gpuProgramSwitcher)
	if !ok {
		return objectiveBatch{}, fmt.Errorf("trainer does not support RTD generator program switching")
	}
	need := cfg.Training.BatchTokens
	probe, err := prepareRTDGeneratorProbeBatch(cfg, batch, step, need, seqLen)
	if err != nil {
		return objectiveBatch{}, err
	}
	if err := switcher.SetProgramGPU(generatorProg); err != nil {
		return objectiveBatch{}, fmt.Errorf("switch to RTD generator program: %w", err)
	}
	restore := func() error {
		if restoreProg == nil {
			return nil
		}
		return switcher.SetProgramGPU(restoreProg)
	}
	if _, err := trainer.EvaluateObjectiveGPU(probe, batchSize, seqLen); err != nil {
		if restoreErr := restore(); restoreErr != nil {
			return objectiveBatch{}, fmt.Errorf("evaluate RTD generator: %w; restore program: %v", err, restoreErr)
		}
		return objectiveBatch{}, fmt.Errorf("evaluate RTD generator: %w", err)
	}
	outputName, err := rtdGeneratorLogitsOutputName(cfg)
	if err != nil {
		_ = restore()
		return objectiveBatch{}, err
	}
	logits, err := readTrainerOutput(trainer, outputName, []int{need, cfg.VocabSize})
	if err != nil {
		if restoreErr := restore(); restoreErr != nil {
			return objectiveBatch{}, fmt.Errorf("read RTD generator logits: %w; restore program: %v", err, restoreErr)
		}
		return objectiveBatch{}, fmt.Errorf("read RTD generator logits: %w", err)
	}
	if err := restore(); err != nil {
		return objectiveBatch{}, fmt.Errorf("restore training program after RTD generator: %w", err)
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
	generatorProg, restoreProg *arch.Program,
) (objectiveBatch, error) {
	if !rtdActive(cfg) {
		return prepared, nil
	}
	switcher, ok := trainer.(gpuProgramSwitcher)
	if !ok {
		return objectiveBatch{}, fmt.Errorf("trainer does not support RTD generator program switching")
	}
	need := cfg.Training.BatchTokens
	probe, err := prepareRTDDedicatedGeneratorProbeBatch(cfg, batch, step, need)
	if err != nil {
		return objectiveBatch{}, err
	}
	attachDedicatedGeneratorInputs(&prepared, probe)
	if err := switcher.SetProgramGPU(generatorProg); err != nil {
		return objectiveBatch{}, fmt.Errorf("switch to RTD dedicated generator program: %w", err)
	}
	restore := func() error {
		if restoreProg == nil {
			return nil
		}
		return switcher.SetProgramGPU(restoreProg)
	}
	if _, err := trainer.EvaluateObjectiveGPU(prepared, batchSize, seqLen); err != nil {
		if restoreErr := restore(); restoreErr != nil {
			return objectiveBatch{}, fmt.Errorf("evaluate RTD dedicated generator: %w; restore program: %v", err, restoreErr)
		}
		return objectiveBatch{}, fmt.Errorf("evaluate RTD dedicated generator: %w", err)
	}
	logits, err := readTrainerOutput(trainer, arch.RTDGeneratorLogitsName, []int{need, cfg.VocabSize})
	if err != nil {
		if restoreErr := restore(); restoreErr != nil {
			return objectiveBatch{}, fmt.Errorf("read RTD dedicated generator logits: %w; restore program: %v", err, restoreErr)
		}
		return objectiveBatch{}, fmt.Errorf("read RTD dedicated generator logits: %w", err)
	}
	if err := restore(); err != nil {
		return objectiveBatch{}, fmt.Errorf("restore training program after RTD dedicated generator: %w", err)
	}
	if err := applyRTDDedicatedGeneratorCorruption(cfg, batch, step, seqLen, probe, logits, &prepared); err != nil {
		return objectiveBatch{}, err
	}
	return prepared, nil
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
	rtdIdx := multiheadHeadIndexByObjective(cfg, arch.ObjectiveRTD)
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
	rtdIdx := multiheadHeadIndexByObjective(cfg, arch.ObjectiveRTD)
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

func sampleTokenFromLogits(logits []float32, temperature float64, u float64) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty logits")
	}
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return 0, fmt.Errorf("invalid temperature=%g", temperature)
	}
	maxVal := math.Inf(-1)
	for _, v := range logits {
		scaled := float64(v) / temperature
		if !math.IsNaN(scaled) && scaled > maxVal {
			maxVal = scaled
		}
	}
	if math.IsInf(maxVal, -1) {
		return 0, fmt.Errorf("all logits are non-finite")
	}
	total := 0.0
	for _, v := range logits {
		scaled := float64(v)/temperature - maxVal
		if math.IsNaN(scaled) {
			continue
		}
		total += math.Exp(scaled)
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, fmt.Errorf("non-finite sampling normalization")
	}
	if u < 0 {
		u = 0
	}
	if u >= 1 {
		u = math.Nextafter(1, 0)
	}
	threshold := u * total
	cumulative := 0.0
	for i, v := range logits {
		scaled := float64(v)/temperature - maxVal
		if math.IsNaN(scaled) {
			continue
		}
		cumulative += math.Exp(scaled)
		if cumulative >= threshold {
			return i, nil
		}
	}
	return len(logits) - 1, nil
}
