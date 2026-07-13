package train

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

type trainBatch struct {
	x, y []int
	err  error
}

type trainingProgramCacheKey struct {
	recurrencePhase int
	recurrenceOn    bool
	headUntied      bool
	mtpAuxOn        bool
	objective       string
	seqLen          int
}

func qatModeForStep(spec TrainingSpec, step int) string {
	mode := strings.TrimSpace(strings.ToLower(spec.QAT))
	if mode == "" {
		mode = "none"
	}
	if spec.QATStart > 0 && step < spec.QATStart {
		return "none"
	}
	return mode
}

// runTrain is the core training loop. It:
// 1. Builds the IR program from the config
// 2. Initializes the GPU trainer
// 3. Runs the training loop with LR scheduling
// 4. Optionally exports weights and runs BPB evaluation
// 5. Reports loss and validation metrics
func runTrain(cfg *ArchConfig, trainPattern string, opts TrainOptions) (TrainResult, error) {
	// MLX CUDA uses thread-local stream state, so trainer creation and all
	// subsequent trainer calls must stay on the same OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if trainPattern == "" {
		return TrainResult{}, fmt.Errorf("training data pattern is required; pass -train 'data/train_*.bin'")
	}
	if opts.CheckpointEvery < 0 {
		return TrainResult{}, fmt.Errorf("checkpoint interval must be >= 0")
	}
	if opts.CheckpointEvery > 0 && opts.CheckpointDir == "" {
		return TrainResult{}, fmt.Errorf("-checkpoint-dir is required when -checkpoint-every > 0")
	}
	swaOverrideLogs, err := applyTrainingSWAOverrides(cfg, opts)
	if err != nil {
		return TrainResult{}, err
	}

	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	swaStart := cfg.Training.SWAStart
	swaDecay := cfg.Training.SWADecay
	swaInterval := cfg.Training.SWAInterval
	if batchTokens%seqLen != 0 {
		return TrainResult{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	batchSize := batchTokens / seqLen
	seed := cfg.Training.Seed
	name := cfg.Name
	for _, msg := range swaOverrideLogs {
		fmt.Printf("  [%s] %s\n", name, msg)
	}
	if swaStart > 0 {
		fmt.Printf("  [%s] SWA/EMA enabled: start=%d interval=%d decay=%g\n", name, swaStart, swaInterval, swaDecay)
	}
	targetValLoss := cfg.Training.TargetValLoss
	earlyStop := newEarlyStopState(cfg.Training.EarlyStop)
	if earlyStop != nil {
		fmt.Printf("  [%s] early stop enabled: patience=%d min_delta=%g min_steps=%d val_gt=%g at_step=%d\n",
			name, cfg.Training.EarlyStop.Patience, cfg.Training.EarlyStop.MinDelta,
			cfg.Training.EarlyStop.MinSteps, cfg.Training.EarlyStop.ValGT, cfg.Training.EarlyStop.AtStep)
	}
	flops := arch.EstimateFLOPs(cfg)
	recurrencePhaseStarts := cfg.PhaseStartSteps()
	recurrencePhasesScheduled := len(recurrencePhaseStarts) > 0
	if cfg.Training.FirstByteMask {
		source, err := configureFirstByteMaskForTraining(cfg, trainPattern)
		if err != nil {
			return TrainResult{}, err
		}
		fmt.Printf("  [%s] first-byte mask enabled (%s)\n", name, source)
	}
	if cfg.CharVocabSize > 0 {
		source, err := configureCharFeaturesForTraining(cfg, trainPattern)
		if err != nil {
			return TrainResult{}, err
		}
		fmt.Printf("  [%s] char features enabled (%s)\n", name, source)
	}
	if cfg.Training.ExampleFramingEnabled() {
		f := cfg.Training.ExampleFraming
		fmt.Printf("  [%s] example framing: content_len=%d bos_id=%d eos_id=%d examples/batch=%d\n",
			name, f.ContentLen, f.BosID, f.EosID, batchSize)
	}
	pairSampler, err := newMinimalPairSampler(cfg)
	if err != nil {
		return TrainResult{}, err
	}
	if pairSampler != nil {
		fmt.Printf("  [%s] minimal pairs: source=%s records=%d loss=%s margin=%g pair_batch_fraction=%.3f\n",
			name, pairSampler.path, len(pairSampler.records), cfg.Training.MinimalPair.Loss, cfg.Training.MinimalPair.Margin, cfg.Training.MinimalPair.PairBatchFraction)
	}
	invarianceSampler, err := newInvariancePairSampler(cfg)
	if err != nil {
		return TrainResult{}, err
	}
	if invarianceSampler != nil {
		s := cfg.Training.Invariance
		fmt.Printf("  [%s] invariance: source=%s records=%d loss=%s weight=%g batch_fraction=%.3f target=%s\n",
			name, invarianceSampler.path, len(invarianceSampler.records), s.Loss, s.Weight, s.BatchFraction, s.Target)
	}
	pllMarginSampler, err := newPLLMarginPairSampler(cfg)
	if err != nil {
		return TrainResult{}, err
	}
	if pllMarginSampler != nil {
		s := cfg.Training.PLLMargin
		fmt.Printf("  [%s] PLL margin: source=%s records=%d margin=%g weight=%g anchor_weight=%g batch_fraction=%.3f target=%s\n",
			name, pllMarginSampler.path, len(pllMarginSampler.records), s.Margin, s.Weight, s.AnchorWeight, s.BatchFraction, s.Target)
	}

	// Build IR program
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return TrainResult{}, fmt.Errorf("build IR program: %w", err)
	}
	fmt.Printf("  [%s] IR program: %d ops, %d weights\n", name, len(prog.Ops), prog.NumWeights)
	recurrenceActivationStep := cfg.Training.EffectiveRecurrenceActivationStep()
	recurrenceScheduled := !recurrencePhasesScheduled && recurrenceActivationStep > 0 && len(cfg.Recurrence) > 0
	recurrenceActive := !recurrenceScheduled
	currentRecurrencePhase := initialRecurrencePhase(recurrencePhasesScheduled)
	mtpUntieScheduled := cfg.MTPUntieEnabled()
	mtpUntieStep := cfg.EffectiveMTPUntieStep()
	headUntied := mtpUntieScheduled && mtpUntieStep <= 0
	mtpAuxActivationScheduled := cfg.MTPActivateAuxLossEnabled()
	mtpAuxActivateStep := cfg.EffectiveMTPActivateStep()
	mtpAuxActive := !mtpAuxActivationScheduled || mtpAuxActivateStep <= 0
	currentObjective := objectiveForStep(cfg.Training, 0)
	currentSeqLen := cfg.Training.EffectiveSeqLenForStep(seqLen, 0)
	currentBatchSize := batchTokens / currentSeqLen
	programCache := make(map[trainingProgramCacheKey]*arch.Program)
	trainingProgramForKey := func(key trainingProgramCacheKey) (*arch.Program, error) {
		if cached := programCache[key]; cached != nil {
			return cached, nil
		}
		programCfg := cfg
		if key.seqLen > 0 && key.seqLen != cfg.SeqLen {
			clone := *cfg
			clone.SeqLen = key.seqLen
			programCfg = &clone
		}
		state := TrainingProgramState{
			RecurrenceActive: key.recurrenceOn,
			HeadUntied:       key.headUntied,
			MTPAuxInactive:   !key.mtpAuxOn,
			Objective:        key.objective,
		}
		var built *arch.Program
		var buildErr error
		if recurrencePhasesScheduled {
			built, buildErr = arch.BuildTrainingIRProgramForRecurrencePhaseFromConfig(programCfg, key.recurrencePhase, state)
		} else {
			built, buildErr = BuildTrainingIRProgramFromConfig(programCfg, state)
		}
		if buildErr != nil {
			return nil, buildErr
		}
		programCache[key] = built
		return built, nil
	}
	currentProgramKey := trainingProgramCacheKey{
		recurrencePhase: currentRecurrencePhase,
		recurrenceOn:    recurrenceActive,
		headUntied:      headUntied,
		mtpAuxOn:        mtpAuxActive,
		objective:       currentObjective,
		seqLen:          currentSeqLen,
	}
	initialProg, err := trainingProgramForKey(currentProgramKey)
	if err != nil {
		return TrainResult{}, fmt.Errorf("build initial training IR program: %w", err)
	}
	if initialProg.NumWeights != prog.NumWeights {
		return TrainResult{}, fmt.Errorf("initial training weight count mismatch: initial=%d final=%d", initialProg.NumWeights, prog.NumWeights)
	}
	if recurrencePhasesScheduled {
		logRecurrencePhasesSchedule(name, cfg, recurrencePhaseStarts, initialProg)
	}
	if recurrenceScheduled {
		fmt.Printf("  [%s] recurrence activates at step %d: pre-activation IR program: %d ops, %d weights\n",
			name, recurrenceActivationStep, len(initialProg.Ops), initialProg.NumWeights)
	}
	if mtpUntieScheduled {
		fmt.Printf("  [%s] LM head unties from embedding at step %d\n", name, mtpUntieStep)
	}
	if mtpAuxActivationScheduled {
		fmt.Printf("  [%s] MTP auxiliary loss activates at step %d\n", name, mtpAuxActivateStep)
	}
	switch cfg.Training.EffectiveObjective() {
	case arch.ObjectiveMLM, arch.ObjectiveMNTP:
		fmt.Printf("  [%s] training objective: %s\n", name, cfg.Training.EffectiveObjective())
	case arch.ObjectiveHybrid:
		causalLabel := fmt.Sprintf("%.2f", cfg.Training.EffectiveHybridCLMFractionForStep(0))
		if len(cfg.Training.HybridCLMFractionSchedule) > 0 {
			causalLabel = fmt.Sprintf("%.2f scheduled(%s)", cfg.Training.EffectiveHybridCLMFractionForStep(0), cfg.Training.HybridCLMFractionScheduleMode)
		}
		fmt.Printf("  [%s] training objective: hybrid granularity=%s causal=%s secondary=%s\n",
			name, cfg.Training.EffectiveHybridMixGranularity(), causalLabel, cfg.Training.EffectiveHybridSecondaryObjective())
		if cfg.Training.EffectiveHybridSecondaryObjective() == arch.ObjectiveBlockDiffusion && cfg.Training.Diffusion != nil {
			fmt.Printf("  [%s] hybrid diffusion: block_size=%d steps_per_block=%d confidence_threshold=%.3f commit_floor=%d\n",
				name, cfg.Training.Diffusion.BlockSize, cfg.Training.Diffusion.StepsPerBlock, cfg.Training.Diffusion.ConfidenceThreshold, cfg.Training.Diffusion.CommitFloor)
		}
	case arch.ObjectiveMultihead:
		fmt.Printf("  [%s] training objective: multihead export_head=%s diffusion_head=%s heads=[%s]\n",
			name, cfg.Training.ExportHead, cfg.Training.DiffusionHead, formatMultiheadHeadsForLog(cfg.Training.Heads))
	}
	if len(cfg.Training.SeqLenSchedule) > 0 {
		fmt.Printf("  [%s] seq_len schedule: max=%d active_step0=%d\n", name, seqLen, currentSeqLen)
	}
	if len(cfg.Training.MLMMaskProbSchedule) > 0 {
		fmt.Printf("  [%s] MLM mask probability schedule enabled: step0=%.3f\n", name, cfg.Training.EffectiveMLMMaskProbForStep(0))
	}

	var shapes []WeightShape
	if opts.SafetensorsPath != "" || opts.CheckpointEvery > 0 || cfg.Training.Data2VecActive() {
		shapes, err = computeWeightShapes(cfg)
		if err != nil {
			return TrainResult{}, fmt.Errorf("compute weight shapes: %w", err)
		}
	}

	// Load weights from safetensors if requested
	var loadedWeights [][]float32
	if opts.SafetensorsLoad != "" {
		if len(shapes) == 0 {
			shapes, err = computeWeightShapes(cfg)
			if err != nil {
				return TrainResult{}, fmt.Errorf("compute weight shapes for load: %w", err)
			}
		}
		loadedWeights, err = loadSafetensorsWeights(opts.SafetensorsLoad, shapes)
		if err != nil {
			return TrainResult{}, fmt.Errorf("load safetensors %q: %w", opts.SafetensorsLoad, err)
		}
		fmt.Printf("  [%s] loaded %d weights from %s\n", name, len(loadedWeights), opts.SafetensorsLoad)
	}

	if err := configureMLXMemoryLimits(name); err != nil {
		return TrainResult{}, err
	}

	distiller, err := newDistillationEnsemble(cfg)
	if err != nil {
		return TrainResult{}, fmt.Errorf("init distillation teachers: %w", err)
	}
	if distiller != nil {
		defer distiller.Close()
		fmt.Printf("  [%s] distillation: teachers=%d strategy=%s objective=%s ce=%.3g kl=%.3g temperature=%.3g\n",
			name, len(distiller.teachers), distiller.strategy,
			distiller.objective, cfg.Training.Distillation.LossWeightCE, cfg.Training.Distillation.LossWeightKL,
			cfg.Training.Distillation.EffectiveTemperature())
	}

	// Initialize GPU trainer
	trainer, err := initGPUTrainer(initialProg, cfg, loadedWeights, opts.OptimizerOverride)
	if err != nil {
		return TrainResult{}, fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()
	tttTelemetryEnabled := len(arch.TTTMLPInnerLRScalesForStep(cfg.Blocks, 0)) > 0
	if opts.telemetry != nil || tttTelemetryEnabled {
		if err := enableTrainingStepComponentLossCapture(trainer); err != nil {
			return TrainResult{}, err
		}
	}
	var data2vec *data2VecTeacher
	if cfg.Training.Data2VecActive() {
		initialWeights, err := readTrainerWeights(trainer)
		if err != nil {
			return TrainResult{}, fmt.Errorf("data2vec initial weight read: %w", err)
		}
		data2vec, err = newData2VecTeacher(cfg, initialWeights, data2VecTeacherObjective(cfg, currentObjective))
		if err != nil {
			return TrainResult{}, fmt.Errorf("init data2vec teacher: %w", err)
		}
		defer data2vec.Close()
	}
	causalEval := causalEvalSwitcher{
		trainer:       trainer,
		hybrid:        cfg.Training.EffectiveObjective() == arch.ObjectiveHybrid,
		programForKey: trainingProgramForKey,
		batchSize:     batchSize,
		seqLen:        seqLen,
	}

	var swaEMA [][]float32
	if swaStart > 0 {
		swaEMA = make([][]float32, prog.NumWeights)
	}

	// Create data loader
	loader, err := data.NewLoaderWithOptions(trainPattern, seed, effectiveLoaderOptions(cfg))
	if err != nil {
		return TrainResult{}, err
	}

	// Load validation set
	const defaultValBatchCount = 10
	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	valSet, valErr := data.NewValSetWithOptions(valPattern, seed, defaultValBatchCount, batchTokens, seqLen, effectiveLoaderOptions(cfg))
	if valErr != nil {
		fmt.Printf("  [%s] no val set: %v\n", name, valErr)
	}

	sched, steps := buildTrainingScheduler(cfg.Training)
	phaseSched, hasPhases := sched.(phaseSchedule)

	var firstLoss, lastLoss float64
	lastUnmaskedLoss := math.NaN()
	lastValLoss := math.NaN()
	hasValLoss := false
	logEvery := effectiveTrainEvery(opts.LogEvery, "MIXLAB_LOG_EVERY", 100)
	valEvery := effectiveTrainEvery(opts.ValEvery, "MIXLAB_VAL_EVERY", 100)
	mlxMemLogEvery := effectiveTrainEvery(0, mlxMemLogEveryEnv, 0)
	mlxClearCacheEvery := effectiveTrainEvery(0, mlxClearCacheEveryEnv, 0)
	telemetry := opts.telemetry
	if telemetry == nil {
		telemetry = &telemetryRuntime{state: newTelemetryState()}
	}
	componentTelemetryEnabled := opts.telemetry != nil || tttTelemetryEnabled
	stepLookaheadEnabled := !envTruthy("MIXLAB_DISABLE_GPU_STEP_LOOKAHEAD")
	start := time.Now()
	// steadyStart is set after step 0 completes — excludes one-time
	// compile/warmup costs from tok/s and ETA estimates.
	var steadyStart time.Time
	telemetry.state.update(telemetryUpdate{
		Model:       name,
		Step:        0,
		TotalSteps:  steps,
		LR:          sched.At(0),
		Objective:   currentObjective,
		SeqLen:      currentSeqLen,
		BatchTokens: batchTokens,
	})
	done := make(chan struct{})
	batchCh := make(chan trainBatch, 4)
	var loadWG sync.WaitGroup
	loadWG.Add(1)
	go func() {
		defer loadWG.Done()
		defer close(batchCh)
		for {
			x, y, err := loader.NextBatch(batchTokens, seqLen)
			batch := trainBatch{x: x, y: y, err: err}
			select {
			case <-done:
				return
			case batchCh <- batch:
			}
			if err != nil {
				return
			}
		}
	}()
	defer func() {
		close(done)
		for range batchCh {
		}
		loadWG.Wait()
	}()

	if steps > 0 {
		initialWaitStart := time.Now()
		stopInitialBatchWait := startSlowTrainingPhaseLogger(name, 0, "load_initial_batch")
		batch, ok := <-batchCh
		stopInitialBatchWait()
		initialDataDuration := time.Since(initialWaitStart)
		if !ok {
			return TrainResult{}, fmt.Errorf("load initial batch: prefetch pipeline closed unexpectedly")
		}
		if batch.err != nil {
			return TrainResult{}, fmt.Errorf("load initial batch: %w", batch.err)
		}
		lastTrainBatch := batch
		initialSubmitStart := time.Now()
		stopInitialSubmit := startSlowTrainingPhaseLogger(name, 0, "submit_step")
		prepared, err := prepareObjectiveBatchWithSeqLen(cfg, batch, 0, currentObjective, currentSeqLen)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("prepare step 0 objective batch: %w", err)
		}
		prepared, err = maybeAttachMinimalPairs(pairSampler, cfg, 0, prepared, currentBatchSize, currentSeqLen)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("prepare step 0 minimal-pair batch: %w", err)
		}
		prepared, err = maybeAttachRTDCorruption(trainer, cfg, batch, 0, prepared, currentBatchSize, currentSeqLen, currentObjective)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, err
		}
		prepared, err = maybeAttachInvariancePairs(invarianceSampler, cfg, 0, prepared, currentBatchSize, currentSeqLen, currentObjective)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("prepare step 0 invariance batch: %w", err)
		}
		prepared, err = maybeAttachPLLMarginPairs(pllMarginSampler, cfg, 0, prepared, currentBatchSize, currentSeqLen, currentObjective)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("prepare step 0 PLL margin batch: %w", err)
		}
		prepared, err = attachDistillationTeacherProbs(distiller, prepared, currentBatchSize, currentSeqLen)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("prepare step 0 distillation batch: %w", err)
		}
		prepared, err = attachData2VecTargets(data2vec, prepared, currentObjective, currentBatchSize, currentSeqLen)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("prepare step 0 data2vec batch: %w", err)
		}
		if err := submitPreparedStepGPU(trainer, prepared, currentBatchSize, currentSeqLen, sched.At(0)); err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("submit step 0: %w", err)
		}
		stopInitialSubmit()
		currentSubmitDuration := time.Since(initialSubmitStart)
		currentPhaseIdx := -1
		submitStepWithScheduledState := func(nextStep int, batch trainBatch) (time.Duration, error) {
			if nextMode := qatModeForStep(cfg.Training, nextStep); nextMode != qatModeForStep(cfg.Training, nextStep-1) {
				if err := trainer.SetQATGPU(nextMode); err != nil {
					return 0, fmt.Errorf("set QAT mode at step %d: %w", nextStep, err)
				}
				if nextMode != "none" {
					fmt.Printf("  [%s] QAT enabled at step %d\n", name, nextStep)
				}
			}
			nextRecurrenceActive := recurrenceActive || (recurrenceScheduled && nextStep >= recurrenceActivationStep)
			nextRecurrencePhase := currentRecurrencePhase
			if recurrencePhasesScheduled {
				nextRecurrencePhase = recurrencePhaseIndexForStep(recurrencePhaseStarts, nextStep)
			}
			nextHeadUntied := headUntied || (mtpUntieScheduled && nextStep >= mtpUntieStep)
			nextMTPAuxActive := mtpAuxActive || (mtpAuxActivationScheduled && nextStep >= mtpAuxActivateStep)
			nextObjective := objectiveForStep(cfg.Training, nextStep)
			nextSeqLen := cfg.Training.EffectiveSeqLenForStep(seqLen, nextStep)
			nextBatchSize := batchTokens / nextSeqLen
			nextProgramKey := trainingProgramCacheKey{
				recurrencePhase: nextRecurrencePhase,
				recurrenceOn:    nextRecurrenceActive,
				headUntied:      nextHeadUntied,
				mtpAuxOn:        nextMTPAuxActive,
				objective:       nextObjective,
				seqLen:          nextSeqLen,
			}
			if nextProgramKey != currentProgramKey {
				switcher, ok := trainer.(gpuProgramSwitcher)
				if !ok {
					return 0, fmt.Errorf("trainer does not support scheduled program switching")
				}
				if nextHeadUntied && !headUntied {
					copier, ok := trainer.(gpuWeightCopier)
					if !ok {
						return 0, fmt.Errorf("trainer does not support LM head untie weight copy")
					}
					if err := copier.CopyWeightGPU("head", "embed"); err != nil {
						return 0, fmt.Errorf("untie LM head at step %d: %w", nextStep, err)
					}
				}
				nextProg, err := trainingProgramForKey(nextProgramKey)
				if err != nil {
					return 0, fmt.Errorf("build scheduled IR program at step %d: %w", nextStep, err)
				}
				if nextProg.NumWeights != prog.NumWeights {
					return 0, fmt.Errorf("scheduled IR weight count mismatch at step %d: scheduled=%d final=%d", nextStep, nextProg.NumWeights, prog.NumWeights)
				}
				if err := switcher.SetProgramGPU(nextProg); err != nil {
					return 0, fmt.Errorf("switch scheduled IR program at step %d: %w", nextStep, err)
				}
				if nextRecurrenceActive && !recurrenceActive {
					fmt.Printf("  [%s] recurrence activated at step %d\n", name, nextStep)
				}
				if nextRecurrencePhase != currentRecurrencePhase {
					logRecurrencePhaseTransition(name, cfg, currentRecurrencePhase, nextRecurrencePhase, nextStep)
				}
				if nextHeadUntied && !headUntied {
					fmt.Printf("  [%s] LM head untied from embedding at step %d\n", name, nextStep)
				}
				if nextMTPAuxActive && !mtpAuxActive {
					fmt.Printf("  [%s] MTP auxiliary loss activated at step %d\n", name, nextStep)
				}
				if nextSeqLen != currentSeqLen {
					fmt.Printf("  [%s] seq_len changed at step %d: %d -> %d\n", name, nextStep, currentSeqLen, nextSeqLen)
				}
				currentRecurrencePhase = nextRecurrencePhase
				recurrenceActive = nextRecurrenceActive
				headUntied = nextHeadUntied
				mtpAuxActive = nextMTPAuxActive
				currentObjective = nextObjective
				currentSeqLen = nextSeqLen
				currentBatchSize = nextBatchSize
				currentProgramKey = nextProgramKey
			}
			submitStart := time.Now()
			stopSubmit := startSlowTrainingPhaseLogger(name, nextStep, "submit_step")
			prepared, err := prepareObjectiveBatchWithSeqLen(cfg, batch, nextStep, nextObjective, nextSeqLen)
			if err != nil {
				stopSubmit()
				return 0, fmt.Errorf("prepare step %d objective batch: %w", nextStep, err)
			}
			prepared, err = maybeAttachMinimalPairs(pairSampler, cfg, nextStep, prepared, nextBatchSize, nextSeqLen)
			if err != nil {
				stopSubmit()
				return 0, fmt.Errorf("prepare step %d minimal-pair batch: %w", nextStep, err)
			}
			prepared, err = maybeAttachRTDCorruption(trainer, cfg, batch, nextStep, prepared, nextBatchSize, nextSeqLen, nextObjective)
			if err != nil {
				stopSubmit()
				return 0, err
			}
			prepared, err = maybeAttachInvariancePairs(invarianceSampler, cfg, nextStep, prepared, nextBatchSize, nextSeqLen, nextObjective)
			if err != nil {
				stopSubmit()
				return 0, fmt.Errorf("prepare step %d invariance batch: %w", nextStep, err)
			}
			prepared, err = maybeAttachPLLMarginPairs(pllMarginSampler, cfg, nextStep, prepared, nextBatchSize, nextSeqLen, nextObjective)
			if err != nil {
				stopSubmit()
				return 0, fmt.Errorf("prepare step %d PLL margin batch: %w", nextStep, err)
			}
			prepared, err = attachDistillationTeacherProbs(distiller, prepared, nextBatchSize, nextSeqLen)
			if err != nil {
				stopSubmit()
				return 0, fmt.Errorf("prepare step %d distillation batch: %w", nextStep, err)
			}
			prepared, err = attachData2VecTargets(data2vec, prepared, nextObjective, nextBatchSize, nextSeqLen)
			if err != nil {
				stopSubmit()
				return 0, fmt.Errorf("prepare step %d data2vec batch: %w", nextStep, err)
			}
			if err := submitPreparedStepGPU(trainer, prepared, nextBatchSize, nextSeqLen, sched.At(nextStep)); err != nil {
				stopSubmit()
				return 0, fmt.Errorf("submit step %d: %w", nextStep, err)
			}
			stopSubmit()
			lastTrainBatch = batch
			return time.Since(submitStart), nil
		}
		canSubmitNextBeforeCollect := func(step int) bool {
			if !stepLookaheadEnabled ||
				data2vec != nil ||
				distiller != nil ||
				rtdActive(cfg) ||
				step >= steps-1 ||
				shouldLogTrainingStep(step, steps, logEvery) ||
				shouldWriteCheckpoint(step, opts.CheckpointEvery) ||
				shouldUpdateSWA(step, swaStart, swaInterval) {
				return false
			}
			nextStep := step + 1
			if qatModeForStep(cfg.Training, nextStep) != qatModeForStep(cfg.Training, step) {
				return false
			}
			if objectiveForStep(cfg.Training, nextStep) != currentObjective {
				return false
			}
			if cfg.Training.EffectiveSeqLenForStep(seqLen, nextStep) != currentSeqLen {
				return false
			}
			nextRecurrenceActive := recurrenceActive || (recurrenceScheduled && nextStep >= recurrenceActivationStep)
			nextRecurrencePhase := currentRecurrencePhase
			if recurrencePhasesScheduled {
				nextRecurrencePhase = recurrencePhaseIndexForStep(recurrencePhaseStarts, nextStep)
			}
			nextHeadUntied := headUntied || (mtpUntieScheduled && nextStep >= mtpUntieStep)
			nextMTPAuxActive := mtpAuxActive || (mtpAuxActivationScheduled && nextStep >= mtpAuxActivateStep)
			return nextRecurrencePhase == currentRecurrencePhase &&
				nextRecurrenceActive == recurrenceActive &&
				nextHeadUntied == headUntied &&
				nextMTPAuxActive == mtpAuxActive
		}

		for step := 0; step < steps; step++ {
			dataDuration := time.Duration(0)
			if step == 0 {
				dataDuration = initialDataDuration
			}
			if hasPhases {
				phaseIdx := phaseSched.phaseIndex[step]
				if phaseIdx != currentPhaseIdx {
					phase := phaseSched.phases[phaseIdx]
					fmt.Printf("  [%s] entering %s (%d/%d) steps=%d lr=%.6f\n",
						name, phaseDisplayLabel(phase, phaseIdx), phaseIdx+1, len(phaseSched.phases), phase.Steps, phase.LR)
					currentPhaseIdx = phaseIdx
				}
			}

			var nextBatch trainBatch
			if step < steps-1 {
				batchWaitStart := time.Now()
				stopBatchWait := startSlowTrainingPhaseLogger(name, step+1, "load_batch")
				nextBatch, ok = <-batchCh
				stopBatchWait()
				dataDuration = time.Since(batchWaitStart)
				if !ok {
					return TrainResult{}, fmt.Errorf("load batch at step %d: prefetch pipeline closed unexpectedly", step+1)
				}
				if nextBatch.err != nil {
					return TrainResult{}, fmt.Errorf("load batch at step %d: %w", step+1, nextBatch.err)
				}
			}

			submittedNextEarly := false
			earlySubmitDuration := time.Duration(0)
			if step < steps-1 && canSubmitNextBeforeCollect(step) {
				var err error
				earlySubmitDuration, err = submitStepWithScheduledState(step+1, nextBatch)
				if err != nil {
					return TrainResult{}, err
				}
				submittedNextEarly = true
			}

			collectStart := time.Now()
			stopCollect := startSlowTrainingPhaseLogger(name, step, "collect_loss")
			lossV, err := trainer.CollectLossGPU()
			stopCollect()
			gpuDuration := currentSubmitDuration + time.Since(collectStart)
			if err != nil {
				return TrainResult{}, fmt.Errorf("collect loss at step %d: %w", step, err)
			}
			v := float64(lossV)
			readStepDiagnostics := opts.telemetry != nil || (tttTelemetryEnabled && shouldLogTrainingStep(step, steps, logEvery))
			componentLosses, err := readTrainingStepComponentLosses(trainer, componentTelemetryEnabled && readStepDiagnostics)
			if err != nil {
				return TrainResult{}, fmt.Errorf("read component losses at step %d: %w", step, err)
			}
			componentLosses, trainingExtra := splitTrainingDiagnostics(componentLosses)
			optimizerStats, err := readOptimizerStats(trainer)
			if err != nil {
				return TrainResult{}, fmt.Errorf("read optimizer stats at step %d: %w", step, err)
			}

			if step == 0 {
				firstLoss = v
				// Anchor steady-state timing after the first step so the
				// one-time compile/warmup cost doesn't poison tok/s and ETA.
				steadyStart = time.Now()
			}
			lastLoss = v
			elapsed := time.Since(start)
			steadyElapsed := elapsed
			stepsForRate := step + 1
			if !steadyStart.IsZero() && step >= 1 {
				steadyElapsed = time.Since(steadyStart)
				stepsForRate = step
			}
			var tokensPerSec float64
			if steadyElapsed > 0 {
				tokensPerSec = float64(batchTokens) * float64(stepsForRate) / steadyElapsed.Seconds()
			}
			telemetry.state.update(telemetryUpdate{
				Model:         name,
				Step:          step,
				TotalSteps:    steps,
				Loss:          v,
				HasLoss:       true,
				ValLoss:       lastValLoss,
				HasValLoss:    hasValLoss,
				LR:            sched.At(step),
				Objective:     currentObjective,
				SeqLen:        currentSeqLen,
				BatchTokens:   batchTokens,
				Elapsed:       elapsed,
				SteadyElapsed: steadyElapsed,
				TokensPerSec:  tokensPerSec,
				Timing: telemetryTiming{
					DataMS: float64(dataDuration) / float64(time.Millisecond),
					GPUMS:  float64(gpuDuration) / float64(time.Millisecond),
				},
				HasTiming:             true,
				ComponentLosses:       componentLosses,
				Extra:                 trainingExtra,
				OptimizerSteps:        optimizerStats.CommittedSteps,
				SkippedOptimizerSteps: optimizerStats.SkippedSteps,
				ConsecutiveSkipped:    optimizerStats.ConsecutiveSkipped,
				OptimizerStepSkipped:  optimizerStats.LastStepSkipped,
			})
			handleMLXMemoryControls(name, step, mlxMemLogEvery, mlxClearCacheEvery, telemetry)

			if data2vec != nil {
				stopData2Vec := startSlowTrainingPhaseLogger(name, step, "data2vec_ema_read")
				weights, err := readTrainerWeights(trainer)
				stopData2Vec()
				if err != nil {
					return TrainResult{}, fmt.Errorf("data2vec read at step %d: %w", step, err)
				}
				if err := data2vec.updateFromStudentWeights(weights, step); err != nil {
					return TrainResult{}, fmt.Errorf("data2vec EMA update at step %d: %w", step, err)
				}
			}

			if shouldUpdateSWA(step, swaStart, swaInterval) {
				stopSWA := startSlowTrainingPhaseLogger(name, step, "swa_read")
				weights, err := readTrainerWeights(trainer)
				stopSWA()
				if err != nil {
					return TrainResult{}, fmt.Errorf("swa read at step %d: %w", step, err)
				}
				if !hasSWAWeights(swaEMA) {
					fmt.Printf("  [%s] SWA: averaging started at step %d\n", name, step)
				}
				updateEMAWeights(swaEMA, weights, swaDecay)
			}

			if shouldLogTrainingStep(step, steps, logEvery) {
				logStart := time.Now()
				valDuration := time.Duration(0)
				valStr := ""
				ranValidation := false
				if valSet != nil && len(valSet.Batches) > 0 && shouldRunValidationStep(step, steps, valEvery) {
					valStart := time.Now()
					stopValidation := startSlowTrainingPhaseLogger(name, step, "validation")
					var valAvg float64
					var err error
					if cfg.Training.MultiheadEnabled() {
						err = causalEval.withCausalEvalProgram(currentProgramKey, func() error {
							var evalErr error
							valAvg, evalErr = meanMultiheadValidationLoss(cfg, valSet, trainer, pairSampler, invarianceSampler, pllMarginSampler, step, batchSize, seqLen)
							return evalErr
						})
					} else {
						valAvg, err = causalEval.meanValidationLossCausal(currentProgramKey, valSet)
					}
					stopValidation()
					valDuration = time.Since(valStart)
					if err == nil {
						lastValLoss = valAvg
						hasValLoss = true
						ranValidation = true
						valStr = fmt.Sprintf(" val=%.4f", valAvg)
					}
				}
				// Use wall-clock average for tok/s and MFU — per-step EMA is
				// unreliable with pipelined training because collect returns
				// near-instantly when the GPU has already finished.
				// Anchor the rate calculation at steadyStart (set after step 0)
				// so the one-time compile/warmup cost doesn't skew estimates.
				mfuStr := ""
				if cfg.Training.HardwareTFLOPs > 0 && flops.TrainingFLOPs > 0 && steadyElapsed > 0 {
					mfu := (float64(flops.TrainingFLOPs) * float64(stepsForRate) / steadyElapsed.Seconds()) / (cfg.Training.HardwareTFLOPs * 1e12)
					mfuStr = fmt.Sprintf(" MFU=%.1f%%", mfu*100)
				}
				phaseStr := ""
				if hasPhases {
					phase := phaseSched.PhaseAt(step)
					phaseStr = fmt.Sprintf(" phase=%s", phaseDisplayLabel(phase, currentPhaseIdx))
				}
				fmt.Printf("  [%s] step %d/%d loss=%.4f%s lr=%.6f%s tok/s=%.0f%s %s\n",
					name, step, steps, v, valStr, sched.At(step), phaseStr, tokensPerSec, mfuStr, formatProgressTiming(time.Since(start), steadyElapsed, stepsForRate, step, steps))
				if extra := formatTrainingExtraDiagnostics(trainingExtra); extra != "" {
					fmt.Printf("  [%s] [ttt] %s\n", name, extra)
				}
				logDuration := time.Since(logStart)
				if opts.Timing {
					compileStatsStr := formatCompileStats(trainer)
					fmt.Printf("  [%s] [timing] data=%.1fms gpu=%.1fms val=%.1fms log=%.1fms%s\n",
						name,
						float64(dataDuration)/float64(time.Millisecond),
						float64(gpuDuration)/float64(time.Millisecond),
						float64(valDuration)/float64(time.Millisecond),
						float64(logDuration)/float64(time.Millisecond),
						compileStatsStr)
				}
				telemetry.state.update(telemetryUpdate{
					Model:         name,
					Step:          step,
					TotalSteps:    steps,
					Loss:          v,
					HasLoss:       true,
					ValLoss:       lastValLoss,
					HasValLoss:    hasValLoss,
					LR:            sched.At(step),
					Objective:     currentObjective,
					SeqLen:        currentSeqLen,
					BatchTokens:   batchTokens,
					Elapsed:       time.Since(start),
					SteadyElapsed: steadyElapsed,
					TokensPerSec:  tokensPerSec,
					Timing: telemetryTiming{
						DataMS:       float64(dataDuration) / float64(time.Millisecond),
						GPUMS:        float64(gpuDuration) / float64(time.Millisecond),
						ValidationMS: float64(valDuration) / float64(time.Millisecond),
						LogMS:        float64(logDuration) / float64(time.Millisecond),
					},
					HasTiming:             true,
					ComponentLosses:       componentLosses,
					Extra:                 trainingExtra,
					OptimizerSteps:        optimizerStats.CommittedSteps,
					SkippedOptimizerSteps: optimizerStats.SkippedSteps,
					ConsecutiveSkipped:    optimizerStats.ConsecutiveSkipped,
					OptimizerStepSkipped:  optimizerStats.LastStepSkipped,
				})
				if err := telemetry.writeSnapshot(true); err != nil {
					return TrainResult{}, err
				}
				if ranValidation && hasValLoss && targetValLoss > 0 && lastValLoss <= targetValLoss {
					fmt.Printf("  [%s] target val loss %.4f reached at step %d (val=%.4f), stopping early\n",
						name, targetValLoss, step, lastValLoss)
					break
				}
				if ranValidation && earlyStop != nil {
					if stop, reason := earlyStop.observe(step, lastValLoss); stop {
						fmt.Printf("  [%s] early stop at step %d: %s\n", name, step, reason)
						break
					}
				}
			}

			if shouldWriteCheckpoint(step, opts.CheckpointEvery) {
				stopCheckpoint := startSlowTrainingPhaseLogger(name, step, "checkpoint")
				artifacts, err := writeCheckpoint(cfg, trainer, shapes, opts.CheckpointDir, step+1, swaEMA)
				if err != nil {
					stopCheckpoint()
					return TrainResult{}, fmt.Errorf("checkpoint at step %d: %w", step+1, err)
				}
				stopCheckpoint()
				fmt.Printf("  [%s] checkpoint saved: %s\n", name, artifacts.Summary())
			}

			// Plain steps may submit the next batch before collecting the
			// current loss. Boundary steps still submit here, after any
			// validation/checkpoint/logging that can flush trainer state.
			if step < steps-1 {
				if submittedNextEarly {
					currentSubmitDuration = earlySubmitDuration
				} else {
					var err error
					currentSubmitDuration, err = submitStepWithScheduledState(step+1, nextBatch)
					if err != nil {
						return TrainResult{}, err
					}
				}
			}
		}
		finalEvalBatch := objectiveBatch{x: lastTrainBatch.x, y: lastTrainBatch.y}
		if cfg.Training.MultiheadEnabled() {
			var err error
			finalEvalBatch, err = prepareObjectiveBatchWithSeqLen(cfg, lastTrainBatch, steps, arch.ObjectiveMultihead, seqLen)
			if err != nil {
				return TrainResult{}, fmt.Errorf("prepare final multihead training loss batch: %w", err)
			}
			finalEvalBatch, err = maybeAttachMinimalPairs(pairSampler, cfg, steps, finalEvalBatch, batchSize, seqLen)
			if err != nil {
				return TrainResult{}, fmt.Errorf("prepare final minimal-pair training loss batch: %w", err)
			}
			finalEvalBatch, err = maybeAttachInvariancePairs(invarianceSampler, cfg, steps, finalEvalBatch, batchSize, seqLen, arch.ObjectiveMultihead)
			if err != nil {
				return TrainResult{}, fmt.Errorf("prepare final invariance training loss batch: %w", err)
			}
			finalEvalBatch, err = maybeAttachPLLMarginPairs(pllMarginSampler, cfg, steps, finalEvalBatch, batchSize, seqLen, arch.ObjectiveMultihead)
			if err != nil {
				return TrainResult{}, fmt.Errorf("prepare final PLL margin training loss batch: %w", err)
			}
		} else if cfg.Training.ExampleFramingEnabled() {
			var err error
			finalEvalBatch, err = prepareObjectiveBatchWithSeqLen(cfg, lastTrainBatch, steps, arch.ObjectiveCausal, seqLen)
			if err != nil {
				return TrainResult{}, fmt.Errorf("prepare final framed training loss batch: %w", err)
			}
		}
		var evalLoss float32
		switch {
		case cfg.Training.MultiheadEnabled():
			err = causalEval.withCausalEvalProgram(currentProgramKey, func() error {
				var evalErr error
				evalLoss, evalErr = evaluateObjectiveTrainingLossGPU(trainer, finalEvalBatch, batchSize, seqLen)
				return evalErr
			})
		case cfg.Training.ExampleFramingEnabled():
			evalLoss, err = causalEval.evaluateCausalObjectiveTrainingLossGPU(currentProgramKey, finalEvalBatch)
		default:
			evalLoss, err = causalEval.evaluateCausalObjectiveGPU(currentProgramKey, finalEvalBatch)
		}
		if err != nil {
			return TrainResult{}, fmt.Errorf("evaluate final training loss: %w", err)
		}
		lastUnmaskedLoss = float64(evalLoss)
	}

	// Export safetensors if requested (before closing trainer)
	if opts.SafetensorsPath != "" {
		artifacts, err := exportTrainingSafetensorsArtifacts(cfg, trainer, shapes, opts, swaEMA)
		if err != nil {
			fmt.Printf("  [%s] safetensors export failed: %v\n", name, err)
		} else {
			fmt.Printf("  [%s] safetensors artifacts: %s\n", name, artifacts.Summary())
		}
	}

	// Full BPB evaluation if requested
	if opts.DoFullEval {
		switch {
		case cfg.Training.MultiheadEnabled():
			fmt.Printf("  [%s] full validation BPB failed: training.objective=multihead is not supported by continuous-stream full eval in v1\n", name)
		case cfg.Training.ExampleFramingEnabled():
			fmt.Printf("  [%s] full validation BPB failed: training.example_framing is not supported by continuous-stream full eval in v1\n", name)
		default:
			if cfg.Training.TTTSteps > 0 {
				if cfg.Training.TTTMode == "lora" {
					fmt.Printf("  [%s] computing full validation BPB with LoRA-TTT (steps=%d lr=%g rank=%d)...\n", name, cfg.Training.TTTSteps, cfg.Training.TTTLR, cfg.Training.TTTRank)
				} else {
					fmt.Printf("  [%s] computing full validation BPB with score-first TTT (steps=%d lr=%g)...\n", name, cfg.Training.TTTSteps, cfg.Training.TTTLR)
				}
			} else {
				fmt.Printf("  [%s] computing full validation BPB...\n", name)
			}
			lutDir := opts.LUTDir
			if lutDir == "" {
				lutDir = "data"
			}
			if err := causalEval.withCausalEvalProgram(currentProgramKey, func() error {
				return runFullEval(cfg, valPattern, trainer, lutDir)
			}); err != nil {
				fmt.Printf("  [%s] full validation BPB failed: %v\n", name, err)
			}
		}
	}

	elapsed := time.Since(start)
	result := TrainResult{
		Name:             name,
		FirstLoss:        firstLoss,
		LastLoss:         lastLoss,
		LastUnmaskedLoss: lastUnmaskedLoss,
		LastValLoss:      lastValLoss,
		HasValLoss:       hasValLoss,
		Delta:            lastLoss - firstLoss,
		Elapsed:          elapsed,
		StepFLOPs:        flops.TrainingFLOPs,
		FLOPsPerTok:      flops.FLOPsPerToken,
	}
	fmt.Println(result.formatSummary())

	return result, nil
}
