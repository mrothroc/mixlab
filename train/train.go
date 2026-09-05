package train

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// runTrain executes one configured training run end to end.
func runTrain(cfg *ArchConfig, trainPattern string, opts TrainOptions) (TrainResult, error) {
	// MLX CUDA uses thread-local stream state, so trainer creation and all
	// subsequent trainer calls must stay on the same OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if trainPattern == "" {
		return TrainResult{}, fmt.Errorf("training data pattern is required; pass -train 'data/train_*.bin'")
	}
	if err := validateRunTrainOptions(opts); err != nil {
		return TrainResult{}, err
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
	if !cfg.Training.LengthBucketsChangeShape(seqLen) && batchTokens%seqLen != 0 {
		return TrainResult{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	batchSize := batchTokens / seqLen
	seed := cfg.Training.Seed
	name := cfg.Name
	if err := configureDatasetForTraining(cfg, trainPattern, name); err != nil {
		return TrainResult{}, err
	}
	targetValLoss := cfg.Training.TargetValLoss
	earlyStop := newEarlyStopState(cfg.Training.EarlyStop)
	flops := arch.EstimateFLOPs(cfg)
	logTrainingRunSetup(cfg, name, batchSize, batchTokens, swaOverrideLogs, swaStart, swaInterval, swaDecay, earlyStop, &flops)
	recurrencePhaseStarts := cfg.PhaseStartSteps()
	recurrencePhasesScheduled := len(recurrencePhaseStarts) > 0
	if err := configureTrainingTokenFeatures(cfg, trainPattern, name); err != nil {
		return TrainResult{}, err
	}
	logDatasetFraming(cfg, name, batchSize)
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

	resumeSetup, err := prepareResumeRun(cfg, trainPattern, opts.Resume, earlyStop)
	if err != nil {
		return TrainResult{}, err
	}
	sched := resumeSetup.Scheduler
	steps := resumeSetup.Steps
	startStep := resumeSetup.StartStep
	resumed := resumeSetup.Loaded
	labelDiversityMonitor := newClassificationLabelDiversityMonitor(
		cfg.Training.DatasetNumLabels, steps-startStep,
	)
	if resumed != nil {
		fmt.Printf("  [%s] resuming complete checkpoint %s at step %d/%d\n", name, resumed.Manifest.ManifestPath, startStep, steps)
	}

	// Build IR program
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return TrainResult{}, fmt.Errorf("build IR program: %w", err)
	}
	fmt.Printf("  [%s] IR program: %d ops, %d weights\n", name, len(prog.Ops), prog.NumWeights)
	recurrenceActivationStep := cfg.Training.EffectiveRecurrenceActivationStep()
	recurrenceScheduled := !recurrencePhasesScheduled && recurrenceActivationStep > 0 && len(cfg.Recurrence) > 0
	recurrenceActive := !recurrenceScheduled || startStep >= recurrenceActivationStep
	currentRecurrencePhase := initialRecurrencePhase(recurrencePhasesScheduled)
	if recurrencePhasesScheduled {
		currentRecurrencePhase = recurrencePhaseIndexForStep(recurrencePhaseStarts, startStep)
	}
	mtpUntieScheduled := cfg.MTPUntieEnabled()
	mtpUntieStep := cfg.EffectiveMTPUntieStep()
	headUntied := mtpUntieScheduled && startStep >= mtpUntieStep
	mtpAuxActivationScheduled := cfg.MTPActivateAuxLossEnabled()
	mtpAuxActivateStep := cfg.EffectiveMTPActivateStep()
	mtpAuxActive := !mtpAuxActivationScheduled || startStep >= mtpAuxActivateStep
	currentObjective := objectiveForStep(cfg.Training, startStep)
	currentSeqLen := cfg.Training.EffectiveSeqLenForStep(seqLen, startStep)
	currentBatchSize := batchTokens / currentSeqLen
	programCache := make(map[trainingProgramCacheKey]*arch.Program)
	trainingProgramForKey := func(key trainingProgramCacheKey) (*arch.Program, error) {
		if cached := programCache[key]; cached != nil {
			return cached, nil
		}
		programCfg := cfg
		effectiveSeqLen := key.seqLen
		if effectiveSeqLen <= 0 {
			effectiveSeqLen = cfg.SeqLen
		}
		effectiveBatchSize := key.batchSize
		if effectiveBatchSize <= 0 {
			effectiveBatchSize = cfg.Training.BatchTokens / effectiveSeqLen
		}
		effectiveBatchTokens := effectiveBatchSize * effectiveSeqLen
		if effectiveSeqLen != cfg.SeqLen || effectiveBatchTokens != cfg.Training.BatchTokens {
			clone := *cfg
			clone.SeqLen = effectiveSeqLen
			clone.Training = cfg.Training
			clone.Training.BatchTokens = effectiveBatchTokens
			programCfg = &clone
		}
		state := TrainingProgramState{
			RecurrenceActive: key.recurrenceOn,
			HeadUntied:       key.headUntied,
			MTPAuxInactive:   !key.mtpAuxOn,
			DropoutInactive:  key.dropoutInactive,
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
		batchSize:       currentBatchSize,
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
	case arch.ObjectiveClassification:
		fmt.Printf("  [%s] training objective: classification num_labels=%d pooling=%s classifier_dropout=%g\n",
			name, cfg.Training.Classification.NumLabels, cfg.EffectiveClassificationPooling(), cfg.EffectiveClassifierDropout())
	case arch.ObjectiveHybrid:
		causalLabel := fmt.Sprintf("%.2f", cfg.Training.EffectiveHybridCLMFractionForStep(startStep))
		if len(cfg.Training.HybridCLMFractionSchedule) > 0 {
			causalLabel = fmt.Sprintf("%.2f scheduled(%s)", cfg.Training.EffectiveHybridCLMFractionForStep(startStep), cfg.Training.HybridCLMFractionScheduleMode)
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
		fmt.Printf("  [%s] seq_len schedule: max=%d active_step%d=%d\n", name, seqLen, startStep, currentSeqLen)
	}
	if len(cfg.Training.MLMMaskProbSchedule) > 0 {
		fmt.Printf("  [%s] MLM mask probability schedule enabled: step%d=%.3f\n", name, startStep, cfg.Training.EffectiveMLMMaskProbForStep(startStep))
	}
	if len(cfg.Training.MLMMaskUnitSchedule) > 0 {
		fmt.Printf("  [%s] MLM mask unit schedule: %s\n", name, formatMLMMaskUnitSchedule(cfg.Training.MLMMaskUnitSchedule))
	} else if cfg.Training.EffectiveMLMMaskUnit() != arch.MLMMaskUnitToken {
		fmt.Printf("  [%s] MLM mask unit: %s\n", name, cfg.Training.EffectiveMLMMaskUnit())
	}

	var shapes []WeightShape
	if opts.SafetensorsPath != "" || opts.CheckpointEvery > 0 || cfg.Training.Data2VecActive() || resumed != nil {
		shapes, err = computeWeightShapes(cfg)
		if err != nil {
			return TrainResult{}, fmt.Errorf("compute weight shapes: %w", err)
		}
	}

	// Load weights from safetensors if requested
	var loadedWeights [][]float32
	loadPath := opts.SafetensorsLoad
	if resumed != nil {
		loadPath = resumed.ModelPath
	}
	if loadPath != "" {
		if len(shapes) == 0 {
			shapes, err = computeWeightShapes(cfg)
			if err != nil {
				return TrainResult{}, fmt.Errorf("compute weight shapes for load: %w", err)
			}
		}
		freshClassificationWeights := 0
		if cfg.ClassificationEnabled() && resumed == nil {
			loadedWeights, freshClassificationWeights, err = loadClassificationWarmStartWeights(
				loadPath, shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd,
			)
		} else {
			loadedWeights, err = loadSafetensorsWeights(loadPath, shapes)
		}
		if err != nil {
			return TrainResult{}, fmt.Errorf("load safetensors %q: %w", loadPath, err)
		}
		if freshClassificationWeights > 0 {
			fmt.Printf("  [%s] warm-started %d LM/backbone weights from %s; initialized %d classifier weights\n",
				name, len(loadedWeights)-freshClassificationWeights, loadPath, freshClassificationWeights)
		} else {
			fmt.Printf("  [%s] loaded %d weights from %s\n", name, len(loadedWeights), loadPath)
		}
	}

	memoryLimitPlan, err := configureMLXMemoryLimits(name)
	if err != nil {
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
	stepSetter, ok := trainer.(gpuTrainingStepSetter)
	if !ok {
		return TrainResult{}, fmt.Errorf("trainer does not support deterministic training-step RNG")
	}
	if err := stepSetter.SetTrainingStepGPU(startStep); err != nil {
		return TrainResult{}, fmt.Errorf("set training step %d: %w", startStep, err)
	}
	if resumed != nil {
		if err := restoreResumableTrainerState(trainer, *resumed); err != nil {
			return TrainResult{}, err
		}
		if err := trainer.SetQATGPU(qatModeForStep(cfg.Training, startStep)); err != nil {
			return TrainResult{}, fmt.Errorf("restore QAT mode at step %d: %w", startStep, err)
		}
	}
	tttDiagnosticsEnabled := len(arch.TTTMLPInnerLRScalesForStep(cfg.Blocks, startStep)) > 0
	if opts.telemetry != nil {
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
		if resumed != nil {
			if err := data2vec.restoreEMAWeights(resumed.Data2Vec); err != nil {
				return TrainResult{}, fmt.Errorf("restore data2vec EMA: %w", err)
			}
		}
		defer data2vec.Close()
	} else if resumed != nil && len(resumed.Data2Vec) != 0 {
		return TrainResult{}, fmt.Errorf("checkpoint contains data2vec EMA state but training.data2vec is disabled")
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
	if resumed != nil {
		if err := restoreSWAWeights(&swaEMA, resumed.SWA, shapes, swaStart > 0); err != nil {
			return TrainResult{}, err
		}
	}

	// Create data loader
	loader, err := data.NewLoaderWithOptions(trainPattern, seed, effectiveLoaderOptions(cfg))
	if err != nil {
		return TrainResult{}, err
	}
	if startStep > 0 {
		fmt.Printf("  [%s] replaying %d loader batches to restore deterministic data position\n", name, startStep)
		if err := replayTrainingLoader(loader, startStep, batchTokens, seqLen); err != nil {
			return TrainResult{}, err
		}
	}

	valSet, valPattern, err := loadTrainingValidationSet(cfg, trainPattern, name, seed, batchTokens, seqLen)
	if err != nil {
		return TrainResult{}, err
	}

	phaseSched, hasPhases := sched.(phaseSchedule)
	metricSched, hasMetricSchedule := sched.(metricTrainingScheduler)
	if cfg.Training.EffectiveLRSchedule() == arch.LRScheduleNewBob && !hasMetricSchedule {
		return TrainResult{}, fmt.Errorf("training.lr_schedule=%q did not construct a metric-driven scheduler", arch.LRScheduleNewBob)
	}

	var firstLoss, lastLoss float64
	lastUnmaskedLoss := math.NaN()
	lastValLoss := math.NaN()
	hasValLoss := false
	logEvery := effectiveTrainEvery(opts.LogEvery, "MIXLAB_LOG_EVERY", 100)
	valEveryFallback := 100
	if cfg.Training.ConfiguredMetricValidation() {
		valEveryFallback = cfg.Training.ValEverySteps
	}
	valEvery := effectiveTrainEvery(opts.ValEvery, "MIXLAB_VAL_EVERY", valEveryFallback)
	if cfg.Training.ConfiguredMetricValidation() {
		fmt.Printf("  [%s] configured validation: every=%d completed steps examples=%d/%d\n",
			name, valEvery, valSet.EvaluatedExamples, valSet.TotalExamples)
	}
	mlxMemLogEvery := effectiveTrainEvery(0, mlxMemLogEveryEnv, 0)
	mlxClearCacheEvery := effectiveTrainEvery(0, mlxClearCacheEveryEnv, 0)
	telemetry := opts.telemetry
	if telemetry == nil {
		telemetry = &telemetryRuntime{state: newTelemetryState()}
	}
	componentTelemetryEnabled := opts.telemetry != nil
	sobolevBindings, err := arch.CollectS4DSobolevWeightBindings(cfg)
	if err != nil {
		return TrainResult{}, fmt.Errorf("collect S4D Sobolev diagnostics: %w", err)
	}
	stepLookaheadEnabled := !envTruthy("MIXLAB_DISABLE_GPU_STEP_LOOKAHEAD")
	start := time.Now()
	var progress trainingProgress
	telemetry.state.update(telemetryUpdate{
		Model:       name,
		Step:        startStep,
		TotalSteps:  steps,
		LR:          sched.At(startStep),
		Objective:   currentObjective,
		SeqLen:      currentSeqLen,
		BatchTokens: batchSize * currentSeqLen,
	})
	done := make(chan struct{})
	batchCh := make(chan trainBatch, 4)
	var loadWG sync.WaitGroup
	loadWG.Add(1)
	go func() {
		defer loadWG.Done()
		defer close(batchCh)
		for {
			loaded, err := loader.NextBatchDetailed(batchTokens, seqLen)
			batch := trainBatchFromDataBatch(loaded, err)
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
		stopInitialBatchWait := startSlowTrainingPhaseLogger(name, startStep, "load_initial_batch")
		batch, ok := <-batchCh
		stopInitialBatchWait()
		initialDataDuration := time.Since(initialWaitStart)
		if !ok {
			return TrainResult{}, fmt.Errorf("load initial batch: prefetch pipeline closed unexpectedly")
		}
		if batch.err != nil {
			return TrainResult{}, fmt.Errorf("load initial batch: %w", batch.err)
		}
		initialBatchSize, initialSeqLen := batch.effectiveShape(currentBatchSize, currentSeqLen)
		if initialBatchSize != currentBatchSize || initialSeqLen != currentSeqLen {
			initialKey := currentProgramKey
			initialKey.batchSize = initialBatchSize
			initialKey.seqLen = initialSeqLen
			initialBucketProg, err := trainingProgramForKey(initialKey)
			if err != nil {
				return TrainResult{}, fmt.Errorf("build initial length-bucket program: %w", err)
			}
			switcher, ok := trainer.(gpuProgramSwitcher)
			if !ok {
				return TrainResult{}, fmt.Errorf("trainer does not support length-bucket program switching")
			}
			if err := switcher.SetProgramGPU(initialBucketProg); err != nil {
				return TrainResult{}, fmt.Errorf("switch initial length-bucket program: %w", err)
			}
			currentBatchSize, currentSeqLen, currentProgramKey = initialBatchSize, initialSeqLen, initialKey
		}
		lastTrainBatch := batch
		initialSubmitStart := time.Now()
		stopInitialSubmit := startSlowTrainingPhaseLogger(name, startStep, "submit_step")
		if warning := labelDiversityMonitor.observe(batch); warning != "" {
			fmt.Printf("  [%s] warning: %s\n", name, warning)
		}
		prepared, err := prepareTrainingBatch(
			cfg, trainer, batch, startStep, currentObjective, currentBatchSize, currentSeqLen,
			pairSampler, invarianceSampler, pllMarginSampler, distiller, data2vec,
		)
		if err != nil {
			stopInitialSubmit()
			return TrainResult{}, err
		}
		currentDiagnosticBatch := prepared
		currentDiagnosticBatchSize := currentBatchSize
		currentDiagnosticSeqLen := currentSeqLen
		if err := submitPreparedStepGPU(trainer, prepared, currentBatchSize, currentSeqLen, sched.At(startStep)); err != nil {
			stopInitialSubmit()
			return TrainResult{}, fmt.Errorf("submit step %d: %w", startStep,
				annotateMLXTrainingStepError(err, memoryLimitPlan, currentBatchSize, currentSeqLen))
		}
		stopInitialSubmit()
		currentSubmitDuration := time.Since(initialSubmitStart)
		currentPhaseIdx := -1
		currentMLMMaskUnit := cfg.Training.EffectiveMLMMaskUnitForStep(startStep)
		submitStepWithScheduledState := func(nextStep int, batch trainBatch) (time.Duration, error) {
			if nextMode := qatModeForStep(cfg.Training, nextStep); nextMode != qatModeForStep(cfg.Training, nextStep-1) {
				if err := trainer.SetQATGPU(nextMode); err != nil {
					return 0, fmt.Errorf("set QAT mode at step %d: %w", nextStep, err)
				}
				if nextMode != "none" {
					fmt.Printf("  [%s] QAT enabled at step %d\n", name, nextStep)
				}
			}
			nextMLMMaskUnit := cfg.Training.EffectiveMLMMaskUnitForStep(nextStep)
			if nextMLMMaskUnit != currentMLMMaskUnit {
				fmt.Printf("  [%s] MLM mask unit changed at step %d: %s -> %s\n", name, nextStep, currentMLMMaskUnit, nextMLMMaskUnit)
				currentMLMMaskUnit = nextMLMMaskUnit
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
			nextBatchSize, nextSeqLen = batch.effectiveShape(nextBatchSize, nextSeqLen)
			nextProgramKey := trainingProgramCacheKey{
				recurrencePhase: nextRecurrencePhase,
				recurrenceOn:    nextRecurrenceActive,
				headUntied:      nextHeadUntied,
				mtpAuxOn:        nextMTPAuxActive,
				objective:       nextObjective,
				seqLen:          nextSeqLen,
				batchSize:       nextBatchSize,
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
			if warning := labelDiversityMonitor.observe(batch); warning != "" {
				fmt.Printf("  [%s] warning: %s\n", name, warning)
			}
			prepared, err := prepareTrainingBatch(
				cfg, trainer, batch, nextStep, nextObjective, nextBatchSize, nextSeqLen,
				pairSampler, invarianceSampler, pllMarginSampler, distiller, data2vec,
			)
			if err != nil {
				stopSubmit()
				return 0, err
			}
			if err := submitPreparedStepGPU(trainer, prepared, nextBatchSize, nextSeqLen, sched.At(nextStep)); err != nil {
				stopSubmit()
				return 0, fmt.Errorf("submit step %d: %w", nextStep,
					annotateMLXTrainingStepError(err, memoryLimitPlan, nextBatchSize, nextSeqLen))
			}
			// Logging steps cannot have a later step pending, so this single slot
			// always identifies the batch whose loss is collected and sampled.
			currentDiagnosticBatch = prepared
			currentDiagnosticBatchSize = nextBatchSize
			currentDiagnosticSeqLen = nextSeqLen
			stopSubmit()
			lastTrainBatch = batch
			return time.Since(submitStart), nil
		}
		canSubmitNextBeforeCollect := func(step int, batch trainBatch) bool {
			if !trainerAllowsStepLookahead(trainer, stepLookaheadEnabled) ||
				step == startStep ||
				data2vec != nil ||
				distiller != nil ||
				rtdActive(cfg) ||
				step >= steps-1 ||
				shouldLogTrainingStep(step, steps, logEvery) ||
				shouldRunTrainingValidationStep(cfg.Training, step, steps, valEvery) ||
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
			nextBatchSize, nextSeqLen := batch.effectiveShape(batchTokens/cfg.Training.EffectiveSeqLenForStep(seqLen, nextStep), cfg.Training.EffectiveSeqLenForStep(seqLen, nextStep))
			if nextSeqLen != currentSeqLen || nextBatchSize != currentBatchSize {
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

		for step := startStep; step < steps; step++ {
			appliedLR := sched.At(step)
			dataDuration := time.Duration(0)
			if step == startStep {
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
			if step < steps-1 && canSubmitNextBeforeCollect(step, nextBatch) {
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
				return TrainResult{}, fmt.Errorf("collect loss at step %d: %w", step,
					annotateMLXTrainingStepError(err, memoryLimitPlan, currentDiagnosticBatchSize, currentDiagnosticSeqLen))
			}
			v := float64(lossV)
			stepTokens := currentDiagnosticBatchSize * currentDiagnosticSeqLen
			componentLosses, err := readTrainingStepComponentLosses(trainer, componentTelemetryEnabled)
			if err != nil {
				return TrainResult{}, fmt.Errorf("read component losses at step %d: %w", step, err)
			}
			if tttDiagnosticsEnabled && shouldLogTrainingStep(step, steps, logEvery) {
				diagnosticStart := time.Now()
				tttDiagnostics, err := sampleTTTDiagnostics(
					trainer,
					currentDiagnosticBatch,
					currentDiagnosticBatchSize,
					currentDiagnosticSeqLen,
					true,
				)
				gpuDuration += time.Since(diagnosticStart)
				if err != nil {
					return TrainResult{}, fmt.Errorf("sample TTT diagnostics at step %d: %w", step, err)
				}
				if len(tttDiagnostics) > 0 && componentLosses == nil {
					componentLosses = make(map[string]float64, len(tttDiagnostics))
				}
				for diagnostic, value := range tttDiagnostics {
					componentLosses[diagnostic] = value
				}
			}
			componentLosses, trainingExtra := splitTrainingDiagnostics(componentLosses)
			var sobolevDiagnostics *telemetryS4DSobolev
			if len(sobolevBindings) > 0 && shouldLogTrainingStep(step, steps, logEvery) {
				diagnosticStart := time.Now()
				sobolevDiagnostics, err = sampleS4DSobolevDiagnostics(trainer, sobolevBindings)
				gpuDuration += time.Since(diagnosticStart)
				if err != nil {
					return TrainResult{}, fmt.Errorf("sample S4D Sobolev diagnostics at step %d: %w", step, err)
				}
			}
			optimizerStats, err := readOptimizerStats(trainer)
			if err != nil {
				return TrainResult{}, fmt.Errorf("read optimizer stats at step %d: %w", step, err)
			}

			if step == startStep {
				firstLoss = v
			}
			lastLoss = v
			elapsed := time.Since(start)
			steadyElapsed, stepsForRate, tokensPerSec := progress.collected(time.Now(), stepTokens)
			telemetry.state.update(telemetryUpdate{
				Model:         name,
				Step:          step,
				TotalSteps:    steps,
				Loss:          v,
				HasLoss:       true,
				ValLoss:       lastValLoss,
				HasValLoss:    hasValLoss,
				LR:            appliedLR,
				Objective:     currentObjective,
				SeqLen:        currentSeqLen,
				BatchTokens:   stepTokens,
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
				Masking:               currentDiagnosticBatch.mlmMaskStats.telemetry(),
				S4DSobolev:            sobolevDiagnostics,
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

			validation, err := runTrainingValidation(trainingValidationRequest{
				cfg: cfg, name: name, valSet: valSet, trainer: trainer, causalEval: causalEval,
				currentProgramKey: currentProgramKey, pairSampler: pairSampler,
				invarianceSampler: invarianceSampler, pllMarginSampler: pllMarginSampler,
				step: step, totalSteps: steps, valEvery: valEvery, batchSize: batchSize, seqLen: seqLen,
				metricScheduler: metricSched,
			})
			if err != nil {
				return TrainResult{}, err
			}
			valDuration, valStr, ranValidation := validation.duration, validation.log, validation.ran
			if ranValidation {
				lastValLoss = validation.loss
				hasValLoss = true
			}

			logStep := shouldLogTrainingStep(step, steps, logEvery)
			if ranValidation && !logStep {
				fmt.Printf("  [%s] completed_steps=%d%s\n", name, step+1, valStr)
			}
			if logStep {
				logStart := time.Now()
				// Use wall-clock average for tok/s and MFU — per-step EMA is
				// unreliable with pipelined training because collect returns
				// near-instantly when the GPU has already finished.
				// The progress clock excludes the complete startup iteration.
				mfuStr := ""
				if flops.TrainingFLOPsReliable && cfg.Training.HardwareTFLOPs > 0 && flops.TrainingFLOPs > 0 && steadyElapsed > 0 {
					mfu := (float64(flops.TrainingFLOPs) * float64(stepsForRate) / steadyElapsed.Seconds()) / (cfg.Training.HardwareTFLOPs * 1e12)
					mfuStr = fmt.Sprintf(" MFU=%.1f%%", mfu*100)
				}
				phaseStr := ""
				if hasPhases {
					phase := phaseSched.PhaseAt(step)
					phaseStr = fmt.Sprintf(" phase=%s", phaseDisplayLabel(phase, currentPhaseIdx))
				}
				fmt.Printf("  [%s] step %d/%d loss=%.4f%s lr=%.6f%s tok/s=%s%s %s\n",
					name, step, steps, v, valStr, appliedLR, phaseStr, formatTrainingThroughput(tokensPerSec), mfuStr, formatProgressTiming(time.Since(start), steadyElapsed, stepsForRate, step, steps))
				if masking := formatMLMMaskStatsForLog(currentDiagnosticBatch.mlmMaskStats); masking != "" {
					fmt.Printf("  [%s] [masking] %s\n", name, masking)
				}
				if extra := formatTrainingExtraDiagnostics(trainingExtra); extra != "" {
					fmt.Printf("  [%s] [ttt] %s\n", name, extra)
				}
				if beta := formatS4DSobolevDiagnostics(sobolevDiagnostics); beta != "" {
					fmt.Printf("  [%s] [s4d-sobolev] %s\n", name, beta)
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
					LR:            appliedLR,
					Objective:     currentObjective,
					SeqLen:        currentSeqLen,
					BatchTokens:   stepTokens,
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
					Masking:               currentDiagnosticBatch.mlmMaskStats.telemetry(),
					S4DSobolev:            sobolevDiagnostics,
				})
				if err := telemetry.writeSnapshot(true); err != nil {
					return TrainResult{}, err
				}
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

			if shouldWriteCheckpoint(step, opts.CheckpointEvery) {
				checkpointSchedule, err := resumeScheduleFrom(cfg.Training, sched, steps)
				if err != nil {
					return TrainResult{}, fmt.Errorf("snapshot live training schedule at step %d: %w", step+1, err)
				}
				stopCheckpoint := startSlowTrainingPhaseLogger(name, step, "checkpoint")
				artifacts, manifestPath, err := writeResumableCheckpoint(cfg, trainer, shapes, opts.CheckpointDir, step+1, resumableCheckpointContext{
					TrainPattern: trainPattern,
					Schedule:     checkpointSchedule,
					SWA:          swaEMA,
					Data2Vec:     data2vec,
					EarlyStop:    earlyStop,
				})
				if err != nil {
					stopCheckpoint()
					return TrainResult{}, fmt.Errorf("checkpoint at step %d: %w", step+1, err)
				}
				stopCheckpoint()
				fmt.Printf("  [%s] checkpoint saved: %s resume=%s\n", name, artifacts.Summary(), manifestPath)
			}

			// Plain steps may submit the next batch before collecting the
			// current loss. Boundary steps still submit here, after any
			// validation/checkpoint/logging that can flush trainer state.
			progress.afterStep(time.Now())
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
		finalSeqLen, finalBatchSize := seqLen, batchSize
		if cfg.Training.LengthBucketsChangeShape(seqLen) {
			finalBatchSize, finalSeqLen = lastTrainBatch.effectiveShape(currentBatchSize, currentSeqLen)
		}
		evalLoss, err := computeFinalTrainingLoss(cfg, lastTrainBatch, steps, finalSeqLen, finalBatchSize,
			trainer, causalEval, currentProgramKey, pairSampler, invarianceSampler, pllMarginSampler)
		if err != nil {
			return TrainResult{}, err
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
		runFullEvaluation(cfg, name, valPattern, valSet, trainer, causalEval, currentProgramKey, opts, steps, batchSize, seqLen)
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
		FLOPsReliable:    flops.TrainingFLOPsReliable,
	}
	fmt.Println(result.formatSummary())

	return result, nil
}
