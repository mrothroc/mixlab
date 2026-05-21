package train

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

// LRSchedule defines a cosine learning rate schedule with warmup and hold.
type LRSchedule struct {
	BaseLR             float32
	MinLR              float32
	Warmup             int
	Hold               int
	Warmdown           int
	MaxSteps           int
	ClampWarmdownToMin bool
}

type trainingScheduler interface {
	At(step int) float32
}

type phaseSchedule struct {
	lrs        []float32
	phaseIndex []int
	phases     []TrainingPhase
}

// At returns the learning rate at the given step.
func (s LRSchedule) At(step int) float32 {
	baseAt := func(step int) float32 {
		if step < s.Warmup {
			if s.Warmup == 0 {
				return s.BaseLR
			}
			return s.BaseLR * float32(step) / float32(s.Warmup)
		}
		if step < s.Warmup+s.Hold {
			return s.BaseLR
		}
		decaySteps := s.MaxSteps - s.Warmup - s.Hold
		if decaySteps <= 0 {
			return s.BaseLR
		}
		progress := float64(step-s.Warmup-s.Hold) / float64(decaySteps)
		if progress > 1.0 {
			progress = 1.0
		}
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return s.MinLR + (s.BaseLR-s.MinLR)*float32(cosine)
	}

	lr := baseAt(step)
	if s.Warmdown <= 0 || s.MaxSteps <= 0 {
		return lr
	}
	warmdownStart := s.MaxSteps - s.Warmdown
	if warmdownStart < 0 {
		warmdownStart = 0
	}
	if step < warmdownStart {
		return lr
	}
	startLR := baseAt(warmdownStart)
	targetLR := s.MinLR / 10
	if s.ClampWarmdownToMin && targetLR < s.MinLR {
		targetLR = s.MinLR
	}
	progress := float32(step-warmdownStart) / float32(s.Warmdown)
	if progress > 1 {
		progress = 1
	}
	return startLR + (targetLR-startLR)*progress
}

// At returns the per-step LR for a phase-based schedule.
func (s phaseSchedule) At(step int) float32 {
	if len(s.lrs) == 0 {
		return 0
	}
	if step < 0 {
		step = 0
	}
	if step >= len(s.lrs) {
		step = len(s.lrs) - 1
	}
	return s.lrs[step]
}

func (s phaseSchedule) PhaseAt(step int) TrainingPhase {
	if len(s.phases) == 0 {
		return TrainingPhase{}
	}
	if step < 0 {
		step = 0
	}
	if step >= len(s.phaseIndex) {
		step = len(s.phaseIndex) - 1
	}
	idx := s.phaseIndex[step]
	if idx < 0 || idx >= len(s.phases) {
		return TrainingPhase{}
	}
	return s.phases[idx]
}

// trainingSchedule constructs the standard LR schedule from base LR and total steps.
func trainingSchedule(lr float32, steps, warmdown int, minLRFraction float32) LRSchedule {
	if warmdown < 0 {
		warmdown = 0
	}
	if warmdown > steps {
		warmdown = steps
	}
	warmup := 100
	if steps < warmup {
		warmup = steps
	}
	hold := 200
	if steps < warmup+hold {
		hold = steps - warmup
		if hold < 0 {
			hold = 0
		}
	}
	minLR := lr * 0.1
	if minLRFraction > 0 {
		minLR = lr * minLRFraction
	}
	return LRSchedule{
		BaseLR:             lr,
		MinLR:              minLR,
		Warmup:             warmup,
		Hold:               hold,
		Warmdown:           warmdown,
		MaxSteps:           steps,
		ClampWarmdownToMin: minLRFraction > 0,
	}
}

func newPhaseSchedule(phases []TrainingPhase, warmdown int, minLRFraction float32) phaseSchedule {
	totalSteps := 0
	for _, phase := range phases {
		totalSteps += phase.Steps
	}
	sched := phaseSchedule{
		lrs:        make([]float32, totalSteps),
		phaseIndex: make([]int, totalSteps),
		phases:     append([]TrainingPhase(nil), phases...),
	}
	offset := 0
	for phaseIdx, phase := range phases {
		lr := float32(phase.LR)
		for i := 0; i < phase.Steps; i++ {
			sched.lrs[offset+i] = lr
			sched.phaseIndex[offset+i] = phaseIdx
		}
		offset += phase.Steps
	}
	if totalSteps == 0 || warmdown <= 0 || len(phases) == 0 {
		return sched
	}
	lastPhase := phases[len(phases)-1]
	lastPhaseWarmdown := warmdown
	if lastPhaseWarmdown > lastPhase.Steps {
		lastPhaseWarmdown = lastPhase.Steps
	}
	if lastPhaseWarmdown <= 0 {
		return sched
	}
	warmdownStart := totalSteps - lastPhaseWarmdown
	startLR := sched.lrs[warmdownStart]
	targetLR := float32(lastPhase.LR) * 0.01
	if minLRFraction > 0 {
		minLR := float32(lastPhase.LR) * minLRFraction
		if targetLR < minLR {
			targetLR = minLR
		}
	}
	for step := warmdownStart; step < totalSteps; step++ {
		progress := float32(step-warmdownStart) / float32(lastPhaseWarmdown)
		if progress > 1 {
			progress = 1
		}
		sched.lrs[step] = startLR + (targetLR-startLR)*progress
	}
	return sched
}

func buildTrainingScheduler(spec TrainingSpec) (trainingScheduler, int) {
	if len(spec.Phases) > 0 {
		totalSteps := spec.TotalSteps()
		return newPhaseSchedule(spec.Phases, spec.WarmdownSteps, spec.MinLRFraction), totalSteps
	}
	return trainingSchedule(float32(spec.LR), spec.Steps, spec.WarmdownSteps, spec.MinLRFraction), spec.Steps
}

func phaseDisplayLabel(phase TrainingPhase, index int) string {
	if strings.TrimSpace(phase.Label) != "" {
		return phase.Label
	}
	return fmt.Sprintf("phase-%d", index+1)
}

// TrainResult holds the outcome of a training run.
type TrainResult struct {
	Name      string
	FirstLoss float64
	// LastLoss is the value of the IR "loss" output collected at the final
	// training step — the quantity the optimizer was actually minimising. This
	// is masked cross-entropy when training.first_byte_mask is enabled, the
	// MTP-weighted multi-token loss when mtp.n>1, otherwise the standard
	// next-token cross-entropy.
	LastLoss float64
	// LastUnmaskedLoss is the unmasked next-token cross-entropy of the trained
	// model on the final training batch, measured by an extra forward pass
	// after the last optimizer update. This is commensurate across runs
	// regardless of whether training.first_byte_mask or MTP is on, so it can
	// be compared directly between configurations. NaN when training did not
	// run (steps == 0).
	LastUnmaskedLoss float64
	LastValLoss      float64
	HasValLoss       bool
	Delta            float64
	Elapsed          time.Duration
	StepFLOPs        int64
	FLOPsPerTok      int64
}

// formatSummary returns a one-line summary of the training result.
func (r TrainResult) formatSummary() string {
	if r.HasValLoss {
		return fmt.Sprintf("%-12s first=%.4f last=%.4f val=%.4f delta=%.4f (%s)",
			r.Name, r.FirstLoss, r.LastLoss, r.LastValLoss, r.Delta, r.Elapsed.Round(time.Millisecond))
	}
	return fmt.Sprintf("%-12s first=%.4f last=%.4f delta=%.4f (%s)",
		r.Name, r.FirstLoss, r.LastLoss, r.Delta, r.Elapsed.Round(time.Millisecond))
}

// GPUTrainer abstracts the GPU training interface.
// This is implemented by the MLX backend when available.
type GPUTrainer interface {
	TrainStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) (float32, error)
	SubmitStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) error
	CollectLossGPU() (float32, error)
	FlushGPU() error
	SetQATGPU(mode string) error
	SetWeightGPU(name string, data []float32) error
	EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error)
	EvaluatePerTokenGPU(xTok, yTok []int, batchSize, seqLen int) ([]float32, error)
	EvaluateLoRATTTGPU(xTok, yTok []int, batchSize, seqLen, tttSteps int, tttLR float32, tttRank int) (float32, error)
	CloseTrainer()
}

type gpuProgramSwitcher interface {
	SetProgramGPU(*arch.Program) error
}

type gpuWeightCopier interface {
	CopyWeightGPU(dstName, srcName string) error
}

// TrainOptions holds optional parameters for runTrain.
type TrainOptions struct {
	SafetensorsPath string // If set, export weights after training
	SafetensorsLoad string // If set, load weights from safetensors file before training
	Quantize        string // Quantization mode: "none", "int8", or "int6"
	QuantMethod     string // Quantization clipping method: "quantile" or "sdclip"
	QuantK          float32
	QuantKEmbed     float32
	DoFullEval      bool   // If true, run full BPB evaluation after training
	LUTDir          string // Directory containing BPB lookup tables
	CheckpointDir   string // Directory for periodic safetensors checkpoints
	CheckpointEvery int    // Save checkpoint every N steps; 0 disables
	LogEvery        int    // Print progress every N steps; 0 uses default/env cadence
	ValEvery        int    // Run validation every N steps; 0 uses default/env cadence
	Timing          bool   // If true, print per-step timing breakdown at log intervals

	// OptimizerOverride lets callers customize the optimizer plan that RunArch
	// builds before the GPU trainer is created.
	//
	// The callback receives the auto-generated default plan and the weight shapes
	// that plan was built from, and returns a replacement plan. The returned plan
	// must cover every weight exactly once; RunArch validates this and returns an
	// error otherwise.
	//
	// Example:
	//
	//	OptimizerOverride: func(spec gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
	//		frozen := len(spec.Groups)
	//		spec.Groups = append(spec.Groups, gpu.OptimizerGroup{
	//			Kind: gpu.OptimizerAdamW,
	//			LR:   0,
	//		})
	//		for i, shape := range shapes {
	//			if shape.Name == "embed" {
	//				spec.Weights[i].GroupIndex = frozen
	//			}
	//		}
	//		return spec, nil
	//	}
	//
	// Most callers should leave this nil.
	OptimizerOverride func(defaultPlan gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error)
}

type trainBatch struct {
	x, y []int
	err  error
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
	targetValLoss := cfg.Training.TargetValLoss
	shuffleChunkTokens := effectiveShuffleChunkTokens(cfg)
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
	initialProg := prog
	if recurrencePhasesScheduled || recurrenceScheduled || (mtpUntieScheduled && !headUntied) || (mtpAuxActivationScheduled && !mtpAuxActive) {
		state := TrainingProgramState{
			RecurrenceActive: recurrenceActive,
			HeadUntied:       headUntied,
			MTPAuxInactive:   !mtpAuxActive,
		}
		if recurrencePhasesScheduled {
			initialProg, err = arch.BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, currentRecurrencePhase, state)
		} else {
			initialProg, err = BuildTrainingIRProgramFromConfig(cfg, state)
		}
		if err != nil {
			return TrainResult{}, fmt.Errorf("build initial training IR program: %w", err)
		}
		if initialProg.NumWeights != prog.NumWeights {
			return TrainResult{}, fmt.Errorf("initial training weight count mismatch: initial=%d final=%d", initialProg.NumWeights, prog.NumWeights)
		}
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

	var shapes []WeightShape
	if opts.SafetensorsPath != "" || opts.CheckpointEvery > 0 {
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

	// Initialize GPU trainer
	trainer, err := initGPUTrainer(initialProg, cfg, loadedWeights, opts.OptimizerOverride)
	if err != nil {
		return TrainResult{}, fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()

	var swaEMA [][]float32
	if swaStart > 0 {
		swaEMA = make([][]float32, prog.NumWeights)
	}

	// Create data loader
	loader, err := data.NewLoader(trainPattern, seed, shuffleChunkTokens)
	if err != nil {
		return TrainResult{}, err
	}

	// Load validation set
	const defaultValBatchCount = 10
	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	valSet, valErr := data.NewValSet(valPattern, seed, defaultValBatchCount, batchTokens, seqLen, shuffleChunkTokens)
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
	stepLookaheadEnabled := !envTruthy("MIXLAB_DISABLE_GPU_STEP_LOOKAHEAD")
	start := time.Now()
	// steadyStart is set after step 0 completes — excludes one-time
	// compile/warmup costs from tok/s and ETA estimates.
	var steadyStart time.Time
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
		if err := trainer.SubmitStepGPU(batch.x, batch.y, batchSize, seqLen, sched.At(0)); err != nil {
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
			if nextRecurrencePhase != currentRecurrencePhase || nextRecurrenceActive != recurrenceActive || nextHeadUntied != headUntied || nextMTPAuxActive != mtpAuxActive {
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
				state := TrainingProgramState{
					RecurrenceActive: nextRecurrenceActive,
					HeadUntied:       nextHeadUntied,
					MTPAuxInactive:   !nextMTPAuxActive,
				}
				var nextProg *arch.Program
				if recurrencePhasesScheduled {
					nextProg, err = arch.BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, nextRecurrencePhase, state)
				} else {
					nextProg, err = BuildTrainingIRProgramFromConfig(cfg, state)
				}
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
				currentRecurrencePhase = nextRecurrencePhase
				recurrenceActive = nextRecurrenceActive
				headUntied = nextHeadUntied
				mtpAuxActive = nextMTPAuxActive
			}
			submitStart := time.Now()
			stopSubmit := startSlowTrainingPhaseLogger(name, nextStep, "submit_step")
			if err := trainer.SubmitStepGPU(batch.x, batch.y, batchSize, seqLen, sched.At(nextStep)); err != nil {
				stopSubmit()
				return 0, fmt.Errorf("submit step %d: %w", nextStep, err)
			}
			stopSubmit()
			lastTrainBatch = batch
			return time.Since(submitStart), nil
		}
		canSubmitNextBeforeCollect := func(step int) bool {
			if !stepLookaheadEnabled ||
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

			if step == 0 {
				firstLoss = v
				// Anchor steady-state timing after the first step so the
				// one-time compile/warmup cost doesn't poison tok/s and ETA.
				steadyStart = time.Now()
			}
			lastLoss = v

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
				if valSet != nil && len(valSet.Batches) > 0 && shouldRunValidationStep(step, steps, valEvery) {
					valStart := time.Now()
					stopValidation := startSlowTrainingPhaseLogger(name, step, "validation")
					valAvg, err := meanValidationLoss(valSet, trainer, batchSize, seqLen)
					stopValidation()
					valDuration = time.Since(valStart)
					if err == nil {
						lastValLoss = valAvg
						hasValLoss = true
						valStr = fmt.Sprintf(" val=%.4f", valAvg)
					}
				}
				// Use wall-clock average for tok/s and MFU — per-step EMA is
				// unreliable with pipelined training because collect returns
				// near-instantly when the GPU has already finished.
				// Anchor the rate calculation at steadyStart (set after step 0)
				// so the one-time compile/warmup cost doesn't skew estimates.
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
				logDuration := time.Since(logStart)
				if opts.Timing {
					fmt.Printf("  [%s] [timing] data=%.1fms gpu=%.1fms val=%.1fms log=%.1fms\n",
						name,
						float64(dataDuration)/float64(time.Millisecond),
						float64(gpuDuration)/float64(time.Millisecond),
						float64(valDuration)/float64(time.Millisecond),
						float64(logDuration)/float64(time.Millisecond))
				}
				if hasValLoss && targetValLoss > 0 && lastValLoss <= targetValLoss {
					fmt.Printf("  [%s] target val loss %.4f reached at step %d (val=%.4f), stopping early\n",
						name, targetValLoss, step, lastValLoss)
					break
				}
			}

			if shouldWriteCheckpoint(step, opts.CheckpointEvery) {
				stopCheckpoint := startSlowTrainingPhaseLogger(name, step, "checkpoint")
				if err := writeCheckpoint(cfg, trainer, shapes, opts.CheckpointDir, step+1); err != nil {
					stopCheckpoint()
					return TrainResult{}, fmt.Errorf("checkpoint at step %d: %w", step+1, err)
				}
				stopCheckpoint()
				fmt.Printf("  [%s] checkpoint saved: %s\n", name, checkpointPath(opts.CheckpointDir, step+1))
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
		evalLoss, err := trainer.EvaluateGPU(lastTrainBatch.x, lastTrainBatch.y, batchSize, seqLen)
		if err != nil {
			return TrainResult{}, fmt.Errorf("evaluate final unmasked training loss: %w", err)
		}
		lastUnmaskedLoss = float64(evalLoss)
	}

	// Export safetensors if requested (before closing trainer)
	if opts.SafetensorsPath != "" {
		weights, err := exportWeightsForTrainer(trainer, swaEMA)
		if err != nil {
			fmt.Printf("  [%s] safetensors export skipped (read error): %v\n", name, err)
		} else {
			qmode := opts.Quantize
			if qmode == "int8" || qmode == "int6" {
				if err := exportSafetensorsQuantized(opts.SafetensorsPath, cfg, shapes, weights, qmode, opts.QuantMethod, opts.QuantK, opts.QuantKEmbed); err != nil {
					fmt.Printf("  [%s] quantized safetensors export failed: %v\n", name, err)
				}
			} else {
				if err := exportSafetensors(opts.SafetensorsPath, cfg, shapes, weights); err != nil {
					fmt.Printf("  [%s] safetensors export failed: %v\n", name, err)
				}
			}
		}
	}

	// Full BPB evaluation if requested
	if opts.DoFullEval {
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
		if err := runFullEval(cfg, valPattern, trainer, lutDir); err != nil {
			fmt.Printf("  [%s] full validation BPB failed: %v\n", name, err)
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

// meanValidationLoss computes the mean loss across validation batches.
func meanValidationLoss(valSet *data.ValSet, trainer GPUTrainer, batchSize, seqLen int) (float64, error) {
	return meanValidationLossWithTTT(valSet, trainer, batchSize, seqLen, "full", 0, 0, 0)
}

// meanValidationLossWithTTT computes score-first validation loss and, when
// tttSteps > 0, adapts weights after each scored batch.
func meanValidationLossWithTTT(
	valSet *data.ValSet,
	trainer GPUTrainer,
	batchSize, seqLen int,
	tttMode string,
	tttSteps int,
	tttLR float32,
	tttRank int,
) (float64, error) {
	if valSet == nil || len(valSet.Batches) == 0 {
		return 0, fmt.Errorf("no validation batches")
	}
	if tttSteps < 0 {
		return 0, fmt.Errorf("ttt_steps must be >= 0")
	}
	if tttMode == "" {
		tttMode = "full"
	}
	if tttMode != "full" && tttMode != "lora" {
		return 0, fmt.Errorf("ttt_mode must be \"full\" or \"lora\"")
	}
	if tttMode == "lora" && tttRank <= 0 {
		return 0, fmt.Errorf("ttt_rank must be > 0")
	}
	sum := 0.0
	count := 0
	failures := 0
	for _, vb := range valSet.Batches {
		var (
			loss float32
			err  error
		)
		if tttMode == "lora" && tttSteps > 0 {
			loss, err = trainer.EvaluateLoRATTTGPU(vb.X, vb.Y, batchSize, seqLen, tttSteps, tttLR, tttRank)
		} else {
			loss, err = trainer.EvaluateGPU(vb.X, vb.Y, batchSize, seqLen)
		}
		if err != nil {
			failures++
			continue
		}
		sum += float64(loss)
		count++
		if tttMode == "full" {
			for step := 0; step < tttSteps; step++ {
				if _, err := trainer.TrainStepGPU(vb.X, vb.Y, batchSize, seqLen, tttLR); err != nil {
					return 0, fmt.Errorf("ttt step %d after val batch %d: %w", step+1, count, err)
				}
			}
		}
	}
	if count == 0 {
		return 0, fmt.Errorf("validation evaluation failed for all %d batches", len(valSet.Batches))
	}
	if failures > 0 {
		fmt.Printf("  warning: %d/%d val batches failed, using %d successful\n", failures, len(valSet.Batches), count)
	}
	return sum / float64(count), nil
}

func exportWeightsForTrainer(trainer GPUTrainer, swaEMA [][]float32) ([][]float32, error) {
	if hasSWAWeights(swaEMA) {
		return cloneWeights(swaEMA), nil
	}
	return readTrainerWeights(trainer)
}

// readTrainerWeights reads weights from a trainer via the weight-reading interface.
// Falls back gracefully if the trainer doesn't support weight reading.
func readTrainerWeights(trainer GPUTrainer) ([][]float32, error) {
	type weightReader interface {
		ReadWeights() ([][]float32, error)
	}
	if wr, ok := trainer.(weightReader); ok {
		return wr.ReadWeights()
	}
	return nil, fmt.Errorf("trainer does not support weight reading; ensure you are using the MLX backend")
}

func readTrainerOutput(trainer GPUTrainer, name string, shape []int) ([]float32, error) {
	type outputReader interface {
		ReadOutput(name string, shape []int) ([]float32, error)
	}
	if or, ok := trainer.(outputReader); ok {
		return or.ReadOutput(name, shape)
	}
	return nil, fmt.Errorf("trainer does not support reading named outputs; ensure you are using the MLX backend")
}

func checkpointPath(dir string, step int) string {
	return filepath.Join(dir, fmt.Sprintf("step_%06d.st", step))
}

func writeCheckpoint(cfg *ArchConfig, trainer GPUTrainer, shapes []WeightShape, dir string, step int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create checkpoint dir %q: %w", dir, err)
	}
	weights, err := readTrainerWeights(trainer)
	if err != nil {
		return fmt.Errorf("read trainer weights: %w", err)
	}
	path := checkpointPath(dir, step)
	if err := exportSafetensors(path, cfg, shapes, weights); err != nil {
		return fmt.Errorf("export safetensors %q: %w", path, err)
	}
	return nil
}

func formatProgressTiming(elapsed, steadyElapsed time.Duration, stepsForRate, step, totalSteps int) string {
	if step < 1 || totalSteps <= 0 || stepsForRate < 1 || steadyElapsed <= 0 {
		return fmt.Sprintf("(%.1fs)", elapsed.Seconds())
	}
	// ETA uses steady-state rate (post-warmup) so the one-time compile cost
	// doesn't dominate early estimates.
	avgStepDuration := steadyElapsed / time.Duration(stepsForRate)
	remainingSteps := totalSteps - (step + 1)
	if remainingSteps < 0 {
		remainingSteps = 0
	}
	eta := time.Duration(remainingSteps) * avgStepDuration
	return fmt.Sprintf("(%.1fs, ~%s remaining)", elapsed.Seconds(), eta.Round(time.Second))
}
