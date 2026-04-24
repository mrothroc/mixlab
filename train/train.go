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
)

// LRSchedule defines a cosine learning rate schedule with warmup and hold.
type LRSchedule struct {
	BaseLR   float32
	MinLR    float32
	Warmup   int
	Hold     int
	Warmdown int
	MaxSteps int
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
func trainingSchedule(lr float32, steps, warmdown int) LRSchedule {
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
	return LRSchedule{
		BaseLR:   lr,
		MinLR:    lr * 0.1,
		Warmup:   warmup,
		Hold:     hold,
		Warmdown: warmdown,
		MaxSteps: steps,
	}
}

func newPhaseSchedule(phases []TrainingPhase, warmdown int) phaseSchedule {
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
		return newPhaseSchedule(spec.Phases, spec.WarmdownSteps), totalSteps
	}
	return trainingSchedule(float32(spec.LR), spec.Steps, spec.WarmdownSteps), spec.Steps
}

func phaseDisplayLabel(phase TrainingPhase, index int) string {
	if strings.TrimSpace(phase.Label) != "" {
		return phase.Label
	}
	return fmt.Sprintf("phase-%d", index+1)
}

// TrainResult holds the outcome of a training run.
type TrainResult struct {
	Name        string
	FirstLoss   float64
	LastLoss    float64
	LastValLoss float64
	HasValLoss  bool
	Delta       float64
	Elapsed     time.Duration
	StepFLOPs   int64
	FLOPsPerTok int64
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
	EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error)
	EvaluateLoRATTTGPU(xTok, yTok []int, batchSize, seqLen, tttSteps int, tttLR float32, tttRank int) (float32, error)
	CloseTrainer()
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
	Timing          bool   // If true, print per-step timing breakdown at log intervals
}

type trainBatch struct {
	x, y []int
	err  error
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

	// Build IR program
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return TrainResult{}, fmt.Errorf("build IR program: %w", err)
	}
	fmt.Printf("  [%s] IR program: %d ops, %d weights\n", name, len(prog.Ops), prog.NumWeights)

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
	trainer, err := initGPUTrainer(prog, cfg, loadedWeights)
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
	lastValLoss := math.NaN()
	hasValLoss := false
	start := time.Now()
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
		batch, ok := <-batchCh
		initialDataDuration := time.Since(initialWaitStart)
		if !ok {
			return TrainResult{}, fmt.Errorf("load initial batch: prefetch pipeline closed unexpectedly")
		}
		if batch.err != nil {
			return TrainResult{}, fmt.Errorf("load initial batch: %w", batch.err)
		}
		initialSubmitStart := time.Now()
		if err := trainer.SubmitStepGPU(batch.x, batch.y, batchSize, seqLen, sched.At(0)); err != nil {
			return TrainResult{}, fmt.Errorf("submit step 0: %w", err)
		}
		currentSubmitDuration := time.Since(initialSubmitStart)
		currentPhaseIdx := -1

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
				nextBatch, ok = <-batchCh
				dataDuration = time.Since(batchWaitStart)
				if !ok {
					return TrainResult{}, fmt.Errorf("load batch at step %d: prefetch pipeline closed unexpectedly", step+1)
				}
				if nextBatch.err != nil {
					return TrainResult{}, fmt.Errorf("load batch at step %d: %w", step+1, nextBatch.err)
				}
			}

			collectStart := time.Now()
			lossV, err := trainer.CollectLossGPU()
			gpuDuration := currentSubmitDuration + time.Since(collectStart)
			if err != nil {
				return TrainResult{}, fmt.Errorf("collect loss at step %d: %w", step, err)
			}
			v := float64(lossV)

			if step == 0 {
				firstLoss = v
			}
			lastLoss = v

			if shouldUpdateSWA(step, swaStart, swaInterval) {
				weights, err := readTrainerWeights(trainer)
				if err != nil {
					return TrainResult{}, fmt.Errorf("swa read at step %d: %w", step, err)
				}
				if !hasSWAWeights(swaEMA) {
					fmt.Printf("  [%s] SWA: averaging started at step %d\n", name, step)
				}
				updateEMAWeights(swaEMA, weights, swaDecay)
			}

			// Log every 100 steps or at the last step
			if step%100 == 0 || step == steps-1 {
				logStart := time.Now()
				valDuration := time.Duration(0)
				valStr := ""
				if valSet != nil && len(valSet.Batches) > 0 {
					valStart := time.Now()
					valAvg, err := meanValidationLoss(valSet, trainer, batchSize, seqLen)
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
				elapsed := time.Since(start)
				tokensPerSec := float64(batchTokens) * float64(step+1) / elapsed.Seconds()
				mfuStr := ""
				if cfg.Training.HardwareTFLOPs > 0 && flops.TrainingFLOPs > 0 {
					mfu := (float64(flops.TrainingFLOPs) * float64(step+1) / elapsed.Seconds()) / (cfg.Training.HardwareTFLOPs * 1e12)
					mfuStr = fmt.Sprintf(" MFU=%.1f%%", mfu*100)
				}
				phaseStr := ""
				if hasPhases {
					phase := phaseSched.PhaseAt(step)
					phaseStr = fmt.Sprintf(" phase=%s", phaseDisplayLabel(phase, currentPhaseIdx))
				}
				fmt.Printf("  [%s] step %d/%d loss=%.4f%s lr=%.6f%s tok/s=%.0f%s %s\n",
					name, step, steps, v, valStr, sched.At(step), phaseStr, tokensPerSec, mfuStr, formatProgressTiming(time.Since(start), step, steps))
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
				if err := writeCheckpoint(cfg, trainer, shapes, opts.CheckpointDir, step+1); err != nil {
					return TrainResult{}, fmt.Errorf("checkpoint at step %d: %w", step+1, err)
				}
				fmt.Printf("  [%s] checkpoint saved: %s\n", name, checkpointPath(opts.CheckpointDir, step+1))
			}

			// Submit next step AFTER validation/checkpoint/logging so flush
			// inside EvaluateGPU doesn't discard the pending step.
			if step < steps-1 {
				submitStart := time.Now()
				if err := trainer.SubmitStepGPU(nextBatch.x, nextBatch.y, batchSize, seqLen, sched.At(step+1)); err != nil {
					return TrainResult{}, fmt.Errorf("submit step %d: %w", step+1, err)
				}
				currentSubmitDuration = time.Since(submitStart)
			}
		}
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
		Name:        name,
		FirstLoss:   firstLoss,
		LastLoss:    lastLoss,
		LastValLoss: lastValLoss,
		HasValLoss:  hasValLoss,
		Delta:       lastLoss - firstLoss,
		Elapsed:     elapsed,
		StepFLOPs:   flops.TrainingFLOPs,
		FLOPsPerTok: flops.FLOPsPerToken,
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

func shouldUpdateSWA(step, start, interval int) bool {
	return start > 0 && interval > 0 && step >= start && (step-start)%interval == 0
}

func hasSWAWeights(ema [][]float32) bool {
	for _, weight := range ema {
		if len(weight) != 0 {
			return true
		}
	}
	return false
}

func updateEMAWeights(ema, current [][]float32, decay float32) {
	oneMinusDecay := 1 - decay
	for i, weight := range current {
		if len(ema[i]) == 0 {
			ema[i] = append([]float32(nil), weight...)
			continue
		}
		for j, value := range weight {
			ema[i][j] = decay*ema[i][j] + oneMinusDecay*value
		}
	}
}

func exportWeightsForTrainer(trainer GPUTrainer, swaEMA [][]float32) ([][]float32, error) {
	if hasSWAWeights(swaEMA) {
		return cloneWeights(swaEMA), nil
	}
	return readTrainerWeights(trainer)
}

func cloneWeights(weights [][]float32) [][]float32 {
	cloned := make([][]float32, len(weights))
	for i, weight := range weights {
		cloned[i] = append([]float32(nil), weight...)
	}
	return cloned
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

func shouldWriteCheckpoint(step, every int) bool {
	return every > 0 && (step+1)%every == 0
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

func formatProgressTiming(elapsed time.Duration, step, totalSteps int) string {
	if step < 1 || totalSteps <= 0 {
		return fmt.Sprintf("(%.1fs)", elapsed.Seconds())
	}
	// Wall-clock ETA: time_so_far / steps_done * steps_remaining
	avgStepDuration := elapsed / time.Duration(step+1)
	remainingSteps := totalSteps - (step + 1)
	if remainingSteps < 0 {
		remainingSteps = 0
	}
	eta := time.Duration(remainingSteps) * avgStepDuration
	return fmt.Sprintf("(%.1fs, ~%s remaining)", elapsed.Seconds(), eta.Round(time.Second))
}
