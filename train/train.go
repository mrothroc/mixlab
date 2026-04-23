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
	EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error)
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
}

type trainBatch struct {
	x, y []int
	err  error
}

const stepDurationEMADecay = 0.95

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

	steps := cfg.Training.Steps
	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	swaStart := cfg.Training.SWAStart
	swaDecay := cfg.Training.SWADecay
	swaInterval := cfg.Training.SWAInterval
	if batchTokens%seqLen != 0 {
		return TrainResult{}, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	batchSize := batchTokens / seqLen
	lr := float32(cfg.Training.LR)
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

	sched := trainingSchedule(lr, steps, cfg.Training.WarmdownSteps)

	var firstLoss, lastLoss float64
	lastValLoss := math.NaN()
	hasValLoss := false
	start := time.Now()
	var stepDurationEMA time.Duration
	done := make(chan struct{})
	batchCh := make(chan trainBatch, 2)
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

	for step := 0; step < steps; step++ {
		stepStart := time.Now()
		batch, ok := <-batchCh
		if !ok {
			return TrainResult{}, fmt.Errorf("load batch at step %d: prefetch pipeline closed unexpectedly", step)
		}
		if batch.err != nil {
			return TrainResult{}, fmt.Errorf("load batch at step %d: %w", step, batch.err)
		}

		lossV, err := trainer.TrainStepGPU(batch.x, batch.y, batchSize, seqLen, sched.At(step))
		if err != nil {
			return TrainResult{}, fmt.Errorf("train step %d: %w", step, err)
		}
		v := float64(lossV)

		if step == 0 {
			firstLoss = v
		}
		lastLoss = v

		stepDuration := time.Since(stepStart)
		if step == 0 {
			stepDurationEMA = stepDuration
		} else {
			stepDurationEMA = time.Duration(stepDurationEMADecay*float64(stepDurationEMA) + (1-stepDurationEMADecay)*float64(stepDuration))
		}

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
			valStr := ""
			if valSet != nil && len(valSet.Batches) > 0 {
				valAvg, err := meanValidationLoss(valSet, trainer, batchSize, seqLen)
				if err == nil {
					lastValLoss = valAvg
					hasValLoss = true
					valStr = fmt.Sprintf(" val=%.4f", valAvg)
				}
			}
			tokensPerSec := float64(batchTokens) / stepDuration.Seconds()
			mfuStr := ""
			if cfg.Training.HardwareTFLOPs > 0 && flops.TrainingFLOPs > 0 {
				mfu := (float64(flops.TrainingFLOPs) / stepDuration.Seconds()) / (cfg.Training.HardwareTFLOPs * 1e12)
				mfuStr = fmt.Sprintf(" MFU=%.1f%%", mfu*100)
			}
			fmt.Printf("  [%s] step %d/%d loss=%.4f%s lr=%.6f tok/s=%.0f%s %s\n",
				name, step, steps, v, valStr, sched.At(step), tokensPerSec, mfuStr, formatProgressTiming(time.Since(start), stepDurationEMA, step, steps))
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
			fmt.Printf("  [%s] computing full validation BPB with score-first TTT (steps=%d lr=%g)...\n", name, cfg.Training.TTTSteps, cfg.Training.TTTLR)
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
	return meanValidationLossWithTTT(valSet, trainer, batchSize, seqLen, 0, 0)
}

// meanValidationLossWithTTT computes score-first validation loss and, when
// tttSteps > 0, adapts weights after each scored batch.
func meanValidationLossWithTTT(valSet *data.ValSet, trainer GPUTrainer, batchSize, seqLen int, tttSteps int, tttLR float32) (float64, error) {
	if valSet == nil || len(valSet.Batches) == 0 {
		return 0, fmt.Errorf("no validation batches")
	}
	if tttSteps < 0 {
		return 0, fmt.Errorf("ttt_steps must be >= 0")
	}
	sum := 0.0
	count := 0
	failures := 0
	for _, vb := range valSet.Batches {
		loss, err := trainer.EvaluateGPU(vb.X, vb.Y, batchSize, seqLen)
		if err != nil {
			failures++
			continue
		}
		sum += float64(loss)
		count++
		for step := 0; step < tttSteps; step++ {
			if _, err := trainer.TrainStepGPU(vb.X, vb.Y, batchSize, seqLen, tttLR); err != nil {
				return 0, fmt.Errorf("ttt step %d after val batch %d: %w", step+1, count, err)
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

func formatProgressTiming(elapsed, stepDurationEMA time.Duration, step, totalSteps int) string {
	if step < 10 || stepDurationEMA <= 0 || totalSteps <= 0 {
		return fmt.Sprintf("(%.1fs)", elapsed.Seconds())
	}
	remainingSteps := totalSteps - (step + 1)
	if remainingSteps < 0 {
		remainingSteps = 0
	}
	eta := time.Duration(remainingSteps) * stepDurationEMA
	return fmt.Sprintf("(%.1fs, ~%s remaining)", elapsed.Seconds(), eta.Round(time.Second))
}
