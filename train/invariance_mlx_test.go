//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestInvarianceTinyTrainingSmokeAndDeterminism(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	first := runInvarianceOnlyTinySteps(t, 8)
	second := runInvarianceOnlyTinySteps(t, 8)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same seed and pairs produced different loss curves: %v vs %v", first, second)
	}
}

func TestInvariance500StepTrainingSmoke(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	losses := runInvarianceOnlyTinySteps(t, 500)
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("symmetric-KL loss did not decrease over 500 steps: first=%g last=%g", losses[0], losses[len(losses)-1])
	}
}

func TestInvarianceTrainingStepCachesExactComponentLoss(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	cfg := parseTrainInvarianceConfig(t, arch.ObjectiveMLM)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: arch.ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()
	objectiveTrainer, ok := trainer.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("initGPUTrainer returned %T, want *mlxGPUTrainer", trainer)
	}
	if err := objectiveTrainer.EnableComponentLossCapture(); err != nil {
		t.Fatalf("EnableComponentLossCapture: %v", err)
	}

	sampler := &invariancePairSampler{records: []invariancePairRecord{{
		ID: "p", Family: "distractor_agreement",
		ViewA: []int{1, 2, 3, 4}, ViewAPos: 2,
		ViewB: []int{1, 5, 3, 4}, ViewBPos: 2,
		viewAPosSet: true, viewBPosSet: true,
	}}}
	raw := trainBatch{x: []int{1, 2, 3, 4, 4, 3, 2, 1, 2, 3, 4, 5, 5, 4, 3, 2}, y: make([]int, 16)}
	for i := range raw.y {
		raw.y[i] = raw.x[i]
	}
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	batch, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepare objective batch: %v", err)
	}
	batch, err = maybeAttachInvariancePairs(sampler, cfg, 0, batch, batchSize, cfg.SeqLen, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("attach invariance pairs: %v", err)
	}
	if _, err := objectiveTrainer.TrainObjectiveStepGPU(batch, batchSize, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
		t.Fatalf("TrainObjectiveStepGPU: %v", err)
	}
	captured, err := objectiveTrainer.ReadComponentLossesGPU()
	if err != nil {
		t.Fatalf("ReadComponentLossesGPU: %v", err)
	}
	got, ok := captured["invariance_loss"]
	if !ok || !(got > 0) || math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("captured invariance_loss=%v, want finite positive", captured)
	}
	direct, err := objectiveTrainer.ReadOutput("invariance_loss", []int{1})
	if err != nil {
		t.Fatalf("ReadOutput(invariance_loss): %v", err)
	}
	if len(direct) != 1 || math.Abs(got-float64(direct[0])) > 1e-6 {
		t.Fatalf("captured invariance_loss=%g, direct IR output=%v", got, direct)
	}
}

func TestInvarianceTelemetryJSONLIncludesComponentLoss(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	dir := t.TempDir()
	pairPath := filepath.Join(dir, "pairs.bin")
	pairFile, err := os.Create(pairPath)
	if err != nil {
		t.Fatalf("create pairs: %v", err)
	}
	records := []invariancePairRecord{{
		ID: "p", Family: "distractor_agreement",
		ViewA: []int{1, 2, 3, 4}, ViewAPos: 2,
		ViewB: []int{1, 5, 3, 4}, ViewBPos: 2,
		viewAPosSet: true, viewBPosSet: true,
	}}
	if err := writeInvariancePairBinary(pairFile, records, 32, 4); err != nil {
		pairFile.Close()
		t.Fatalf("write pairs: %v", err)
	}
	if err := pairFile.Close(); err != nil {
		t.Fatalf("close pairs: %v", err)
	}
	trainDir := filepath.Join(dir, "data")
	if err := os.Mkdir(trainDir, 0o755); err != nil {
		t.Fatalf("create data dir: %v", err)
	}
	tokens := make([]uint16, 128)
	for i := range tokens {
		tokens[i] = uint16(i % 12)
	}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), tokens)

	cfg := parseTrainInvarianceConfig(t, arch.ObjectiveMLM)
	cfg.Training.Invariance.Path = pairPath
	telemetryPath := filepath.Join(dir, "run.telemetry.jsonl")
	rt, err := newTelemetryRuntime("", telemetryPath)
	if err != nil {
		t.Fatalf("newTelemetryRuntime: %v", err)
	}
	_, err = runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{
		LogEvery: 1, ValEvery: 0, telemetry: rt,
	})
	rt.Close()
	if err != nil {
		t.Fatalf("runTrain: %v", err)
	}
	raw, err := os.ReadFile(telemetryPath)
	if err != nil {
		t.Fatalf("read telemetry: %v", err)
	}
	var snap telemetrySnapshot
	if err := json.Unmarshal(raw, &snap); err != nil {
		t.Fatalf("decode telemetry: %v", err)
	}
	got, ok := snap.ComponentLosses["invariance_loss"]
	if !ok || !(got > 0) || math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("telemetry component_losses=%v, want finite invariance_loss", snap.ComponentLosses)
	}
}

func runInvarianceOnlyTinySteps(t *testing.T, steps int) []float32 {
	t.Helper()
	cfg := parseTrainInvarianceConfig(t, arch.ObjectiveMLM)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: arch.ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()
	objectiveTrainer, ok := trainer.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("initGPUTrainer returned %T, want *mlxGPUTrainer", trainer)
	}
	before, err := objectiveTrainer.ReadWeights()
	if err != nil {
		t.Fatalf("read initial weights: %v", err)
	}
	sampler := &invariancePairSampler{records: []invariancePairRecord{{
		ID: "p", Family: "distractor_agreement",
		ViewA: []int{1, 2, 3, 4}, ViewAPos: 2,
		ViewB: []int{1, 5, 3, 4}, ViewBPos: 2,
		viewAPosSet: true, viewBPosSet: true,
	}}}
	raw := trainBatch{x: []int{1, 2, 3, 4, 4, 3, 2, 1, 2, 3, 4, 5, 5, 4, 3, 2}, y: make([]int, 16)}
	for i := range raw.y {
		raw.y[i] = raw.x[i]
	}
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	losses := make([]float32, 0, steps)
	for step := 0; step < steps; step++ {
		batch, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("prepare step %d: %v", step, err)
		}
		batch, err = maybeAttachInvariancePairs(sampler, cfg, step, batch, batchSize, cfg.SeqLen, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("attach step %d: %v", step, err)
		}
		// Isolate the symmetric-KL path: pair rows are still fed through both
		// views, but the ordinary MLM loss is zeroed for this gradient check.
		clear(batch.lossMask)
		loss, err := objectiveTrainer.TrainObjectiveStepGPU(batch, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
		}
		if !(loss > 0) || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
		losses = append(losses, loss)
	}
	after, err := objectiveTrainer.ReadWeights()
	if err != nil {
		t.Fatalf("read final weights: %v", err)
	}
	if !weightSlicesDiffer(before, after) {
		t.Fatal("symmetric-KL-only steps did not update any model weight")
	}
	return losses
}

func weightSlicesDiffer(before, after [][]float32) bool {
	if len(before) != len(after) {
		return true
	}
	for i := range before {
		if len(before[i]) != len(after[i]) {
			return true
		}
		for j := range before[i] {
			if before[i][j] != after[i][j] {
				return true
			}
		}
	}
	return false
}
