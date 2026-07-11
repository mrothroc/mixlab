//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestPLLMarginHighWeightLAMBStaysFinite(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	for _, tc := range []struct {
		weight float64
		margin float64
		steps  int
	}{
		{weight: 0.5, margin: 1, steps: 128},
		{weight: 1, margin: 2, steps: 3000},
		{weight: 2, margin: 1, steps: 128},
	} {
		t.Run(fmt.Sprintf("weight_%g_margin_%g", tc.weight, tc.margin), func(t *testing.T) {
			cfg := parseTrainPLLMarginConfig(t, arch.ObjectiveMNTP)
			cfg.Training.PLLMargin.Weight = tc.weight
			cfg.Training.PLLMargin.Margin = tc.margin
			cfg.Training.Optimizer = "lamb"
			cfg.Training.LR = 0.007
			cfg.Training.GradClip = 1
			cfg.Training.WeightDecay = 0.1
			cfg.Training.LAMBBeta1 = 0.9
			cfg.Training.LAMBBeta2 = 0.98
			cfg.Training.LAMBEps = 1e-6
			cfg.Training.LAMBTrustRatioCap = 10
			prog, err := arch.BuildTrainingIRProgramFromConfig(cfg, arch.TrainingProgramState{Objective: arch.ObjectiveMNTP})
			if err != nil {
				t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
			}
			trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
			if err != nil {
				t.Fatalf("initGPUTrainer: %v", err)
			}
			defer trainerIface.CloseTrainer()
			trainer, ok := trainerIface.(*mlxGPUTrainer)
			if !ok {
				t.Fatalf("initGPUTrainer returned %T, want *mlxGPUTrainer", trainerIface)
			}
			if err := trainer.EnableComponentLossCapture(); err != nil {
				t.Fatalf("EnableComponentLossCapture: %v", err)
			}
			sampler := &pllMarginPairSampler{records: []pllMarginPairRecord{{
				ID: "p", Family: "agreement",
				ViewPos: []int{1, 2, 3, 4}, TargetPosPositions: []int{1, 2},
				ViewNeg: []int{1, 5, 2, 3}, TargetNegPositions: []int{2, 3}, TargetIDs: []int{2, 3},
				viewPosSet: true, targetPosPositionsSet: true, viewNegSet: true, targetNegPositionsSet: true, targetIDsSet: true,
			}}}
			raw := trainBatch{x: []int{1, 2, 3, 4, 4, 3, 2, 1, 2, 3, 4, 5, 5, 4, 3, 2}, y: make([]int, 16)}
			for i := range raw.y {
				raw.y[i] = raw.x[i]
			}
			batchSize := cfg.Training.BatchTokens / cfg.SeqLen
			sawPositivePLLLoss := false
			for step := 0; step < tc.steps; step++ {
				batch, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveMNTP)
				if err != nil {
					t.Fatalf("prepare step %d: %v", step, err)
				}
				batch, err = maybeAttachPLLMarginPairs(sampler, cfg, step, batch, batchSize, cfg.SeqLen, arch.ObjectiveMNTP)
				if err != nil {
					t.Fatalf("attach step %d: %v", step, err)
				}
				loss, err := trainer.TrainObjectiveStepGPU(batch, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
				if err != nil {
					t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
				}
				if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
					t.Fatalf("loss step %d=%g, want finite", step, loss)
				}
				components, err := trainer.ReadComponentLossesGPU()
				if err != nil {
					t.Fatalf("ReadComponentLossesGPU step %d: %v", step, err)
				}
				pllLoss, ok := components["pll_margin_loss"]
				if !ok || math.IsNaN(pllLoss) || math.IsInf(pllLoss, 0) || pllLoss < 0 {
					t.Fatalf("pll_margin_loss step %d=%v, want finite non-negative", step, components)
				}
				sawPositivePLLLoss = sawPositivePLLLoss || pllLoss > 0
			}
			if !sawPositivePLLLoss {
				t.Fatal("PLL margin loss never engaged before convergence")
			}
		})
	}
}

func TestPLLMarginTelemetryJSONLIncludesComponentLoss(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	dir := t.TempDir()
	pairPath := filepath.Join(dir, "pairs.bin")
	pairFile, err := os.Create(pairPath)
	if err != nil {
		t.Fatalf("create pairs: %v", err)
	}
	records := []pllMarginPairRecord{{
		ID: "p", Family: "agreement",
		ViewPos: []int{1, 2, 3, 4}, TargetPosPositions: []int{1, 2},
		ViewNeg: []int{1, 5, 2, 3}, TargetNegPositions: []int{2, 3}, TargetIDs: []int{2, 3},
		viewPosSet: true, targetPosPositionsSet: true, viewNegSet: true, targetNegPositionsSet: true, targetIDsSet: true,
	}}
	if err := writePLLMarginPairBinary(pairFile, records, 32, 4); err != nil {
		_ = pairFile.Close()
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

	cfg := parseTrainPLLMarginConfig(t, arch.ObjectiveMLM)
	cfg.Training.PLLMargin.Path = pairPath
	telemetryPath := filepath.Join(dir, "run.telemetry.jsonl")
	rt, err := newTelemetryRuntime("", telemetryPath)
	if err != nil {
		t.Fatalf("newTelemetryRuntime: %v", err)
	}
	_, err = runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{LogEvery: 1, ValEvery: 0, telemetry: rt})
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
	got, ok := snap.ComponentLosses["pll_margin_loss"]
	if !ok || !(got > 0) || math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("telemetry component_losses=%v, want finite pll_margin_loss", snap.ComponentLosses)
	}
}
