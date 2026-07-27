//go:build mlx && cgo && (darwin || linux)

package train

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	mixdist "github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

const distributedHardwareAcceptanceEnv = "MIXLAB_DDP_HARDWARE_ACCEPTANCE"

type distributedHardwareStepRecord struct {
	Event       string                        `json:"event"`
	Backend     string                        `json:"backend"`
	Rank        int                           `json:"rank"`
	Step        int                           `json:"step"`
	Loss        float32                       `json:"loss"`
	Distributed *distributedTrainingTelemetry `json:"distributed"`
}

// TestDDPHardwareAcceptanceWorker is an opt-in worker entrypoint for the R1
// multi-host Metal and same-host CUDA acceptance procedure. The launcher must
// create exactly two ranks and provide the selected backend.
func TestDDPHardwareAcceptanceWorker(t *testing.T) {
	if os.Getenv(distributedHardwareAcceptanceEnv) != "1" {
		t.Skip("set MIXLAB_DDP_HARDWARE_ACCEPTANCE=1 under mlx.launch")
	}
	if !mlxAvailable() || !gpu.Available() {
		t.Fatal("MLX device is unavailable")
	}
	rank, err := strconv.Atoi(os.Getenv("MLX_RANK"))
	if err != nil || rank < 0 || rank >= 2 {
		t.Fatalf("invalid MLX_RANK=%q", os.Getenv("MLX_RANK"))
	}
	backend := os.Getenv("MIXLAB_DDP_HW_BACKEND")
	if backend != "ring" && backend != "nccl" {
		t.Fatalf("MIXLAB_DDP_HW_BACKEND=%q, want ring or nccl", backend)
	}
	members := []mixdist.DDPGroupMember{
		{MemberID: "hardware-rank-0", Rank: 0},
		{MemberID: "hardware-rank-1", Rank: 1},
	}
	membership, err := mixdist.NewDDPGroupMembership(
		"r1-hardware-acceptance",
		"workers",
		1,
		backend,
		members,
	)
	if err != nil {
		t.Fatal(err)
	}
	view, err := mixdist.NewLocalGroupView(
		membership,
		members[rank].MemberID,
		rank,
		fmt.Sprintf("hardware-%s", backend),
	)
	if err != nil {
		t.Fatal(err)
	}
	startup, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	groupRuntime, err := gpu.NewGroupRuntime(startup, view)
	if err != nil {
		t.Fatalf("NewGroupRuntime: %v", err)
	}
	defer groupRuntime.Close()

	cfg, err := ParseArchConfig([]byte(`{
		"name": "ddp_hardware_acceptance",
		"model_dim": 64,
		"vocab_size": 64,
		"seq_len": 8,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"},
			{"type": "plain", "heads": 4},
			{"type": "swiglu"}
		],
		"training": {
			"objective": "causal",
			"optimizer": "adamw",
			"steps": 16,
			"lr": 0.001,
			"seed": 97,
			"batch_tokens": 128,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "ddp_hardware_acceptance")
	if err != nil {
		t.Fatal(err)
	}
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := initGPUTrainerWithDistributedContext(
		program,
		cfg,
		nil,
		nil,
		&DistributedTrainerContext{
			GroupRuntime:      groupRuntime,
			LocalView:         view,
			AccumulationSteps: 2,
			DatasetHash:       "r1-hardware-acceptance-fixture-v1",
		},
	)
	if err != nil {
		t.Fatalf("init distributed trainer: %v", err)
	}
	defer trainer.CloseTrainer()
	setter := trainer.(gpuTrainingStepSetter)
	if err := setter.SetTrainingStepGPU(0); err != nil {
		t.Fatal(err)
	}

	var firstLoss, lastLoss float32
	for step := 0; step < 16; step++ {
		x, y := distributedHardwareBatch(cfg, rank)
		prepared, err := prepareObjectiveBatch(
			cfg,
			trainBatch{x: x, y: y},
			step,
			"causal",
		)
		if err != nil {
			t.Fatal(err)
		}
		if err := submitPreparedStepGPU(
			trainer,
			prepared,
			cfg.Training.BatchTokens/cfg.SeqLen,
			cfg.SeqLen,
			float32(cfg.Training.LR),
		); err != nil {
			t.Fatalf("submit step %d: %v", step, err)
		}
		loss, err := trainer.CollectLossGPU()
		if err != nil {
			t.Fatalf("collect step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d loss is non-finite: %g", step, loss)
		}
		if step == 0 {
			firstLoss = loss
		}
		lastLoss = loss
		telemetry, err := trainer.(*mlxGPUTrainer).
			DistributedStepTelemetryGPU()
		if err != nil {
			t.Fatal(err)
		}
		record, err := json.Marshal(distributedHardwareStepRecord{
			Event:       "ddp_hardware_step",
			Backend:     backend,
			Rank:        rank,
			Step:        step + 1,
			Loss:        loss,
			Distributed: telemetry,
		})
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(string(record))
	}
	if !(lastLoss < firstLoss) {
		t.Fatalf(
			"loss did not decrease: first=%g last=%g",
			firstLoss,
			lastLoss,
		)
	}
}

func distributedHardwareBatch(cfg *ArchConfig, rank int) ([]int, []int) {
	x := make([]int, cfg.Training.BatchTokens)
	y := make([]int, cfg.Training.BatchTokens)
	for index := range x {
		x[index] = 1 + (rank*13+index)%60
		y[index] = 1 + (rank*13+index+1)%60
	}
	return x, y
}
