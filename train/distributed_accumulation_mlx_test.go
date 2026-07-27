//go:build mlx && cgo && (darwin || linux)

package train

import (
	"context"
	"strconv"
	"testing"
	"time"

	mixdist "github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

func TestGradientAccumulationEquivalence(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() || !gpu.DistributedBackendAvailable("ring") {
		t.Skip("MLX device or ring backend unavailable")
	}
	distributedCfg := mustParseDistributedAccumulationConfig(t, 8)
	referenceCfg := mustParseDistributedAccumulationConfig(t, 32)
	distributedProgram, err := BuildIRProgramFromConfig(distributedCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(distributed): %v", err)
	}
	referenceProgram, err := BuildIRProgramFromConfig(referenceCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(reference): %v", err)
	}

	membership, err := mixdist.NewDDPGroupMembership(
		"accumulation-run",
		"workers",
		0,
		"ring",
		[]mixdist.DDPGroupMember{{MemberID: "local", Rank: 0}},
	)
	if err != nil {
		t.Fatalf("NewDDPGroupMembership: %v", err)
	}
	view, err := mixdist.NewLocalGroupView(membership, "local", 0, "attempt-1")
	if err != nil {
		t.Fatalf("NewLocalGroupView: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	groupRuntime, err := gpu.NewSingletonGroupRuntime(ctx, view)
	if err != nil {
		t.Fatalf("NewSingletonGroupRuntime: %v", err)
	}
	defer groupRuntime.Close()

	reference, err := initGPUTrainer(referenceProgram, referenceCfg, nil, nil)
	if err != nil {
		t.Fatalf("init reference trainer: %v", err)
	}
	defer reference.CloseTrainer()
	accumulated, err := initGPUTrainerWithDistributedContext(
		distributedProgram,
		distributedCfg,
		nil,
		nil,
		&DistributedTrainerContext{
			GroupRuntime:      groupRuntime,
			LocalView:         view,
			AccumulationSteps: 4,
		},
	)
	if err != nil {
		t.Fatalf("init accumulated trainer: %v", err)
	}
	defer accumulated.CloseTrainer()

	fullX := make([]int, 0, 32)
	fullY := make([]int, 0, 32)
	for microstep := 0; microstep < 4; microstep++ {
		x := make([]int, 8)
		y := make([]int, 8)
		for i := range x {
			x[i] = 1 + (microstep*8+i)%28
			y[i] = 1 + (microstep*8+i+1)%28
		}
		fullX = append(fullX, x...)
		fullY = append(fullY, y...)
		prepared, prepErr := prepareObjectiveBatch(
			distributedCfg,
			trainBatch{x: x, y: y},
			microstep,
			"causal",
		)
		if prepErr != nil {
			t.Fatalf("prepare microstep %d: %v", microstep, prepErr)
		}
		if submitErr := submitPreparedStepGPU(
			accumulated,
			prepared,
			2,
			distributedCfg.SeqLen,
			float32(distributedCfg.Training.LR),
		); submitErr != nil {
			t.Fatalf("submit microstep %d: %v", microstep, submitErr)
		}
		if _, collectErr := accumulated.CollectLossGPU(); collectErr != nil {
			t.Fatalf("collect microstep %d: %v", microstep, collectErr)
		}
	}

	fullPrepared, err := prepareObjectiveBatch(
		referenceCfg,
		trainBatch{x: fullX, y: fullY},
		0,
		"causal",
	)
	if err != nil {
		t.Fatalf("prepare reference: %v", err)
	}
	if err := submitPreparedStepGPU(
		reference,
		fullPrepared,
		8,
		referenceCfg.SeqLen,
		float32(referenceCfg.Training.LR),
	); err != nil {
		t.Fatalf("submit reference: %v", err)
	}
	if _, err := reference.CollectLossGPU(); err != nil {
		t.Fatalf("collect reference: %v", err)
	}

	referenceWeights, err := readTrainerWeights(reference)
	if err != nil {
		t.Fatalf("read reference weights: %v", err)
	}
	accumulatedWeights, err := readTrainerWeights(accumulated)
	if err != nil {
		t.Fatalf("read accumulated weights: %v", err)
	}
	if diff := maxWeightDifference(referenceWeights, accumulatedWeights); diff > 1e-5 {
		t.Fatalf("K=4 accumulated parameter max diff=%g, want <=1e-5", diff)
	}
	stats, err := readOptimizerStats(accumulated)
	if err != nil {
		t.Fatalf("read accumulated optimizer stats: %v", err)
	}
	if stats.AttemptedSteps != 1 || stats.CommittedSteps != 1 {
		t.Fatalf("accumulated optimizer stats=%+v, want one attempt and commit", stats)
	}
}

func mustParseDistributedAccumulationConfig(t *testing.T, batchTokens int) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name": "ddp_accumulation",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"objective": "causal",
			"optimizer": "adamw",
			"steps": 1,
			"lr": 0.0005,
			"seed": 29,
			"batch_tokens": `+strconv.Itoa(batchTokens)+`,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "ddp_accumulation")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}
