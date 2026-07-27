//go:build mlx && cgo && darwin

package train

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	mixdist "github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

const twoRankCaseEnv = "MIXLAB_DDP_TWO_RANK_CASE"

func TestTwoRankRingMatchesSingleProcessAdamW(t *testing.T) {
	runTwoRankCase(t, "adamw_parity", runTwoRankAdamWParityChild)
}

func TestSingleRankNaNGlobalSkip(t *testing.T) {
	runTwoRankCase(t, "global_nan_skip", runTwoRankGlobalSkipChild)
}

func TestInitializationAgreementRejectsMismatch(t *testing.T) {
	runTwoRankCase(t, "initialization_mismatch", runInitializationMismatchChild)
}

func TestDistributedRankZeroControlBroadcast(t *testing.T) {
	runTwoRankCase(t, "control_broadcast", runControlBroadcastChild)
}

func TestUnequalDenominatorWeightedReduction(t *testing.T) {
	runTwoRankCase(t, "unequal_denominator", runUnequalDenominatorChild)
}

func runTwoRankCase(t *testing.T, name string, child func(*testing.T, int)) {
	t.Helper()
	if os.Getenv(twoRankCaseEnv) == name {
		child(t, launchedRank(t))
		return
	}
	if !mlxAvailable() || !gpu.Available() || !gpu.DistributedBackendAvailable("ring") {
		t.Skip("MLX device or ring backend unavailable")
	}
	if _, err := exec.LookPath("mlx.launch"); err != nil {
		t.Skip("mlx.launch is unavailable")
	}
	port := reserveRingPortPair(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	command := exec.CommandContext(
		ctx,
		"mlx.launch",
		"--hosts", "127.0.0.1",
		"--repeat-hosts", "2",
		"--backend", "ring",
		"--starting-port", strconv.Itoa(port),
		"--env", twoRankCaseEnv+"="+name,
		"--",
		os.Args[0],
		"-test.run", "^"+t.Name()+"$",
		"-test.count=1",
	)
	output, err := command.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("two-rank case %s timed out: %v\n%s", name, ctx.Err(), output)
	}
	if err != nil {
		t.Fatalf("two-rank case %s failed: %v\n%s", name, err, output)
	}
}

func reserveRingPortPair(t *testing.T) int {
	t.Helper()
	for attempt := 0; attempt < 20; attempt++ {
		first, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("reserve ring port: %v", err)
		}
		port := first.Addr().(*net.TCPAddr).Port
		second, secondErr := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port+1))
		if secondErr == nil {
			_ = second.Close()
			_ = first.Close()
			return port
		}
		_ = first.Close()
	}
	t.Fatal("could not reserve adjacent ring ports")
	return 0
}

func launchedRank(t *testing.T) int {
	t.Helper()
	rank, err := strconv.Atoi(os.Getenv("MLX_RANK"))
	if err != nil || rank < 0 || rank > 1 {
		t.Fatalf("invalid MLX_RANK=%q", os.Getenv("MLX_RANK"))
	}
	return rank
}

func newTwoRankRuntime(t *testing.T, rank int) (*gpu.GroupRuntime, mixdist.LocalGroupView) {
	t.Helper()
	members := []mixdist.DDPGroupMember{
		{MemberID: "rank-0", Rank: 0},
		{MemberID: "rank-1", Rank: 1},
	}
	membership, err := mixdist.NewDDPGroupMembership(
		"phase2-test",
		"workers",
		7,
		"ring",
		members,
	)
	if err != nil {
		t.Fatalf("NewDDPGroupMembership: %v", err)
	}
	view, err := mixdist.NewLocalGroupView(
		membership,
		members[rank].MemberID,
		rank,
		"phase2-attempt",
	)
	if err != nil {
		t.Fatalf("NewLocalGroupView: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	runtime, err := gpu.NewGroupRuntime(ctx, view)
	if err != nil {
		t.Fatalf("NewGroupRuntime: %v", err)
	}
	t.Cleanup(runtime.Close)
	return runtime, view
}

func runTwoRankAdamWParityChild(t *testing.T, rank int) {
	runtime, view := newTwoRankRuntime(t, rank)
	localCfg := mustParseDistributedAccumulationConfig(t, 8)
	globalCfg := mustParseDistributedAccumulationConfig(t, 16)
	localProgram, err := BuildIRProgramFromConfig(localCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(local): %v", err)
	}
	globalProgram, err := BuildIRProgramFromConfig(globalCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(global): %v", err)
	}
	distributedTrainer, err := initGPUTrainerWithDistributedContext(
		localProgram,
		localCfg,
		nil,
		nil,
		&DistributedTrainerContext{GroupRuntime: runtime, LocalView: view},
	)
	if err != nil {
		t.Fatalf("init distributed trainer: %v", err)
	}
	defer distributedTrainer.CloseTrainer()
	reference, err := initGPUTrainer(globalProgram, globalCfg, nil, nil)
	if err != nil {
		t.Fatalf("init reference trainer: %v", err)
	}
	defer reference.CloseTrainer()

	localX, localY := rankBatch(rank)
	localPrepared, err := prepareObjectiveBatch(
		localCfg,
		trainBatch{x: localX, y: localY},
		0,
		"causal",
	)
	if err != nil {
		t.Fatalf("prepare local: %v", err)
	}
	if err := submitPreparedStepGPU(
		distributedTrainer,
		localPrepared,
		2,
		localCfg.SeqLen,
		float32(localCfg.Training.LR),
	); err != nil {
		t.Fatalf("submit distributed: %v", err)
	}
	if _, err := distributedTrainer.CollectLossGPU(); err != nil {
		t.Fatalf("collect distributed: %v", err)
	}

	x0, y0 := rankBatch(0)
	x1, y1 := rankBatch(1)
	globalPrepared, err := prepareObjectiveBatch(
		globalCfg,
		trainBatch{
			x: append(append([]int(nil), x0...), x1...),
			y: append(append([]int(nil), y0...), y1...),
		},
		0,
		"causal",
	)
	if err != nil {
		t.Fatalf("prepare global: %v", err)
	}
	if err := submitPreparedStepGPU(
		reference,
		globalPrepared,
		4,
		globalCfg.SeqLen,
		float32(globalCfg.Training.LR),
	); err != nil {
		t.Fatalf("submit reference: %v", err)
	}
	if _, err := reference.CollectLossGPU(); err != nil {
		t.Fatalf("collect reference: %v", err)
	}
	distributedWeights, err := readTrainerWeights(distributedTrainer)
	if err != nil {
		t.Fatalf("read distributed weights: %v", err)
	}
	referenceWeights, err := readTrainerWeights(reference)
	if err != nil {
		t.Fatalf("read reference weights: %v", err)
	}
	if diff := maxWeightDifference(distributedWeights, referenceWeights); diff > 1e-5 {
		t.Fatalf("two-rank AdamW max diff=%g, want <=1e-5", diff)
	}
}

func runTwoRankGlobalSkipChild(t *testing.T, rank int) {
	runtime, view := newTwoRankRuntime(t, rank)
	cfg := mustParseDistributedAccumulationConfig(t, 8)
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainerWithDistributedContext(
		program,
		cfg,
		nil,
		nil,
		&DistributedTrainerContext{GroupRuntime: runtime, LocalView: view},
	)
	if err != nil {
		t.Fatalf("init trainer: %v", err)
	}
	defer trainer.CloseTrainer()
	raw := trainer.(*mlxGPUTrainer)
	before, err := readTrainerWeights(trainer)
	if err != nil {
		t.Fatalf("read before weights: %v", err)
	}
	if rank == 1 {
		if err := gpu.TrainerSetDistributedTestPreUpdateBad(raw.handle, true); err != nil {
			t.Fatalf("inject pre-update bad: %v", err)
		}
	}
	x, y := rankBatch(rank)
	prepared, err := prepareObjectiveBatch(cfg, trainBatch{x: x, y: y}, 0, "causal")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if err := submitPreparedStepGPU(trainer, prepared, 2, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if loss, err := trainer.CollectLossGPU(); err != nil || loss != 0 {
		t.Fatalf("collect skipped loss=%g err=%v", loss, err)
	}
	after, err := readTrainerWeights(trainer)
	if err != nil {
		t.Fatalf("read after weights: %v", err)
	}
	if diff := maxWeightDifference(before, after); diff != 0 {
		t.Fatalf("global skip changed weights by %g", diff)
	}
	stats, err := readOptimizerStats(trainer)
	if err != nil {
		t.Fatalf("read optimizer stats: %v", err)
	}
	if !stats.LastStepSkipped || stats.SkippedSteps != 1 || stats.CommittedSteps != 0 {
		t.Fatalf("global skip stats=%+v", stats)
	}
	trace, err := raw.DistributedStageTraceGPU()
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	if strings.Contains(strings.Join(trace, ","), "all_sum_bucket_") {
		t.Fatalf("globally bad pre-update issued gradient collective: %v", trace)
	}
}

func runInitializationMismatchChild(t *testing.T, rank int) {
	runtime, _ := newTwoRankRuntime(t, rank)
	fields := []gpu.InitializationAgreementField{
		{Name: "config", Value: uint64(100 + rank)},
		{Name: "ir", Value: 200},
	}
	err := runtime.ValidateInitializationAgreement(fields)
	if err == nil || !strings.Contains(err.Error(), `rank 1 field "config"`) {
		t.Fatalf("initialization mismatch error=%v", err)
	}
}

func runControlBroadcastChild(t *testing.T, rank int) {
	runtime, _ := newTwoRankRuntime(t, rank)
	local := []int32{0, 0, 0}
	if rank == 0 {
		local = []int32{1, 17, -3}
	}
	got, err := runtime.BroadcastControl(0, local)
	if err != nil {
		t.Fatalf("BroadcastControl: %v", err)
	}
	want := []int32{1, 17, -3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("control[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func runUnequalDenominatorChild(t *testing.T, rank int) {
	runtime, view := newTwoRankRuntime(t, rank)
	localCfg := mustParseMaskedDDPConfig(t, 8)
	globalCfg := mustParseMaskedDDPConfig(t, 16)
	localProgram, err := BuildIRProgramFromConfig(localCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(local): %v", err)
	}
	globalProgram, err := BuildIRProgramFromConfig(globalCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(global): %v", err)
	}
	distributedTrainer, err := initGPUTrainerWithDistributedContext(
		localProgram,
		localCfg,
		nil,
		nil,
		&DistributedTrainerContext{GroupRuntime: runtime, LocalView: view},
	)
	if err != nil {
		t.Fatalf("init distributed trainer: %v", err)
	}
	defer distributedTrainer.CloseTrainer()
	reference, err := initGPUTrainer(globalProgram, globalCfg, nil, nil)
	if err != nil {
		t.Fatalf("init reference trainer: %v", err)
	}
	defer reference.CloseTrainer()

	local := maskedRankBatch(rank)
	if err := submitPreparedStepGPU(
		distributedTrainer,
		local,
		2,
		localCfg.SeqLen,
		float32(localCfg.Training.LR),
	); err != nil {
		t.Fatalf("submit distributed: %v", err)
	}
	if _, err := distributedTrainer.CollectLossGPU(); err != nil {
		t.Fatalf("collect distributed: %v", err)
	}
	rankZero := maskedRankBatch(0)
	rankOne := maskedRankBatch(1)
	global := objectiveBatch{
		x:                 append(append([]int(nil), rankZero.x...), rankOne.x...),
		y:                 append(append([]int(nil), rankZero.y...), rankOne.y...),
		lossMask:          append(append([]float32(nil), rankZero.lossMask...), rankOne.lossMask...),
		lossNormalizer:    rankZero.lossNormalizer + rankOne.lossNormalizer,
		lossNormalizerSet: true,
	}
	if err := submitPreparedStepGPU(
		reference,
		global,
		4,
		globalCfg.SeqLen,
		float32(globalCfg.Training.LR),
	); err != nil {
		t.Fatalf("submit reference: %v", err)
	}
	if _, err := reference.CollectLossGPU(); err != nil {
		t.Fatalf("collect reference: %v", err)
	}
	distributedWeights, err := readTrainerWeights(distributedTrainer)
	if err != nil {
		t.Fatalf("read distributed weights: %v", err)
	}
	referenceWeights, err := readTrainerWeights(reference)
	if err != nil {
		t.Fatalf("read reference weights: %v", err)
	}
	if diff := maxWeightDifference(distributedWeights, referenceWeights); diff > 1e-5 {
		t.Fatalf("unequal-denominator max diff=%g, want <=1e-5", diff)
	}
}

func maskedRankBatch(rank int) objectiveBatch {
	original, _ := rankBatch(rank)
	x := append([]int(nil), original...)
	maskCount := 8
	if rank == 1 {
		maskCount = 4
	}
	mask := make([]float32, len(x))
	for i := 0; i < maskCount; i++ {
		x[i] = 31
		mask[i] = 1
	}
	return objectiveBatch{
		x:                 x,
		y:                 original,
		lossMask:          mask,
		lossNormalizer:    float32(maskCount),
		lossNormalizerSet: true,
	}
}

func mustParseMaskedDDPConfig(t *testing.T, batchTokens int) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name": "ddp_masked_denominator",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
			{"type": "swiglu"}
		],
		"training": {
			"objective": "mlm",
			"mlm_mask_token_id": 31,
			"optimizer": "adamw",
			"steps": 1,
			"lr": 0.0005,
			"seed": 29,
			"batch_tokens": `+strconv.Itoa(batchTokens)+`,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "ddp_masked_denominator")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func rankBatch(rank int) ([]int, []int) {
	x := make([]int, 8)
	y := make([]int, 8)
	for i := range x {
		x[i] = 1 + (rank*8+i)%28
		y[i] = 1 + (rank*8+i+1)%28
	}
	return x, y
}
