//go:build mlx && cgo && (darwin || linux)

package train

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	mixdist "github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

func TestDDPWalkingSkeletonSingleRank(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() || !gpu.DistributedBackendAvailable("ring") {
		t.Skip("MLX device or ring backend unavailable")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name": "ddp_walking_skeleton",
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
			"seed": 17,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "ddp_walking_skeleton")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	membership, err := mixdist.NewDDPGroupMembership(
		"walking-run",
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

	reference, err := initGPUTrainer(program, cfg, nil, nil)
	if err != nil {
		t.Fatalf("init reference trainer: %v", err)
	}
	defer reference.CloseTrainer()
	distributedTrainer, err := initGPUTrainerWithDistributedContext(
		program,
		cfg,
		nil,
		nil,
		&DistributedTrainerContext{
			GroupRuntime: groupRuntime,
			LocalView:    view,
		},
	)
	if err != nil {
		t.Fatalf("init distributed trainer: %v", err)
	}
	defer distributedTrainer.CloseTrainer()
	bucketReader, ok := distributedTrainer.(interface {
		DistributedBucketMetadataGPU() (gpu.DistributedBucketMetadata, error)
	})
	if !ok {
		t.Fatalf("distributed trainer type %T has no bucket metadata", distributedTrainer)
	}
	bucketMetadata, err := bucketReader.DistributedBucketMetadataGPU()
	if err != nil {
		t.Fatalf("DistributedBucketMetadataGPU: %v", err)
	}
	if bucketMetadata.TargetBytes != gpu.DefaultGradientBucketBytes {
		t.Fatalf("bucket target=%d want %d", bucketMetadata.TargetBytes, gpu.DefaultGradientBucketBytes)
	}
	if bucketMetadata.TotalBytes == 0 || bucketMetadata.Digest == 0 {
		t.Fatalf("invalid bucket metadata: %+v", bucketMetadata)
	}
	if bucketMetadata.BucketCount != 1 {
		t.Fatalf("tiny model bucket count=%d want 1", bucketMetadata.BucketCount)
	}

	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 5, 6, 7, 8, 9},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, "causal")
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	if err := submitPreparedStepGPU(
		reference,
		prepared,
		batchSize,
		cfg.SeqLen,
		float32(cfg.Training.LR),
	); err != nil {
		t.Fatalf("submit reference: %v", err)
	}
	if err := submitPreparedStepGPU(
		distributedTrainer,
		prepared,
		batchSize,
		cfg.SeqLen,
		float32(cfg.Training.LR),
	); err != nil {
		t.Fatalf("submit distributed: %v", err)
	}
	referenceLoss, err := reference.CollectLossGPU()
	if err != nil {
		t.Fatalf("collect reference: %v", err)
	}
	distributedLoss, err := distributedTrainer.CollectLossGPU()
	if err != nil {
		t.Fatalf("collect distributed: %v", err)
	}
	if math.Abs(float64(referenceLoss-distributedLoss)) > 1e-5 {
		t.Fatalf("loss mismatch reference=%g distributed=%g", referenceLoss, distributedLoss)
	}

	referenceWeights, err := readTrainerWeights(reference)
	if err != nil {
		t.Fatalf("read reference weights: %v", err)
	}
	distributedWeights, err := readTrainerWeights(distributedTrainer)
	if err != nil {
		t.Fatalf("read distributed weights: %v", err)
	}
	if diff := maxWeightDifference(referenceWeights, distributedWeights); diff > 1e-5 {
		t.Fatalf("staged parameter max diff=%g, want <=1e-5", diff)
	}
	traceReader, ok := distributedTrainer.(interface {
		DistributedStageTraceGPU() ([]string, error)
	})
	if !ok {
		t.Fatalf("distributed trainer type %T has no stage trace", distributedTrainer)
	}
	trace, err := traceReader.DistributedStageTraceGPU()
	if err != nil {
		t.Fatalf("DistributedStageTraceGPU: %v", err)
	}
	wantTrace := []string{
		"numerator_conversion",
		"pre_update_finite",
		"all_max_pre_update_bad",
		"all_sum_denominator",
		"all_sum_bucket_0",
		"clip",
		"candidate",
		"candidate_finite",
		"all_max_candidate_bad",
		"commit",
	}
	if !reflect.DeepEqual(trace, wantTrace) {
		t.Fatalf("stage trace=%v want %v", trace, wantTrace)
	}

	zeroDenominator := prepared
	zeroDenominator.lossNormalizer = 0
	zeroDenominator.lossNormalizerSet = true
	if err := submitPreparedStepGPU(
		distributedTrainer,
		zeroDenominator,
		batchSize,
		cfg.SeqLen,
		float32(cfg.Training.LR),
	); err != nil {
		t.Fatalf("submit zero denominator: %v", err)
	}
	skippedLoss, err := distributedTrainer.CollectLossGPU()
	if err != nil {
		t.Fatalf("collect zero denominator: %v", err)
	}
	if skippedLoss != 0 {
		t.Fatalf("zero-denominator reported loss=%g want 0", skippedLoss)
	}
	afterSkip, err := readTrainerWeights(distributedTrainer)
	if err != nil {
		t.Fatalf("read weights after zero denominator: %v", err)
	}
	if diff := maxWeightDifference(distributedWeights, afterSkip); diff != 0 {
		t.Fatalf("zero-denominator step changed weights: max diff=%g", diff)
	}
	stats, err := readOptimizerStats(distributedTrainer)
	if err != nil {
		t.Fatalf("read zero-denominator optimizer stats: %v", err)
	}
	if !stats.LastStepSkipped || stats.SkippedSteps != 1 || stats.CommittedSteps != 1 {
		t.Fatalf("zero-denominator optimizer stats=%+v", stats)
	}
	trace, err = traceReader.DistributedStageTraceGPU()
	if err != nil {
		t.Fatalf("zero-denominator stage trace: %v", err)
	}
	if want := []string{
		"numerator_conversion",
		"pre_update_finite",
		"all_max_pre_update_bad",
		"all_sum_denominator",
		"skip_zero_denominator",
	}; !reflect.DeepEqual(trace, want) {
		t.Fatalf("zero-denominator stage trace=%v want %v", trace, want)
	}
	rawTrainer, ok := distributedTrainer.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("distributed trainer type %T is not mlxGPUTrainer", distributedTrainer)
	}
	rebuilds, err := gpu.TrainerArgumentLayoutRebuilds(rawTrainer.handle)
	if err != nil {
		t.Fatalf("TrainerArgumentLayoutRebuilds: %v", err)
	}
	if rebuilds != 1 {
		t.Fatalf("stable named-step signature rebuilt argument layout %d times, want 1", rebuilds)
	}
}

func TestGradientBucketingCanonicalOrder(t *testing.T) {
	TestDDPWalkingSkeletonSingleRank(t)
}

func TestIRTrainerArgumentLayoutCache(t *testing.T) {
	TestDDPWalkingSkeletonSingleRank(t)
}

func TestCollectiveTransactionOrder(t *testing.T) {
	TestDDPWalkingSkeletonSingleRank(t)
}

func TestDDPMaskedObjectivesSingleRank(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() || !gpu.DistributedBackendAvailable("ring") {
		t.Skip("MLX device or ring backend unavailable")
	}
	for _, objective := range []string{"mlm", "mntp"} {
		t.Run(objective, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
				"name": "ddp_%s",
				"model_dim": 16,
				"vocab_size": 32,
				"seq_len": 4,
				"blocks": [
					{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
					{"type": "swiglu"}
				],
				"training": {
					"objective": %q,
					"optimizer": "adamw",
					"steps": 1,
					"lr": 0.0005,
					"seed": 17,
					"batch_tokens": 8,
					"grad_clip": 1.0,
					"weight_decay": 0.0,
					"mlm_mask_prob": 0.5,
					"mlm_mask_token_id": 1
				}
			}`, objective, objective)), "ddp_"+objective)
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			program, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}
			membership, err := mixdist.NewDDPGroupMembership(
				"masked-"+objective,
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

			reference, err := initGPUTrainer(program, cfg, nil, nil)
			if err != nil {
				t.Fatalf("init reference trainer: %v", err)
			}
			defer reference.CloseTrainer()
			distributedTrainer, err := initGPUTrainerWithDistributedContext(
				program,
				cfg,
				nil,
				nil,
				&DistributedTrainerContext{GroupRuntime: groupRuntime, LocalView: view},
			)
			if err != nil {
				t.Fatalf("init distributed trainer: %v", err)
			}
			defer distributedTrainer.CloseTrainer()

			raw := trainBatch{
				x: []int{2, 3, 4, 5, 6, 7, 8, 9},
				y: []int{3, 4, 5, 6, 7, 8, 9, 10},
			}
			prepared, err := prepareObjectiveBatch(cfg, raw, 0, objective)
			if err != nil {
				t.Fatalf("prepareObjectiveBatch: %v", err)
			}
			if !prepared.lossNormalizerSet || prepared.lossNormalizer <= 0 {
				t.Fatalf(
					"masked objective normalizer set=%v value=%g",
					prepared.lossNormalizerSet,
					prepared.lossNormalizer,
				)
			}
			batchSize := cfg.Training.BatchTokens / cfg.SeqLen
			for name, trainer := range map[string]GPUTrainer{
				"reference":   reference,
				"distributed": distributedTrainer,
			} {
				if err := submitPreparedStepGPU(
					trainer,
					prepared,
					batchSize,
					cfg.SeqLen,
					float32(cfg.Training.LR),
				); err != nil {
					t.Fatalf("submit %s: %v", name, err)
				}
			}
			referenceLoss, err := reference.CollectLossGPU()
			if err != nil {
				t.Fatalf("collect reference: %v", err)
			}
			distributedLoss, err := distributedTrainer.CollectLossGPU()
			if err != nil {
				t.Fatalf("collect distributed: %v", err)
			}
			if !finiteFloat32(referenceLoss) || !finiteFloat32(distributedLoss) {
				t.Fatalf("non-finite losses reference=%g distributed=%g", referenceLoss, distributedLoss)
			}
			if diff := math.Abs(float64(referenceLoss - distributedLoss)); diff > 1e-5 {
				t.Fatalf("loss mismatch reference=%g distributed=%g", referenceLoss, distributedLoss)
			}
			referenceWeights, err := readTrainerWeights(reference)
			if err != nil {
				t.Fatalf("read reference weights: %v", err)
			}
			distributedWeights, err := readTrainerWeights(distributedTrainer)
			if err != nil {
				t.Fatalf("read distributed weights: %v", err)
			}
			if diff := maxWeightDifference(referenceWeights, distributedWeights); diff > 1e-5 {
				t.Fatalf("parameter max diff=%g, want <=1e-5", diff)
			}
		})
	}
}

func finiteFloat32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func maxWeightDifference(a, b [][]float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var maxDiff float64
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return math.Inf(1)
		}
		for j := range a[i] {
			diff := math.Abs(float64(a[i][j] - b[i][j]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	return maxDiff
}
