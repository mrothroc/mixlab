//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	mixdist "github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

const phase1DistributedOptimizer = "adamw"

// DistributedTrainerContext is the internal Phase 1 assembly boundary. Public
// launcher and config integration is intentionally deferred.
type DistributedTrainerContext struct {
	GroupRuntime        *gpu.GroupRuntime
	LocalView           mixdist.LocalGroupView
	GradientBucketBytes uint64
	AccumulationSteps   int
	DatasetHash         string
	ScheduledPhase      string
}

func (c *DistributedTrainerContext) validate(cfg *ArchConfig) error {
	if c == nil || c.GroupRuntime == nil {
		return fmt.Errorf("distributed trainer context requires a group runtime")
	}
	view, err := mixdist.NewLocalGroupView(
		c.LocalView.Membership,
		c.LocalView.LocalMemberID,
		c.LocalView.LocalRank,
		c.LocalView.LaunchAttemptID,
	)
	if err != nil {
		return fmt.Errorf("validate distributed trainer identity: %w", err)
	}
	runtimeView := c.GroupRuntime.LocalView()
	if view.Membership.MembersHash != runtimeView.Membership.MembersHash ||
		view.Membership.Generation != runtimeView.Membership.Generation ||
		view.LocalMemberID != runtimeView.LocalMemberID ||
		view.LocalRank != runtimeView.LocalRank {
		return fmt.Errorf("distributed trainer identity does not match group runtime")
	}
	if cfg == nil {
		return fmt.Errorf("distributed trainer requires a config")
	}
	if c.AccumulationSteps < 0 {
		return fmt.Errorf("distributed accumulation steps must be positive")
	}
	objective := cfg.Training.EffectiveObjective()
	if objective != arch.ObjectiveCausal &&
		objective != arch.ObjectiveMLM &&
		objective != arch.ObjectiveMNTP {
		return fmt.Errorf(
			"R1 distributed training supports causal, mlm, and mntp objectives, got %q",
			objective,
		)
	}
	if strings.ToLower(strings.TrimSpace(cfg.Training.Optimizer)) != phase1DistributedOptimizer {
		return fmt.Errorf(
			"R1 Phase 1 distributed training supports optimizer=%q only",
			phase1DistributedOptimizer,
		)
	}
	if len(cfg.Training.SeqLenSchedule) > 0 {
		return fmt.Errorf("R1 Phase 1 distributed training does not support seq_len_schedule")
	}
	if cfg.Training.Distillation != nil ||
		cfg.Training.Data2Vec != nil ||
		cfg.MTP != nil ||
		cfg.Training.FirstByteMask ||
		cfg.Training.ExampleFramingEnabled() ||
		cfg.Training.AttentionSegmentMaskEnabled() {
		return fmt.Errorf("R1 distributed training does not support auxiliary loss features")
	}
	return nil
}

func initGPUTrainerWithDistributedContext(
	prog *arch.Program,
	cfg *ArchConfig,
	loadedWeights [][]float32,
	optimizerOverride func(gpu.TrainerOptimizerSpec, []WeightShape) (gpu.TrainerOptimizerSpec, error),
	distributedContext *DistributedTrainerContext,
) (GPUTrainer, error) {
	if !mlxAvailable() {
		return nil, fmt.Errorf("GPU training requires MLX backend; rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab")
	}
	if err := distributedContext.validate(cfg); err != nil {
		return nil, err
	}
	return initMLXGPUTrainerWithDistributedContext(
		prog,
		cfg,
		loadedWeights,
		optimizerOverride,
		distributedContext,
	)
}
