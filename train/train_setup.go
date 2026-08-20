package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

func validateRunTrainOptions(opts TrainOptions) error {
	if opts.CheckpointEvery < 0 {
		return fmt.Errorf("checkpoint interval must be >= 0")
	}
	if opts.CheckpointEvery > 0 && opts.CheckpointDir == "" {
		return fmt.Errorf("-checkpoint-dir is required when -checkpoint-every > 0")
	}
	if opts.Resume != "" && opts.SafetensorsLoad != "" {
		return fmt.Errorf("-resume and -safetensors-load are mutually exclusive; use -resume for full training state or -safetensors-load for a weights-only warm start")
	}
	return nil
}

func configureTrainingTokenFeatures(cfg *ArchConfig, trainPattern, name string) error {
	if cfg.Training.FirstByteMask {
		source, err := configureFirstByteMaskForTraining(cfg, trainPattern)
		if err != nil {
			return err
		}
		fmt.Printf("  [%s] first-byte mask enabled (%s)\n", name, source)
	}
	if cfg.Training.UsesWholeWordMasking() {
		source, err := configureMLMWordBoundariesForTraining(cfg, trainPattern)
		if err != nil {
			return err
		}
		fmt.Printf("  [%s] MLM whole-word boundaries enabled (%s)\n", name, source)
	}
	if cfg.CharVocabSize > 0 {
		source, err := configureCharFeaturesForTraining(cfg, trainPattern)
		if err != nil {
			return err
		}
		fmt.Printf("  [%s] char features enabled (%s)\n", name, source)
	}
	return nil
}

func logTrainingRunSetup(
	cfg *ArchConfig,
	name string,
	batchSize, batchTokens int,
	swaOverrideLogs []string,
	swaStart, swaInterval int,
	swaDecay float32,
	earlyStop *earlyStopState,
	flops *arch.FLOPsEstimate,
) {
	if cfg.Training.LengthBucketsChangeShape(cfg.SeqLen) {
		shapes := make([]string, 0, len(cfg.Training.LengthBuckets))
		for _, bucket := range cfg.Training.LengthBuckets {
			rows, effectiveTokens := cfg.Training.LengthBucketBatchShape(bucket)
			shapes = append(shapes, fmt.Sprintf("%d:%dx%d", bucket, rows, effectiveTokens))
		}
		if fixed := cfg.Training.FixedLengthBucketBatchSize(); fixed > 0 {
			fmt.Printf("  [%s] length buckets: widths=%v fixed_batch_size=%d shapes=%s\n", name, cfg.Training.LengthBuckets, fixed, strings.Join(shapes, ","))
		} else {
			fmt.Printf("  [%s] length buckets: widths=%v batch_token_ceiling=%d shapes=%s\n", name, cfg.Training.LengthBuckets, batchTokens, strings.Join(shapes, ","))
		}
	}
	if cfg.RCEquivarianceEnabled() {
		fmt.Printf("  [%s] DNA reverse-complement equivariance: shared weights, branch_dim=%d, paired_backbone_rows=%d\n",
			name, cfg.ModelDim, 2*batchSize)
	}
	for _, msg := range swaOverrideLogs {
		fmt.Printf("  [%s] %s\n", name, msg)
	}
	if swaStart > 0 {
		fmt.Printf("  [%s] SWA/EMA enabled: start=%d interval=%d decay=%g\n", name, swaStart, swaInterval, swaDecay)
	}
	if earlyStop != nil {
		fmt.Printf("  [%s] early stop enabled: patience=%d min_delta=%g min_steps=%d val_gt=%g at_step=%d\n",
			name, cfg.Training.EarlyStop.Patience, cfg.Training.EarlyStop.MinDelta,
			cfg.Training.EarlyStop.MinSteps, cfg.Training.EarlyStop.ValGT, cfg.Training.EarlyStop.AtStep)
	}
	if cfg.Training.LengthBucketsChangeShape(cfg.SeqLen) {
		flops.TrainingFLOPsReliable = false
		fmt.Printf("  [%s] training FLOPs/MFU unavailable: batch shape varies by length bucket\n", name)
	} else if !flops.TrainingFLOPsReliable {
		fmt.Printf("  [%s] training FLOPs/MFU unavailable: TTT full-meta-gradient backward is not modeled\n", name)
	}
}
