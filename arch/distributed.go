package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// DistributedSpec contains training semantics, never host or launcher identity.
type DistributedSpec struct {
	Mode                      string `json:"mode"`
	Backend                   string `json:"backend"`
	GradientAccumulationSteps int    `json:"gradient_accumulation_steps"`
	GradientBucketBytes       int    `json:"gradient_bucket_bytes"`
}

func (s *DistributedSpec) UnmarshalJSON(blob []byte) error {
	type plain DistributedSpec
	value := plain{Backend: "auto", GradientAccumulationSteps: 1, GradientBucketBytes: 32 << 20}
	d := json.NewDecoder(bytes.NewReader(blob))
	d.DisallowUnknownFields()
	if err := d.Decode(&value); err != nil {
		return fmt.Errorf("training.distributed: %w", err)
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("training.distributed: trailing JSON")
	}
	*s = DistributedSpec(value)
	return nil
}

// ValidateDistributedConfig is also used by programmatic training callers.
// The public R1 surface is intentionally narrower than internal parity fixtures.
func ValidateDistributedConfig(cfg *ArchConfig) error {
	t := cfg.Training
	s := t.Distributed
	if s == nil {
		return nil
	}
	var issues []string
	check := func(bad bool, field string) {
		if bad {
			issues = append(issues, field)
		}
	}
	check(s.Mode != "ddp", "mode must be ddp")
	check(s.Backend != "" && s.Backend != "auto" && s.Backend != "ring" && s.Backend != "nccl", "backend must be auto, ring, or nccl")
	check(s.GradientAccumulationSteps <= 0, "gradient_accumulation_steps must be positive")
	check(s.GradientBucketBytes <= 0, "gradient_bucket_bytes must be positive")
	check(t.EffectiveObjective() != ObjectiveCausal, "objective must be causal")
	check(t.Optimizer != "adamw", "optimizer must be adamw")
	check(t.BatchTokens <= 0 || cfg.SeqLen <= 0 || t.BatchTokens%max(cfg.SeqLen, 1) != 0, "batch_tokens must be divisible by seq_len")
	check(len(t.SeqLenSchedule) > 0 || len(t.LengthBuckets) > 0, "sequence length schedules/bucketing unsupported")
	check(t.Distillation != nil || t.Data2Vec != nil || t.RTD != nil || t.MinimalPair != nil || t.Invariance != nil || t.PLLMargin != nil || t.WordStructuralObjective != nil || cfg.MTP != nil || t.ZLoss != 0, "auxiliary losses unsupported")
	check(t.FirstByteMask || t.ExampleFramingEnabled() || t.AttentionSegmentMaskEnabled(), "masked/framed/segment training unsupported")
	check(t.QAT != "" && t.QAT != "none", "qat unsupported")
	check(t.SWAStart > 0, "swa unsupported")
	check(len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 || len(cfg.ExecutionOrder) > 0 || len(cfg.RecurrencePhaseActivations) > 0, "recurrence/custom execution order unsupported")
	check(cfg.NormType == NormTypeBatchNorm, "batchnorm mutable buffers unsupported")
	check(cfg.EffectiveInputAdapterKind() != InputAdapterTokenEmbedding, "only token input is supported")
	check(t.ReverseComplementProb != 0, "reverse complement augmentation unsupported")
	check(t.CautiousWeightDecay, "cautious_weight_decay unsupported")
	check(t.NewBob != nil, "newbob unsupported")
	check(t.TTTSteps > 0, "TTT validation updates unsupported")
	check(cfg.RCEquivarianceEnabled(), "rc_equivariant unsupported")
	check(cfg.Data.NoShardShuffle, "data.no_shard_shuffle unsupported by counter-based DDP sampling")
	check(t.ShuffleChunkTokens > 0 && (t.ShuffleChunkTokens < cfg.SeqLen || t.ShuffleChunkTokens%max(cfg.SeqLen, 1) != 0), "shuffle_chunk_tokens must be a multiple of seq_len")
	for i, b := range cfg.Blocks {
		check(b.Type == "mamba3-canonical" || b.Type == "moe" || b.Type == "custom", fmt.Sprintf("blocks[%d] %s is outside the R1 support matrix", i, b.Type))
	}
	if len(issues) > 0 {
		return fmt.Errorf("training.distributed: %s", strings.Join(issues, "; "))
	}
	return nil
}
