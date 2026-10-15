package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/mrothroc/mixlab/internal/strictjson"
)

// DistributedSpec contains training semantics, never host or launcher identity.
type DistributedSpec struct {
	Mode                      string   `json:"mode"`
	Backend                   string   `json:"backend"`
	GradientAccumulationSteps int      `json:"gradient_accumulation_steps"`
	GradientBucketBytes       int      `json:"gradient_bucket_bytes"`
	InnerSteps                int      `json:"inner_steps,omitempty"`
	OuterOptimizer            string   `json:"outer_optimizer,omitempty"`
	OuterLR                   float64  `json:"outer_lr,omitempty"`
	OuterMomentum             *float64 `json:"outer_momentum,omitempty"`
	Aggregation               string   `json:"aggregation,omitempty"`
}

func (s *DistributedSpec) UnmarshalJSON(blob []byte) error {
	if len(blob) > 4096 {
		return fmt.Errorf("training.distributed: config exceeds 4096 bytes")
	}
	if err := strictjson.Validate(blob, 4); err != nil {
		return err
	}
	type plain DistributedSpec
	value := plain{Backend: "auto", GradientAccumulationSteps: 1, GradientBucketBytes: 32 << 20}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(blob, &fields); err != nil {
		return err
	}
	for key, raw := range fields {
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return fmt.Errorf("training.distributed.%s cannot be null", key)
		}
	}
	var mode string
	if raw, ok := fields["mode"]; ok {
		if err := json.Unmarshal(raw, &mode); err != nil {
			return err
		}
	}
	if mode == "diloco" {
		value.Backend = ""
		value.GradientBucketBytes = 0
		if _, ok := fields["backend"]; ok {
			return fmt.Errorf("training.distributed.backend is DDP-only")
		}
		if _, ok := fields["gradient_bucket_bytes"]; ok {
			return fmt.Errorf("training.distributed.gradient_bucket_bytes is DDP-only")
		}
	} else {
		for _, key := range []string{"inner_steps", "outer_optimizer", "outer_lr", "outer_momentum", "aggregation"} {
			if _, ok := fields[key]; ok {
				return fmt.Errorf("training.distributed.%s requires mode=diloco", key)
			}
		}
	}
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

// MarshalJSON preserves the shipped DDP encoding and omits collective settings
// from single-process learner configs. A pointer preserves explicit zero momentum.
func (s DistributedSpec) MarshalJSON() ([]byte, error) {
	type plain DistributedSpec
	if s.Mode != "diloco" {
		return json.Marshal(plain(s))
	}
	if s.Backend != "" || s.GradientBucketBytes != 0 {
		return nil, fmt.Errorf("diloco cannot marshal DDP settings")
	}
	return json.Marshal(struct {
		Mode           string   `json:"mode"`
		Accumulation   int      `json:"gradient_accumulation_steps"`
		InnerSteps     int      `json:"inner_steps"`
		OuterOptimizer string   `json:"outer_optimizer"`
		OuterLR        float64  `json:"outer_lr"`
		OuterMomentum  *float64 `json:"outer_momentum"`
		Aggregation    string   `json:"aggregation"`
	}{s.Mode, s.GradientAccumulationSteps, s.InnerSteps, s.OuterOptimizer, s.OuterLR, s.OuterMomentum, s.Aggregation})
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
	check(s.Mode != "ddp" && s.Mode != "diloco", "mode must be ddp or diloco")
	check(s.GradientAccumulationSteps <= 0, "gradient_accumulation_steps must be positive")
	if s.Mode == "diloco" {
		check(s.Backend != "" || s.GradientBucketBytes != 0, "backend and gradient_bucket_bytes are DDP-only")
		check(s.InnerSteps <= 0, "inner_steps must be positive")
		check(t.Steps <= 0 || (s.InnerSteps > 0 && t.Steps%s.InnerSteps != 0), "steps must be positive and divisible by inner_steps")
		check(s.OuterOptimizer != "nesterov", "outer_optimizer must be nesterov")
		check(s.OuterLR <= 0 || math.IsNaN(s.OuterLR) || math.IsInf(s.OuterLR, 0), "outer_lr must be finite and positive")
		check(s.OuterMomentum == nil, "outer_momentum is required (explicit zero is valid)")
		if s.OuterMomentum != nil {
			v := *s.OuterMomentum
			check(v < 0 || v >= 1 || math.IsNaN(v) || math.IsInf(v, 0), "outer_momentum must be finite in [0,1)")
		}
		check(s.Aggregation != "uniform" && s.Aggregation != "effective_tokens", "aggregation must be uniform or effective_tokens")
	} else {
		check(s.Backend != "" && s.Backend != "auto" && s.Backend != "ring" && s.Backend != "nccl", "backend must be auto, ring, or nccl")
		check(s.GradientBucketBytes <= 0, "gradient_bucket_bytes must be positive")
		check(s.InnerSteps != 0 || s.OuterOptimizer != "" || s.OuterLR != 0 || s.OuterMomentum != nil || s.Aggregation != "", "DiLoCo fields require mode=diloco")
	}
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
