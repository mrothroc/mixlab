package arch

import "fmt"

// LengthBucketingEnabled reports whether classification batches use
// variable-width programs selected from training.length_buckets.
func (t TrainingSpec) LengthBucketingEnabled() bool {
	return len(t.LengthBuckets) > 0
}

// LengthBucketsChangeShape reports whether the configured buckets require
// dynamic batch shapes. A single full-width bucket with an exactly divisible
// token budget takes the legacy loader and loss path for strict parity.
func (t TrainingSpec) LengthBucketsChangeShape(seqLen int) bool {
	if !t.LengthBucketingEnabled() {
		return false
	}
	return len(t.LengthBuckets) != 1 || t.LengthBuckets[0] != seqLen ||
		seqLen <= 0 || t.BatchTokens%seqLen != 0
}

// LengthBucketBatchShape returns the fixed program shape for one bucket.
// Under length bucketing batch_tokens is a ceiling, so the effective token
// count can be smaller when the bucket width does not divide it exactly.
func (t TrainingSpec) LengthBucketBatchShape(seqLen int) (batchSize, effectiveBatchTokens int) {
	if seqLen <= 0 || t.BatchTokens < seqLen {
		return 0, 0
	}
	batchSize = t.BatchTokens / seqLen
	return batchSize, batchSize * seqLen
}

func validateLengthBuckets(cfg *ArchConfig, source string) error {
	if cfg == nil || !cfg.Training.LengthBucketingEnabled() {
		return nil
	}
	t := &cfg.Training
	if t.EffectiveObjective() != ObjectiveClassification {
		return fmt.Errorf("config %q training.length_buckets requires training.objective=%q", source, ObjectiveClassification)
	}
	if !cfg.DiscreteCodebooksEnabled() && !cfg.LinearFramesEnabled() {
		return fmt.Errorf("config %q training.length_buckets supports input_adapter.kind=%q or %q in v1", source, InputAdapterDiscreteCodebooks, InputAdapterLinearFrames)
	}
	previous := 0
	for i, width := range t.LengthBuckets {
		if width <= previous {
			return fmt.Errorf("config %q training.length_buckets[%d]=%d must be strictly greater than %d", source, i, width, previous)
		}
		if width > cfg.SeqLen {
			return fmt.Errorf("config %q training.length_buckets[%d]=%d exceeds seq_len=%d", source, i, width, cfg.SeqLen)
		}
		previous = width
	}
	largest := t.LengthBuckets[len(t.LengthBuckets)-1]
	if t.BatchTokens < largest {
		return fmt.Errorf("config %q training.batch_tokens=%d must be >= largest length bucket %d", source, t.BatchTokens, largest)
	}
	if cfg.EffectiveNormSpec().Type == NormTypeBatchNorm {
		return fmt.Errorf("config %q training.length_buckets does not support norm_type=%q until padded-row BatchNorm semantics are defined", source, NormTypeBatchNorm)
	}
	return nil
}
