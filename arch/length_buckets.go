package arch

import "fmt"

// BatchSizeConfigured reports whether fixed-row length bucketing was selected.
func (t TrainingSpec) BatchSizeConfigured() bool {
	return t.batchSizeSet || t.BatchSize != 0
}

// FixedLengthBucketBatchSize returns the configured fixed row count, or zero
// when bucket row counts are derived from the batch-token ceiling.
func (t TrainingSpec) FixedLengthBucketBatchSize() int {
	if !t.BatchSizeConfigured() {
		return 0
	}
	return t.BatchSize
}

// LengthBucketingEnabled reports whether classification batches use
// variable-width programs selected from training.length_buckets.
func (t TrainingSpec) LengthBucketingEnabled() bool {
	return len(t.LengthBuckets) > 0
}

// LengthBucketsChangeShape reports whether the configured buckets require
// dynamic batch shapes. A single full-width bucket with an exactly divisible
// token budget or fixed batch size takes the legacy path for strict parity.
func (t TrainingSpec) LengthBucketsChangeShape(seqLen int) bool {
	if !t.LengthBucketingEnabled() {
		return false
	}
	return len(t.LengthBuckets) != 1 || t.LengthBuckets[0] != seqLen ||
		seqLen <= 0 || t.BatchTokens%seqLen != 0
}

// LengthBucketBatchShape returns the fixed program shape for one bucket.
// batch_size fixes rows and varies tokens; otherwise batch_tokens is a ceiling
// and the effective token count may be smaller when width does not divide it.
func (t TrainingSpec) LengthBucketBatchShape(seqLen int) (batchSize, effectiveBatchTokens int) {
	if seqLen <= 0 {
		return 0, 0
	}
	if fixed := t.FixedLengthBucketBatchSize(); fixed > 0 {
		return fixed, fixed * seqLen
	}
	if t.BatchTokens < seqLen {
		return 0, 0
	}
	batchSize = t.BatchTokens / seqLen
	return batchSize, batchSize * seqLen
}

func validateLengthBuckets(cfg *ArchConfig, source string) error {
	if cfg == nil {
		return nil
	}
	t := &cfg.Training
	hasBatchSize := t.batchSizeSet || t.BatchSize != 0
	hasBatchTokens := t.batchTokensSet || (t.BatchTokens != 0 && !t.batchTokensDerivedFromBatchSize)
	if hasBatchSize && hasBatchTokens {
		return fmt.Errorf("config %q training.batch_size and training.batch_tokens are mutually exclusive; set exactly one", source)
	}
	if hasBatchSize {
		if !t.LengthBucketingEnabled() {
			return fmt.Errorf("config %q training.batch_size requires training.length_buckets", source)
		}
		if t.BatchSize <= 0 {
			return fmt.Errorf("config %q training.batch_size=%d must be > 0", source, t.BatchSize)
		}
		maxInt := int(^uint(0) >> 1)
		if cfg.SeqLen > 0 && t.BatchSize > maxInt/cfg.SeqLen {
			return fmt.Errorf("config %q training.batch_size=%d overflows the maximum batch shape at seq_len=%d", source, t.BatchSize, cfg.SeqLen)
		}
		t.BatchTokens = t.BatchSize * cfg.SeqLen
		t.batchTokensDerivedFromBatchSize = true
	}
	if !t.LengthBucketingEnabled() {
		return nil
	}
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
	if t.FixedLengthBucketBatchSize() == 0 && t.BatchTokens < largest {
		return fmt.Errorf("config %q training.batch_tokens=%d must be >= largest length bucket %d", source, t.BatchTokens, largest)
	}
	if cfg.EffectiveNormSpec().Type == NormTypeBatchNorm {
		return fmt.Errorf("config %q training.length_buckets does not support norm_type=%q until padded-row BatchNorm semantics are defined", source, NormTypeBatchNorm)
	}
	return nil
}
