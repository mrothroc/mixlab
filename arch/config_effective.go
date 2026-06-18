package arch

// EffectiveBigramDim returns the configured bigram embedding dimension,
// defaulting to model_dim when bigram embeddings are enabled but bigram_dim is unset.
func (c *ArchConfig) EffectiveBigramDim() int {
	if c == nil || c.BigramVocabSize <= 0 {
		return 0
	}
	if c.BigramDim <= 0 {
		return c.ModelDim
	}
	return c.BigramDim
}

// EffectiveCharDim returns the configured character feature embedding
// dimension, defaulting to model_dim when char features are enabled but
// char_dim is unset.
func (c *ArchConfig) EffectiveCharDim() int {
	if c == nil || c.CharVocabSize <= 0 {
		return 0
	}
	if c.CharDim <= 0 {
		return c.ModelDim
	}
	return c.CharDim
}

// EffectiveCharMaxPerToken returns the fixed sparse char-slot count used for
// each token id, defaulting to 16 when char features are enabled.
func (c *ArchConfig) EffectiveCharMaxPerToken() int {
	if c == nil || c.CharVocabSize <= 0 {
		return 0
	}
	if c.CharMaxPerToken <= 0 {
		return 16
	}
	return c.CharMaxPerToken
}

// EffectiveTrigramDim returns the configured trigram embedding dimension,
// defaulting to bigram_dim or model_dim when trigram embeddings are enabled
// but trigram_dim is unset.
func (c *ArchConfig) EffectiveTrigramDim() int {
	if c == nil || c.TrigramVocabSize <= 0 {
		return 0
	}
	if c.TrigramDim > 0 {
		return c.TrigramDim
	}
	if c.BigramDim > 0 {
		return c.BigramDim
	}
	return c.ModelDim
}

// EffectiveMLPMult returns the configured FFN expansion multiplier,
// defaulting to 2.67 when unset.
func (c *ArchConfig) EffectiveMLPMult() float64 {
	if c == nil || c.MLPMult <= 0 {
		return 2.67
	}
	return c.MLPMult
}

// EffectiveHiddenDropout returns dropout used on residual/FFN projection
// outputs. The legacy top-level dropout value is the default when hidden_dropout
// is omitted.
func (c *ArchConfig) EffectiveHiddenDropout() float32 {
	if c == nil {
		return 0
	}
	if c.hiddenDropoutSet {
		return c.HiddenDropout
	}
	return c.Dropout
}

// EffectiveAttnDropout returns dropout used on attention probabilities. The
// legacy top-level dropout value is the default when attn_dropout is omitted.
func (c *ArchConfig) EffectiveAttnDropout() float32 {
	if c == nil {
		return 0
	}
	if c.attnDropoutSet {
		return c.AttnDropout
	}
	return c.Dropout
}
