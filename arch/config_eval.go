package arch

// EvalSpec holds optional evaluation-only behavior. When omitted, or when
// ttt_mode is "none", eval runs exactly as the standard single-pass path.
type EvalSpec struct {
	TTTMode       string   `json:"ttt_mode,omitempty"`
	ChunkTokens   int      `json:"chunk_tokens,omitempty"`
	TTTEpochs     int      `json:"ttt_epochs,omitempty"`
	TTTLR         float64  `json:"ttt_lr,omitempty"`
	TTTMomentum   *float64 `json:"ttt_momentum,omitempty"`
	TTTLRSchedule string   `json:"ttt_lr_schedule,omitempty"`
}

// DefaultEvalSpec returns the inactive eval defaults.
func DefaultEvalSpec() EvalSpec {
	return EvalSpec{TTTMode: "none"}
}

// DefaultLegalChunkSGDEvalSpec returns the modded-nanogpt style eval-time TTT defaults.
func DefaultLegalChunkSGDEvalSpec() EvalSpec {
	momentum := 0.9
	return EvalSpec{
		TTTMode:       "legal_chunk_sgd",
		ChunkTokens:   32768,
		TTTEpochs:     3,
		TTTLR:         0.005,
		TTTMomentum:   &momentum,
		TTTLRSchedule: "cosine",
	}
}

// EffectiveEvalSpec returns an inactive spec when eval is omitted.
func (c *ArchConfig) EffectiveEvalSpec() EvalSpec {
	if c == nil || c.Eval == nil {
		return DefaultEvalSpec()
	}
	return *c.Eval
}

// LegalChunkSGDEnabled reports whether eval-time score-first chunk SGD is active.
func (e EvalSpec) LegalChunkSGDEnabled() bool {
	return e.TTTMode == "legal_chunk_sgd"
}

// EffectiveTTTMomentum returns the eval-time SGD momentum value.
func (e EvalSpec) EffectiveTTTMomentum() float64 {
	if e.TTTMomentum == nil {
		return 0.9
	}
	return *e.TTTMomentum
}
