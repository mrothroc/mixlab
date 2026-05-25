package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
)

var mamba3AliasWarningSeen atomic.Bool

const mamba3AliasWarning = `WARN: block type "mamba3" is deprecated; use "gated_linear_ssm". The "mamba3" name will be reassigned to canonical Mamba-3 in a future release.`

// ArchConfig defines a model architecture for mixlab.
type ArchConfig struct {
	Name             string  `json:"name"`
	ModelDim         int     `json:"model_dim"`
	VocabSize        int     `json:"vocab_size"`
	SeqLen           int     `json:"seq_len"`
	MLPMult          float64 `json:"mlp_mult,omitempty"`
	TieEmbeddings    bool    `json:"tie_embeddings,omitempty"`
	BlockScales      bool    `json:"block_scales,omitempty"`
	ResidMix         bool    `json:"resid_mix,omitempty"`
	ParallelResidual bool    `json:"parallel_residual,omitempty"`
	UNet             bool    `json:"unet,omitempty"`
	SmearEmbeddings  bool    `json:"smear_embeddings,omitempty"`
	// SmearEmbeddingsGateShape selects the learned gate used by embedding smearing.
	// Empty defaults to "pr130" when smear_embeddings is enabled.
	SmearEmbeddingsGateShape string       `json:"smear_embeddings_gate_shape,omitempty"`
	BigramVocabSize          int          `json:"bigram_vocab_size,omitempty"`
	BigramDim                int          `json:"bigram_dim,omitempty"`
	TrigramVocabSize         int          `json:"trigram_vocab_size,omitempty"`
	TrigramDim               int          `json:"trigram_dim,omitempty"`
	LogitSoftcap             float32      `json:"logit_softcap,omitempty"`
	Dropout                  float32      `json:"dropout,omitempty"`
	MTP                      *MTPSpec     `json:"mtp,omitempty"`
	Backout                  *BackoutSpec `json:"backout,omitempty"`
	Data                     DataSpec     `json:"data,omitempty"`

	Blocks           []BlockSpec           `json:"blocks"`
	Recurrence       []int                 `json:"recurrence,omitempty"`
	RecurrencePhases []RecurrencePhaseSpec `json:"recurrence_phases,omitempty"`

	// Schema-2 recurrence phase fields are reserved for a possible future API.
	// They are parsed so validation can reject them explicitly instead of failing
	// with a generic unknown-field error.
	ExecutionOrder             []int                           `json:"execution_order,omitempty"`
	RecurrencePhaseActivations []RecurrencePhaseActivationSpec `json:"recurrence_phase_activations,omitempty"`

	Training TrainingSpec `json:"training"`
	Eval     *EvalSpec    `json:"eval,omitempty"`

	recurrencePhasesSet           bool
	executionOrderSet             bool
	recurrencePhaseActivationsSet bool
}

// DataSpec holds data-loader behavior.
type DataSpec struct {
	NoShardShuffle bool `json:"no_shard_shuffle,omitempty"`
}

// Types and validation helpers for recurrence_phases live in
// arch/config_recurrence_phases.go.

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

// WeightSpec declares a named weight for a custom block with symbolic shape.
type WeightSpec struct {
	Name  string   `json:"name"`
	Shape []string `json:"shape"`
}

// CustomWeightSpec is kept as a source-compatible alias for older tests and callers.
type CustomWeightSpec = WeightSpec

// OpSpec declares one operation in a custom block.
type OpSpec struct {
	Op      string                 `json:"op"`
	Inputs  []string               `json:"inputs"`
	Output  string                 `json:"output"`
	Outputs []string               `json:"outputs"`
	Params  map[string]interface{} `json:"params"`
}

// CustomOpSpec is kept as a source-compatible alias for older tests and callers.
type CustomOpSpec = OpSpec

// BlockSpec describes a single model block.
type BlockSpec struct {
	Type             string       `json:"type"`
	Name             string       `json:"name,omitempty"` // custom block name (required for type=custom)
	WeightGroup      string       `json:"weight_group,omitempty"`
	ParallelResidual *bool        `json:"parallel_residual,omitempty"` // enable/disable parallel residual for this block pair start
	Heads            int          `json:"heads"`
	KVHeads          int          `json:"kv_heads,omitempty"`
	KVSource         int          `json:"kv_source,omitempty"`        // -1 or 0 = compute own KV; positive = reuse KV from block N (1-indexed)
	RopeDims         int          `json:"rope_dims,omitempty"`        // RoPE rotation dims per head; 0 or head_dim = full RoPE
	QKGain           float64      `json:"qk_gain,omitempty"`          // per-head learnable QK scaling; 0 disables
	XSA              bool         `json:"xsa,omitempty"`              // enable V-orthogonal projection after attention
	WindowSize       int          `json:"window_size,omitempty"`      // plain: sliding causal attention width; 0 = full causal attention
	AttentionMask    string       `json:"attention_mask,omitempty"`   // plain: "causal", "bidirectional", or "none"; empty resolves from training objective.
	SkipAttention    bool         `json:"skip_attention,omitempty"`   // plain: bypass attention while preserving weight layout.
	SparseAttnGate   bool         `json:"sparse_attn_gate,omitempty"` // plain: narrow per-head output gate over the first gate_window head channels.
	InnerDim         int          `json:"inner_dim,omitempty"`        // Mamba inner dimension; defaults to model_dim.
	DK               int          `json:"d_k,omitempty"`              // gated_deltanet: key/query dim per head.
	DV               int          `json:"d_v,omitempty"`              // gated_deltanet: value dim per head; defaults to 2*d_k.
	DState           int          `json:"d_state,omitempty"`          // hgrn2: matrix-state key/query dim per head; defaults to model_dim/heads.
	KVShare          *bool        `json:"kv_share,omitempty"`         // gated_deltanet: share the K/V projection when true (default).
	ScanChunkSize    *int         `json:"scan_chunk_size,omitempty"`  // gated_deltanet/mamba3-canonical: chunk size for chunked scan; 0 keeps the naive/full scan.
	StateSize        int          `json:"state_size,omitempty"`       // Mamba-3 canonical state expansion; defaults to 16.
	NGroups          int          `json:"n_groups,omitempty"`         // Mamba-3 canonical grouped/MIMO axis; defaults to 4.
	DTRank           int          `json:"dt_rank,omitempty"`          // Mamba-3 canonical low-rank dt/lambda/theta rank; defaults to max(inner/16,1).
	ConvKernel       int          `json:"conv_kernel,omitempty"`      // Mamba-3 canonical causal conv width; defaults to 4.
	UseConv          *bool        `json:"use_conv,omitempty"`         // Mamba-3 canonical short conv toggle; defaults to true.
	DTMin            float64      `json:"dt_min,omitempty"`           // Mamba-3 canonical dt init lower bound; defaults to 0.001.
	DTMax            float64      `json:"dt_max,omitempty"`           // Mamba-3 canonical dt init upper bound; defaults to 0.1.
	NumLatents       int          `json:"num_latents,omitempty"`      // Perceiver/bottleneck latent count.
	SourceStream     string       `json:"source_stream,omitempty"`    // cross_attention: stream providing K/V.
	Decay            float64      `json:"decay,omitempty"`            // RetNet: initial decay rate in (0,1); defaults to 0.95.
	Activation       string       `json:"activation,omitempty"`       // mlp: "silu" (default), "gelu", "relu", "leaky_relu_sq".
	LeakySlope       float64      `json:"leaky_slope,omitempty"`      // mlp leaky_relu_sq negative slope; defaults to 0.5.
	Weights          []WeightSpec `json:"weights,omitempty"`          // custom block weight declarations
	Ops              []OpSpec     `json:"ops,omitempty"`              // custom block operation sequence
}

// TrainingPhase defines one contiguous training phase with a fixed LR.
type TrainingPhase struct {
	Steps int     `json:"steps"`
	LR    float64 `json:"lr"`
	Label string  `json:"label,omitempty"`
}

// DistillationSpec configures optional in-training teacher distillation.
type DistillationSpec struct {
	TeacherCheckpoints []string `json:"teacher_checkpoints,omitempty"`
	TeacherConfigs     []string `json:"teacher_configs,omitempty"`
	LossWeightKL       float64  `json:"loss_weight_kl,omitempty"`
	LossWeightCE       float64  `json:"loss_weight_ce,omitempty"`
	EnsembleStrategy   string   `json:"ensemble_strategy,omitempty"`
}

// TrainingSpec holds training hyperparameters.
type TrainingSpec struct {
	Steps                             int               `json:"steps"`
	LR                                float64           `json:"lr"`
	Phases                            []TrainingPhase   `json:"phases,omitempty"`
	Objective                         string            `json:"objective,omitempty"`
	MLMMaskProb                       float64           `json:"mlm_mask_prob,omitempty"`
	MLMMaskTokenID                    int               `json:"mlm_mask_token_id,omitempty"`
	MLMMaskTokenProb                  float64           `json:"mlm_mask_token_prob,omitempty"`
	MLMRandomTokenProb                float64           `json:"mlm_random_token_prob,omitempty"`
	MLMKeptUnchangedProb              float64           `json:"mlm_kept_unchanged_prob,omitempty"`
	HybridCLMFraction                 float64           `json:"hybrid_clm_fraction,omitempty"`
	HybridSecondaryObjective          string            `json:"hybrid_secondary_objective,omitempty"`
	Distillation                      *DistillationSpec `json:"distillation,omitempty"`
	WarmdownSteps                     int               `json:"warmdown_steps,omitempty"`
	TargetValLoss                     float64           `json:"target_val_loss,omitempty"`
	FirstByteMask                     bool              `json:"first_byte_mask,omitempty"`
	FirstByteMaskValid                []int32           `json:"-"`
	RecurrenceActivationFrac          float64           `json:"recurrence_activation_frac,omitempty"`
	RecurrenceActivationStep          int               `json:"recurrence_activation_step,omitempty"`
	TTTSteps                          int               `json:"ttt_steps,omitempty"`
	TTTMode                           string            `json:"ttt_mode,omitempty"`
	TTTLR                             float64           `json:"ttt_lr,omitempty"`
	TTTRank                           int               `json:"ttt_rank,omitempty"`
	HardwareTFLOPs                    float64           `json:"hardware_tflops,omitempty"` // peak hardware TFLOPS (e.g., 400 for M1 Max, 312 for A100)
	GradClip                          float32           `json:"grad_clip"`
	WeightDecay                       float32           `json:"weight_decay"`
	CautiousWeightDecay               bool              `json:"cautious_weight_decay,omitempty"`
	CautiousWeightDecayActivationFrac float64           `json:"cautious_weight_decay_activation_frac,omitempty"`
	Beta1                             float32           `json:"beta1"`
	Beta2                             float32           `json:"beta2"`
	Epsilon                           float32           `json:"epsilon"`
	Seed                              int64             `json:"seed"`
	BatchTokens                       int               `json:"batch_tokens"`
	ShuffleChunkTokens                int               `json:"shuffle_chunk_tokens,omitempty"`
	EmbedLR                           float32           `json:"embed_lr,omitempty"`
	MatrixLR                          float32           `json:"matrix_lr,omitempty"`
	ScalarLR                          float32           `json:"scalar_lr,omitempty"`
	HeadLR                            float32           `json:"head_lr,omitempty"`
	MuonMomentum                      float32           `json:"muon_momentum,omitempty"`
	MuonBackendSteps                  int               `json:"muon_backend_steps,omitempty"`
	// NewtonSchulzVariant controls Muon's Newton-Schulz coefficient choice.
	// "" or "fixed" = canonical (3.4445, -4.7750, 2.0315) per iteration.
	// "polar_express" = per-iteration Chebyshev minimax-optimal tuples
	// (You Jiacheng, arXiv:2505.16932).
	NewtonSchulzVariant string  `json:"newton_schulz_variant,omitempty"`
	MuonNesterov        *bool   `json:"muon_nesterov,omitempty"`
	Optimizer           string  `json:"optimizer,omitempty"` // "muon" (default), "muon_eq_r", "normuon", or "adamw" for matrix weights
	QAT                 string  `json:"qat,omitempty"`       // "none" (default), "int8", or "int6"
	QATStart            int     `json:"qat_start,omitempty"`
	WeightInit          string  `json:"weight_init,omitempty"`     // "xavier_uniform" (default) or "normal"
	WeightInitStd       float32 `json:"weight_init_std,omitempty"` // std for normal init (default 0.02)
	EmbedWeightDecay    float32 `json:"embed_weight_decay,omitempty"`
	MatrixWeightDecay   float32 `json:"matrix_weight_decay,omitempty"`
	ScalarWeightDecay   float32 `json:"scalar_weight_decay,omitempty"`
	HeadWeightDecay     float32 `json:"head_weight_decay,omitempty"`
	// MinLRFraction sets the minimum LR as a fraction of peak LR.
	// 0 (default) = current behavior (warmdown ends near 0).
	// 0.10 = recommended (LR never drops below 10% of peak).
	// Used as an absolute floor across both cosine decay and warmdown phases.
	MinLRFraction float32 `json:"min_lr_fraction,omitempty"`
	SWAStart      int     `json:"swa_start,omitempty"`
	SWADecay      float32 `json:"swa_decay,omitempty"`
	SWAInterval   int     `json:"swa_interval,omitempty"`

	mlmMaskProbSet        bool
	mlmMaskTokenIDSet     bool
	mlmReplacementProbSet bool
	hybridCLMFractionSet  bool
}

func (t *TrainingSpec) UnmarshalJSON(data []byte) error {
	type alias TrainingSpec
	var raw alias
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*t = TrainingSpec(raw)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	_, t.mlmMaskProbSet = fields["mlm_mask_prob"]
	_, t.mlmMaskTokenIDSet = fields["mlm_mask_token_id"]
	_, maskProbSet := fields["mlm_mask_token_prob"]
	_, randomProbSet := fields["mlm_random_token_prob"]
	_, keptProbSet := fields["mlm_kept_unchanged_prob"]
	t.mlmReplacementProbSet = maskProbSet || randomProbSet || keptProbSet
	_, t.hybridCLMFractionSet = fields["hybrid_clm_fraction"]
	return nil
}

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

// TotalSteps returns the effective training step count.
// When phases are configured, their summed length takes precedence.
func (t TrainingSpec) TotalSteps() int {
	if len(t.Phases) == 0 {
		return t.Steps
	}
	total := 0
	for _, phase := range t.Phases {
		total += phase.Steps
	}
	return total
}

// DefaultTrainingSpec returns sensible training defaults.
func DefaultTrainingSpec() TrainingSpec {
	return TrainingSpec{
		Steps:             200,
		LR:                3e-4,
		WeightDecay:       0.01,
		Beta1:             0.9,
		Beta2:             0.95,
		Epsilon:           1e-8,
		Seed:              42,
		BatchTokens:       1024,
		TTTMode:           "full",
		TTTLR:             1e-5,
		TTTRank:           4,
		QAT:               "none",
		MuonMomentum:      0.9,
		MuonBackendSteps:  5,
		EmbedWeightDecay:  0.01,
		MatrixWeightDecay: 0.01,
		ScalarWeightDecay: 0.01,
		HeadWeightDecay:   0.01,
		SWADecay:          0.999,
		SWAInterval:       10,
	}
}

// ApplyDefaults fills omitted zero-valued training fields using the same
// defaults applied when parsing a JSON architecture config.
func (t *TrainingSpec) ApplyDefaults() {
	if t == nil {
		return
	}
	d := DefaultTrainingSpec()
	if t.Steps <= 0 {
		t.Steps = d.Steps
	}
	if t.LR <= 0 {
		t.LR = d.LR
	}
	// Note: seed=0 in JSON is indistinguishable from omitted; defaults to 42.
	if t.Seed <= 0 {
		t.Seed = d.Seed
	}
	if t.BatchTokens <= 0 {
		t.BatchTokens = d.BatchTokens
	}
	t.Objective = normalizeTrainingObjective(t.Objective)
	if !t.mlmMaskProbSet && t.MLMMaskProb == 0 {
		t.MLMMaskProb = 0.15
	}
	if !t.mlmReplacementProbSet && t.MLMMaskTokenProb == 0 && t.MLMRandomTokenProb == 0 && t.MLMKeptUnchangedProb == 0 {
		t.MLMMaskTokenProb = 0.8
		t.MLMRandomTokenProb = 0.1
		t.MLMKeptUnchangedProb = 0.1
	}
	if !t.hybridCLMFractionSet && t.HybridCLMFraction == 0 {
		t.HybridCLMFraction = 0.5
	}
	if strings.TrimSpace(t.HybridSecondaryObjective) == "" {
		t.HybridSecondaryObjective = ObjectiveMNTP
	} else {
		t.HybridSecondaryObjective = normalizeTrainingObjective(t.HybridSecondaryObjective)
	}
	if t.WeightDecay == 0 {
		t.WeightDecay = d.WeightDecay
	}
	if t.Beta1 == 0 {
		t.Beta1 = d.Beta1
	}
	if t.Beta2 == 0 {
		t.Beta2 = d.Beta2
	}
	if t.Epsilon == 0 {
		t.Epsilon = d.Epsilon
	}
	if t.EmbedLR == 0 {
		t.EmbedLR = float32(t.LR)
	}
	if t.MatrixLR == 0 {
		t.MatrixLR = float32(t.LR)
	}
	if t.ScalarLR == 0 {
		t.ScalarLR = float32(t.LR)
	}
	if t.HeadLR == 0 {
		t.HeadLR = float32(t.LR)
	}
	if t.TTTLR == 0 {
		t.TTTLR = d.TTTLR
	}
	if t.TTTMode == "" {
		t.TTTMode = d.TTTMode
	}
	t.QAT = strings.ToLower(strings.TrimSpace(t.QAT))
	if t.QAT == "" {
		t.QAT = d.QAT
	}
	if t.TTTRank == 0 {
		t.TTTRank = d.TTTRank
	}
	if t.MuonMomentum == 0 {
		t.MuonMomentum = t.Beta1
	}
	if t.MuonBackendSteps <= 0 {
		t.MuonBackendSteps = d.MuonBackendSteps
	}
	t.Optimizer = strings.ToLower(strings.TrimSpace(t.Optimizer))
	if t.EmbedWeightDecay == 0 {
		t.EmbedWeightDecay = t.WeightDecay
	}
	if t.MatrixWeightDecay == 0 {
		t.MatrixWeightDecay = t.WeightDecay
	}
	if t.ScalarWeightDecay == 0 {
		t.ScalarWeightDecay = t.WeightDecay
	}
	if t.HeadWeightDecay == 0 {
		t.HeadWeightDecay = t.WeightDecay
	}
	if t.SWADecay == 0 {
		t.SWADecay = d.SWADecay
	}
	if t.SWAInterval <= 0 {
		t.SWAInterval = d.SWAInterval
	}
}

// LoadArchConfig reads and validates a JSON architecture config from path.
func LoadArchConfig(path string) (*ArchConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if filepath.IsAbs(path) {
			return nil, fmt.Errorf("read config %q: %w", path, err)
		}
		b, err = os.ReadFile(filepath.Join("..", path))
		if err != nil {
			return nil, fmt.Errorf("read config %q: %w", path, err)
		}
	}
	return ParseArchConfig(b, path)
}

// stripJSONComments removes // line comments from JSONC input.
// This allows config files to contain inline documentation.
func stripJSONComments(data []byte) []byte {
	var out []byte
	inString := false
	escaped := false
	for i := 0; i < len(data); i++ {
		c := data[i]
		if escaped {
			out = append(out, c)
			escaped = false
			continue
		}
		if inString {
			out = append(out, c)
			switch c {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
			out = append(out, c)
			continue
		case '/':
			if i+1 >= len(data) || data[i+1] != '/' {
				break
			}
			// Skip to end of line.
			for i < len(data) && data[i] != '\n' {
				i++
			}
			if i < len(data) {
				out = append(out, '\n')
			}
			continue
		}
		out = append(out, c)
	}
	return out
}

// ParseArchConfig parses and validates a JSON architecture config.
// The source parameter is used in error messages (typically the file path).
// Supports JSONC: // line comments are stripped before parsing.
// Unknown JSON fields are rejected to prevent silent misconfiguration.
func ParseArchConfig(data []byte, source string) (*ArchConfig, error) {
	data = stripJSONComments(data)
	var cfg ArchConfig
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w (check field names against docs/config-reference.md)", source, err)
	}
	warnDeprecatedMamba3Blocks(cfg.Blocks)
	return validateConfig(&cfg, source)
}

func warnDeprecatedMamba3Blocks(blocks []BlockSpec) {
	for _, block := range blocks {
		if strings.EqualFold(strings.TrimSpace(block.Type), "mamba3") {
			if mamba3AliasWarningSeen.CompareAndSwap(false, true) {
				fmt.Fprintln(os.Stderr, mamba3AliasWarning)
			}
			return
		}
	}
}

// validateConfig checks invariants and applies defaults.
func validateConfig(cfg *ArchConfig, source string) (*ArchConfig, error) {
	if cfg.Name == "" {
		cfg.Name = source
	}
	if cfg.ModelDim <= 0 {
		return nil, fmt.Errorf("config %q missing/invalid model_dim", source)
	}
	if cfg.VocabSize <= 0 {
		return nil, fmt.Errorf("config %q missing/invalid vocab_size", source)
	}
	if cfg.SeqLen <= 0 {
		cfg.SeqLen = 128
	}
	if cfg.BigramVocabSize < 0 {
		return nil, fmt.Errorf("config %q has invalid bigram_vocab_size=%d (must be >= 0)", source, cfg.BigramVocabSize)
	}
	if cfg.BigramDim < 0 {
		return nil, fmt.Errorf("config %q has invalid bigram_dim=%d (must be >= 0)", source, cfg.BigramDim)
	}
	if cfg.TrigramVocabSize < 0 {
		return nil, fmt.Errorf("config %q has invalid trigram_vocab_size=%d (must be >= 0)", source, cfg.TrigramVocabSize)
	}
	if cfg.TrigramDim < 0 {
		return nil, fmt.Errorf("config %q has invalid trigram_dim=%d (must be >= 0)", source, cfg.TrigramDim)
	}
	if cfg.BigramVocabSize > 0 {
		if cfg.BigramVocabSize <= 1 {
			return nil, fmt.Errorf("config %q has invalid bigram_vocab_size=%d (must be > 1 when enabled)", source, cfg.BigramVocabSize)
		}
		if cfg.BigramDim == 0 {
			cfg.BigramDim = cfg.ModelDim
		}
	} else {
		cfg.BigramDim = 0
	}
	if cfg.TrigramVocabSize > 0 {
		if cfg.TrigramVocabSize <= 1 {
			return nil, fmt.Errorf("config %q has invalid trigram_vocab_size=%d (must be > 1 when enabled)", source, cfg.TrigramVocabSize)
		}
		if cfg.TrigramDim == 0 {
			cfg.TrigramDim = cfg.EffectiveTrigramDim()
		}
	} else {
		cfg.TrigramDim = 0
	}
	if cfg.MLPMult == 0 {
		cfg.MLPMult = 2.67
	}
	if cfg.MLPMult <= 0 {
		return nil, fmt.Errorf("config %q has invalid mlp_mult=%g (must be > 0)", source, cfg.MLPMult)
	}
	if err := validateSmearEmbeddings(cfg, source); err != nil {
		return nil, err
	}
	if cfg.Dropout < 0 || cfg.Dropout > 1 {
		return nil, fmt.Errorf("config %q has invalid dropout=%g (must be in [0,1])", source, cfg.Dropout)
	}
	if cfg.MTP != nil {
		if cfg.MTP.nSet && cfg.MTP.N < 1 {
			return nil, fmt.Errorf("config %q has invalid mtp.n=%d (must be >= 1)", source, cfg.MTP.N)
		}
		if cfg.MTP.N <= 0 {
			cfg.MTP.N = 1
		}
		if cfg.MTP.N > cfg.SeqLen {
			return nil, fmt.Errorf("config %q has invalid mtp.n=%d (must be <= seq_len=%d)", source, cfg.MTP.N, cfg.SeqLen)
		}
		if (cfg.MTP.lossWeightsSet || len(cfg.MTP.LossWeights) > 0) && len(cfg.MTP.LossWeights) != cfg.MTP.N {
			return nil, fmt.Errorf("config %q has invalid mtp.loss_weights length=%d (must equal mtp.n=%d)", source, len(cfg.MTP.LossWeights), cfg.MTP.N)
		}
		weights := cfg.MTP.EffectiveLossWeights()
		var weightSum float32
		for i, w := range weights {
			if w < 0 || w != w {
				return nil, fmt.Errorf("config %q has invalid mtp.loss_weights[%d]=%g (must be finite and >= 0)", source, i, w)
			}
			weightSum += w
		}
		if weightSum <= 0 {
			return nil, fmt.Errorf("config %q has invalid mtp.loss_weights (sum must be > 0)", source)
		}
		frac := cfg.MTP.EffectiveUntieEmbedAtFrac()
		if frac < 0 || frac > 1 || frac != frac {
			return nil, fmt.Errorf("config %q has invalid mtp.untie_embed_at_frac=%g (must be in [0,1])", source, frac)
		}
		if frac < 1 && !cfg.TieEmbeddings {
			return nil, fmt.Errorf("config %q sets mtp.untie_embed_at_frac=%g but tie_embeddings is false", source, frac)
		}
		activateFrac := cfg.MTP.EffectiveActivateAtFrac()
		if activateFrac < 0 || activateFrac > 1 || activateFrac != activateFrac {
			return nil, fmt.Errorf("config %q has invalid mtp.activate_at_frac=%g (must be in [0,1])", source, activateFrac)
		}
		if activateFrac > 0 && cfg.MTP.EffectiveN() <= 1 {
			return nil, fmt.Errorf("config %q mtp.activate_at_frac requires n >= 2", source)
		}
	}

	for i, b := range cfg.Blocks {
		if err := validateBlockSpec(b, source, "blocks", i); err != nil {
			return nil, err
		}
		if err := validateBlockRopeDims(b, cfg.ModelDim, source, "blocks", i); err != nil {
			return nil, err
		}
		if err := validateRecurrentMixerDims(b, cfg.ModelDim, source, "blocks", i); err != nil {
			return nil, err
		}
	}

	if len(cfg.Blocks) == 0 {
		return nil, fmt.Errorf("config %q must define at least one block", source)
	}
	if err := validateBackout(cfg, source); err != nil {
		return nil, err
	}
	if err := validateWeightGroups(cfg, source); err != nil {
		return nil, err
	}
	if err := validateRecurrence(cfg, source); err != nil {
		return nil, err
	}
	if err := validateKVSources(cfg, source); err != nil {
		return nil, err
	}
	if err := validateParallelResidual(cfg, source); err != nil {
		return nil, err
	}

	cfg.Training.ApplyDefaults()
	if err := validateTrainingObjective(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingDistillation(cfg, source); err != nil {
		return nil, err
	}
	for i, b := range cfg.Blocks {
		if err := validatePlainAttentionMask(cfg, source, b, "blocks", i); err != nil {
			return nil, err
		}
	}
	if cfg.Training.WeightDecay < 0 {
		return nil, fmt.Errorf("config %q has invalid training.weight_decay=%g (must be >= 0)", source, cfg.Training.WeightDecay)
	}
	if cfg.Training.EmbedLR < 0 || cfg.Training.MatrixLR < 0 || cfg.Training.ScalarLR < 0 || cfg.Training.HeadLR < 0 {
		return nil, fmt.Errorf("config %q has invalid per-group learning rate (must be >= 0)", source)
	}
	if cfg.Training.MuonMomentum < 0 {
		return nil, fmt.Errorf("config %q has invalid training.muon_momentum=%g (must be >= 0)", source, cfg.Training.MuonMomentum)
	}
	switch cfg.Training.Optimizer {
	case "", "adamw", "muon", "muon_eq_r", "normuon":
	default:
		return nil, fmt.Errorf("config %q has invalid training.optimizer=%q (must be \"adamw\", \"muon\", \"muon_eq_r\", or \"normuon\")", source, cfg.Training.Optimizer)
	}
	switch strings.ToLower(strings.TrimSpace(cfg.Training.NewtonSchulzVariant)) {
	case "", "fixed":
		cfg.Training.NewtonSchulzVariant = "fixed"
	case "polar_express":
		cfg.Training.NewtonSchulzVariant = "polar_express"
	default:
		return nil, fmt.Errorf("config %q has invalid training.newton_schulz_variant=%q (must be \"fixed\" or \"polar_express\")", source, cfg.Training.NewtonSchulzVariant)
	}
	if cfg.Training.GradClip < 0 {
		return nil, fmt.Errorf("config %q has invalid training.grad_clip=%g (must be >= 0)", source, cfg.Training.GradClip)
	}
	if cfg.Training.MinLRFraction < 0 || cfg.Training.MinLRFraction >= 1 {
		return nil, fmt.Errorf("config %q has invalid training.min_lr_fraction=%g (must be in [0,1))", source, cfg.Training.MinLRFraction)
	}
	if cfg.Training.EmbedWeightDecay < 0 || cfg.Training.MatrixWeightDecay < 0 ||
		cfg.Training.ScalarWeightDecay < 0 || cfg.Training.HeadWeightDecay < 0 {
		return nil, fmt.Errorf("config %q has invalid per-group weight decay (must be >= 0)", source)
	}
	if cfg.Training.SWAStart < 0 {
		return nil, fmt.Errorf("config %q has invalid training.swa_start=%d (must be >= 0)", source, cfg.Training.SWAStart)
	}
	if cfg.Training.SWADecay < 0 || cfg.Training.SWADecay >= 1 {
		return nil, fmt.Errorf("config %q has invalid training.swa_decay=%g (must be in [0,1))", source, cfg.Training.SWADecay)
	}
	if cfg.Training.WarmdownSteps < 0 {
		return nil, fmt.Errorf("config %q has invalid training.warmdown_steps=%d (must be >= 0)", source, cfg.Training.WarmdownSteps)
	}
	if cfg.Training.RecurrenceActivationFrac < 0 || cfg.Training.RecurrenceActivationFrac > 1 {
		return nil, fmt.Errorf("config %q has invalid training.recurrence_activation_frac=%g (must be in [0,1])", source, cfg.Training.RecurrenceActivationFrac)
	}
	if cfg.Training.RecurrenceActivationStep < 0 {
		return nil, fmt.Errorf("config %q has invalid training.recurrence_activation_step=%d (must be >= 0)", source, cfg.Training.RecurrenceActivationStep)
	}
	if cfg.Training.RecurrenceActivationFrac > 0 && cfg.Training.RecurrenceActivationStep > 0 {
		return nil, fmt.Errorf("config %q cannot set both training.recurrence_activation_frac and training.recurrence_activation_step", source)
	}
	if err := validateCautiousWeightDecay(cfg, source); err != nil {
		return nil, err
	}
	for i, phase := range cfg.Training.Phases {
		if phase.Steps <= 0 {
			return nil, fmt.Errorf("config %q has invalid training.phases[%d].steps=%d (must be > 0)", source, i, phase.Steps)
		}
		if phase.LR <= 0 {
			return nil, fmt.Errorf("config %q has invalid training.phases[%d].lr=%g (must be > 0)", source, i, phase.LR)
		}
	}
	if len(cfg.Training.Phases) > 0 {
		cfg.Training.Steps = cfg.Training.TotalSteps()
	}
	if err := validateRecurrencePhases(cfg, source); err != nil {
		return nil, err
	}
	if cfg.Training.TargetValLoss < 0 {
		return nil, fmt.Errorf("config %q has invalid training.target_val_loss=%g (must be >= 0)", source, cfg.Training.TargetValLoss)
	}
	if cfg.Training.HardwareTFLOPs < 0 {
		return nil, fmt.Errorf("config %q has invalid training.hardware_tflops=%g (must be >= 0)", source, cfg.Training.HardwareTFLOPs)
	}
	if cfg.Training.TTTSteps < 0 {
		return nil, fmt.Errorf("config %q has invalid training.ttt_steps=%d (must be >= 0)", source, cfg.Training.TTTSteps)
	}
	if cfg.Training.TTTMode != "full" && cfg.Training.TTTMode != "lora" {
		return nil, fmt.Errorf("config %q has invalid training.ttt_mode=%q (must be \"full\" or \"lora\")", source, cfg.Training.TTTMode)
	}
	if cfg.Training.QAT != "none" && cfg.Training.QAT != "int8" && cfg.Training.QAT != "int6" {
		return nil, fmt.Errorf("config %q has invalid training.qat=%q (must be \"none\", \"int8\", or \"int6\")", source, cfg.Training.QAT)
	}
	if cfg.Training.QATStart < 0 {
		return nil, fmt.Errorf("config %q has invalid training.qat_start=%d (must be >= 0)", source, cfg.Training.QATStart)
	}
	if cfg.Training.QATStart > 0 && cfg.Training.QAT == "none" {
		return nil, fmt.Errorf("config %q has training.qat_start=%d but training.qat is not set", source, cfg.Training.QATStart)
	}
	if cfg.Training.TTTLR < 0 {
		return nil, fmt.Errorf("config %q has invalid training.ttt_lr=%g (must be >= 0)", source, cfg.Training.TTTLR)
	}
	if cfg.Training.TTTRank <= 0 {
		return nil, fmt.Errorf("config %q has invalid training.ttt_rank=%d (must be > 0)", source, cfg.Training.TTTRank)
	}
	if err := validateEvalSpec(cfg, source); err != nil {
		return nil, err
	}

	return cfg, nil
}
