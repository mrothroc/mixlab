package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

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
	BigramVocabSize  int     `json:"bigram_vocab_size,omitempty"`
	BigramDim        int     `json:"bigram_dim,omitempty"`
	TrigramVocabSize int     `json:"trigram_vocab_size,omitempty"`
	TrigramDim       int     `json:"trigram_dim,omitempty"`
	LogitSoftcap     float32 `json:"logit_softcap,omitempty"`
	Dropout          float32 `json:"dropout,omitempty"`

	Blocks     []BlockSpec `json:"blocks"`
	Recurrence []int       `json:"recurrence,omitempty"`

	Training TrainingSpec `json:"training"`
}

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
	Type          string       `json:"type"`
	Name          string       `json:"name,omitempty"` // custom block name (required for type=custom)
	WeightGroup   string       `json:"weight_group,omitempty"`
	Heads         int          `json:"heads"`
	KVHeads       int          `json:"kv_heads,omitempty"`
	KVSource      int          `json:"kv_source,omitempty"`      // -1 or 0 = compute own KV; positive = reuse KV from block N (1-indexed)
	RopeDims      int          `json:"rope_dims,omitempty"`      // RoPE rotation dims per head; 0 or head_dim = full RoPE
	QKGain        float64      `json:"qk_gain,omitempty"`        // per-head learnable QK scaling; 0 disables
	XSA           bool         `json:"xsa,omitempty"`            // enable V-orthogonal projection after attention
	SkipAttention bool         `json:"skip_attention,omitempty"` // plain: bypass attention while preserving weight layout.
	InnerDim      int          `json:"inner_dim,omitempty"`      // Mamba inner dimension; defaults to model_dim.
	NumLatents    int          `json:"num_latents,omitempty"`    // Perceiver/bottleneck latent count.
	SourceStream  string       `json:"source_stream,omitempty"`  // cross_attention: stream providing K/V.
	Decay         float64      `json:"decay,omitempty"`          // RetNet: initial decay rate in (0,1); defaults to 0.95.
	Activation    string       `json:"activation,omitempty"`     // mlp: "silu" (default), "gelu", "relu", "leaky_relu_sq".
	LeakySlope    float64      `json:"leaky_slope,omitempty"`    // mlp leaky_relu_sq negative slope; defaults to 0.5.
	Weights       []WeightSpec `json:"weights,omitempty"`        // custom block weight declarations
	Ops           []OpSpec     `json:"ops,omitempty"`            // custom block operation sequence
}

// TrainingPhase defines one contiguous training phase with a fixed LR.
type TrainingPhase struct {
	Steps int     `json:"steps"`
	LR    float64 `json:"lr"`
	Label string  `json:"label,omitempty"`
}

// TrainingSpec holds training hyperparameters.
type TrainingSpec struct {
	Steps              int             `json:"steps"`
	LR                 float64         `json:"lr"`
	Phases             []TrainingPhase `json:"phases,omitempty"`
	WarmdownSteps      int             `json:"warmdown_steps,omitempty"`
	TargetValLoss      float64         `json:"target_val_loss,omitempty"`
	TTTSteps           int             `json:"ttt_steps,omitempty"`
	TTTMode            string          `json:"ttt_mode,omitempty"`
	TTTLR              float64         `json:"ttt_lr,omitempty"`
	TTTRank            int             `json:"ttt_rank,omitempty"`
	HardwareTFLOPs     float64         `json:"hardware_tflops,omitempty"` // peak hardware TFLOPS (e.g., 400 for M1 Max, 312 for A100)
	GradClip           float32         `json:"grad_clip"`
	WeightDecay        float32         `json:"weight_decay"`
	Beta1              float32         `json:"beta1"`
	Beta2              float32         `json:"beta2"`
	Epsilon            float32         `json:"epsilon"`
	Seed               int64           `json:"seed"`
	BatchTokens        int             `json:"batch_tokens"`
	ShuffleChunkTokens int             `json:"shuffle_chunk_tokens,omitempty"`
	EmbedLR            float32         `json:"embed_lr,omitempty"`
	MatrixLR           float32         `json:"matrix_lr,omitempty"`
	ScalarLR           float32         `json:"scalar_lr,omitempty"`
	HeadLR             float32         `json:"head_lr,omitempty"`
	MuonMomentum       float32         `json:"muon_momentum,omitempty"`
	MuonBackendSteps   int             `json:"muon_backend_steps,omitempty"`
	// NewtonSchulzVariant controls Muon's Newton-Schulz coefficient choice.
	// "" or "fixed" = canonical (3.4445, -4.7750, 2.0315) per iteration.
	// "polar_express" = per-iteration Chebyshev minimax-optimal tuples
	// (You Jiacheng, arXiv:2505.16932).
	NewtonSchulzVariant string  `json:"newton_schulz_variant,omitempty"`
	MuonNesterov        *bool   `json:"muon_nesterov,omitempty"`
	Optimizer           string  `json:"optimizer,omitempty"` // "muon" (default) or "adamw" for matrix weights
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
		MuonBackendSteps:  5,
		EmbedWeightDecay:  0.01,
		MatrixWeightDecay: 0.01,
		ScalarWeightDecay: 0.01,
		HeadWeightDecay:   0.01,
		SWADecay:          0.999,
		SWAInterval:       10,
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
	return validateConfig(&cfg, source)
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
	if cfg.Dropout < 0 || cfg.Dropout > 1 {
		return nil, fmt.Errorf("config %q has invalid dropout=%g (must be in [0,1])", source, cfg.Dropout)
	}

	for i, b := range cfg.Blocks {
		if err := validateBlockSpec(b, source, "blocks", i); err != nil {
			return nil, err
		}
		if err := validateBlockRopeDims(b, cfg.ModelDim, source, "blocks", i); err != nil {
			return nil, err
		}
	}

	if len(cfg.Blocks) == 0 {
		return nil, fmt.Errorf("config %q must define at least one block", source)
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

	// Apply training defaults.
	d := DefaultTrainingSpec()
	if cfg.Training.Steps <= 0 {
		cfg.Training.Steps = d.Steps
	}
	if cfg.Training.LR <= 0 {
		cfg.Training.LR = d.LR
	}
	// Note: seed=0 in JSON is indistinguishable from omitted; defaults to 42.
	if cfg.Training.Seed <= 0 {
		cfg.Training.Seed = d.Seed
	}
	if cfg.Training.BatchTokens <= 0 {
		cfg.Training.BatchTokens = d.BatchTokens
	}
	if cfg.Training.WeightDecay == 0 {
		cfg.Training.WeightDecay = d.WeightDecay
	}
	if cfg.Training.Beta1 == 0 {
		cfg.Training.Beta1 = d.Beta1
	}
	if cfg.Training.Beta2 == 0 {
		cfg.Training.Beta2 = d.Beta2
	}
	if cfg.Training.Epsilon == 0 {
		cfg.Training.Epsilon = d.Epsilon
	}
	if cfg.Training.EmbedLR == 0 {
		cfg.Training.EmbedLR = float32(cfg.Training.LR)
	}
	if cfg.Training.MatrixLR == 0 {
		cfg.Training.MatrixLR = float32(cfg.Training.LR)
	}
	if cfg.Training.ScalarLR == 0 {
		cfg.Training.ScalarLR = float32(cfg.Training.LR)
	}
	if cfg.Training.HeadLR == 0 {
		cfg.Training.HeadLR = float32(cfg.Training.LR)
	}
	if cfg.Training.TTTLR == 0 {
		cfg.Training.TTTLR = d.TTTLR
	}
	if cfg.Training.TTTMode == "" {
		cfg.Training.TTTMode = d.TTTMode
	}
	cfg.Training.QAT = strings.ToLower(strings.TrimSpace(cfg.Training.QAT))
	if cfg.Training.QAT == "" {
		cfg.Training.QAT = d.QAT
	}
	if cfg.Training.TTTRank == 0 {
		cfg.Training.TTTRank = d.TTTRank
	}
	if cfg.Training.MuonMomentum == 0 {
		cfg.Training.MuonMomentum = cfg.Training.Beta1
	}
	if cfg.Training.MuonBackendSteps <= 0 {
		cfg.Training.MuonBackendSteps = d.MuonBackendSteps
	}
	if cfg.Training.EmbedWeightDecay == 0 {
		cfg.Training.EmbedWeightDecay = cfg.Training.WeightDecay
	}
	if cfg.Training.MatrixWeightDecay == 0 {
		cfg.Training.MatrixWeightDecay = cfg.Training.WeightDecay
	}
	if cfg.Training.ScalarWeightDecay == 0 {
		cfg.Training.ScalarWeightDecay = cfg.Training.WeightDecay
	}
	if cfg.Training.HeadWeightDecay == 0 {
		cfg.Training.HeadWeightDecay = cfg.Training.WeightDecay
	}
	if cfg.Training.SWADecay == 0 {
		cfg.Training.SWADecay = d.SWADecay
	}
	if cfg.Training.SWAInterval <= 0 {
		cfg.Training.SWAInterval = d.SWAInterval
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

	return cfg, nil
}

func validateParallelResidual(cfg *ArchConfig, source string) error {
	if !cfg.ParallelResidual {
		return nil
	}
	if len(cfg.Blocks)%2 != 0 {
		return fmt.Errorf("config %q parallel_residual requires an even number of blocks", source)
	}
	for i := 0; i < len(cfg.Blocks); i += 2 {
		if cfg.Blocks[i].Type != "plain" {
			return fmt.Errorf("config %q parallel_residual requires blocks[%d].type=plain (got %q)", source, i, cfg.Blocks[i].Type)
		}
		if cfg.Blocks[i+1].Type != "swiglu" {
			return fmt.Errorf("config %q parallel_residual requires blocks[%d].type=swiglu (got %q)", source, i+1, cfg.Blocks[i+1].Type)
		}
	}
	if cfg.UNet {
		return fmt.Errorf("config %q cannot enable parallel_residual with unet", source)
	}
	return nil
}

func validateRecurrence(cfg *ArchConfig, source string) error {
	if cfg.Recurrence == nil {
		return nil
	}
	if len(cfg.Recurrence) != len(cfg.Blocks) {
		return fmt.Errorf("config %q recurrence length=%d must match blocks length=%d", source, len(cfg.Recurrence), len(cfg.Blocks))
	}
	for i, ref := range cfg.Recurrence {
		if ref < 0 || ref >= len(cfg.Blocks) {
			return fmt.Errorf("config %q recurrence[%d]=%d out of range [0,%d)", source, i, ref, len(cfg.Blocks))
		}
		if ref > i {
			return fmt.Errorf("config %q recurrence[%d]=%d is a forward reference", source, i, ref)
		}
		if cfg.Blocks[i].Type != cfg.Blocks[ref].Type {
			return fmt.Errorf("config %q recurrence[%d]=%d type mismatch: blocks[%d].type=%q blocks[%d].type=%q", source, i, ref, i, cfg.Blocks[i].Type, ref, cfg.Blocks[ref].Type)
		}
	}
	return nil
}

func validateWeightGroups(cfg *ArchConfig, source string) error {
	type weightGroupInfo struct {
		idx  int
		spec BlockSpec
	}

	groups := make(map[string]weightGroupInfo)
	for i, block := range cfg.Blocks {
		group := strings.TrimSpace(block.WeightGroup)
		cfg.Blocks[i].WeightGroup = group
		if group == "" {
			continue
		}

		prev, ok := groups[group]
		if !ok {
			groups[group] = weightGroupInfo{idx: i, spec: cfg.Blocks[i]}
			continue
		}

		if blockTypeKey(prev.spec) != blockTypeKey(block) {
			return fmt.Errorf("config %q blocks[%d] weight_group=%q type mismatch with blocks[%d] (%q vs %q)", source, i, group, prev.idx, block.Type, prev.spec.Type)
		}

		if prevHeads, ok := weightGroupHeadCount(prev.spec); ok {
			gotHeads, gotOK := weightGroupHeadCount(block)
			if !gotOK || gotHeads != prevHeads {
				return fmt.Errorf("config %q blocks[%d] weight_group=%q heads=%d must match blocks[%d] heads=%d", source, i, group, gotHeads, prev.idx, prevHeads)
			}
		}

		if err := validateWeightGroupLayout(cfg, prev.idx, prev.spec, i, block); err != nil {
			return fmt.Errorf("config %q %w", source, err)
		}
	}

	return nil
}

func weightGroupHeadCount(spec BlockSpec) (int, bool) {
	switch blockTypeKey(spec) {
	case "plain", "retnet", "perceiver", "bottleneck", "cross_attention":
		return spec.Heads, true
	case "custom":
		if spec.Heads <= 0 {
			return 1, true
		}
		return spec.Heads, true
	default:
		return 0, false
	}
}

func validateWeightGroupLayout(cfg *ArchConfig, firstIdx int, first BlockSpec, curIdx int, cur BlockSpec) error {
	firstShapes, err := blockWeightShapes(first, cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, cfg.EffectiveMLPMult(), cfg.BlockScales, cfg.ResidMix)
	if err != nil {
		return fmt.Errorf("blocks[%d] weight_group=%q references invalid weight layout: %w", firstIdx, first.WeightGroup, err)
	}
	curShapes, err := blockWeightShapes(cur, cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, cfg.EffectiveMLPMult(), cfg.BlockScales, cfg.ResidMix)
	if err != nil {
		return fmt.Errorf("blocks[%d] weight_group=%q has invalid weight layout: %w", curIdx, cur.WeightGroup, err)
	}
	if len(firstShapes) != len(curShapes) {
		return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
	}
	for i := range firstShapes {
		if firstShapes[i].Name != curShapes[i].Name {
			return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
		}
		if len(firstShapes[i].Shape) != len(curShapes[i].Shape) {
			return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
		}
		for dim := range firstShapes[i].Shape {
			if firstShapes[i].Shape[dim] != curShapes[i].Shape[dim] {
				return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
			}
		}
	}
	return nil
}

// validateBlockSpec checks that a single block spec has a valid type.
func validateBlockSpec(b BlockSpec, source, groupName string, idx int) error {
	switch b.Type {
	case "plain", "swiglu", "mlp", "mamba", "mamba3", "rwkv", "retnet", "perceiver", "bottleneck", "cross_attention", "token_blend":
		// valid
	case "custom":
		return validateCustomBlockSpec(b, source, groupName, idx)
	default:
		if _, err := lookupBlock(b); err != nil {
			return fmt.Errorf("config %q %s[%d] has invalid type %q (not in registry)", source, groupName, idx, b.Type)
		}
	}
	if b.Type == "plain" && b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] type=plain requires heads > 0", source, groupName, idx)
	}
	if b.Type == "plain" && b.QKGain < 0 {
		return fmt.Errorf("config %q %s[%d] type=plain has invalid qk_gain=%g (must be >= 0)", source, groupName, idx, b.QKGain)
	}
	if b.Type == "plain" && b.KVHeads != 0 {
		if b.KVHeads < 0 {
			return fmt.Errorf("config %q %s[%d] type=plain has invalid kv_heads=%d (must be > 0 when set)", source, groupName, idx, b.KVHeads)
		}
		if b.Heads%b.KVHeads != 0 {
			return fmt.Errorf("config %q %s[%d] type=plain requires heads %% kv_heads == 0 (got heads=%d kv_heads=%d)", source, groupName, idx, b.Heads, b.KVHeads)
		}
	}
	if b.Type == "retnet" && b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] type=retnet requires heads > 0", source, groupName, idx)
	}
	if (b.Type == "perceiver" || b.Type == "bottleneck") && b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] type=%s requires heads > 0", source, groupName, idx, b.Type)
	}
	if b.Type == "cross_attention" {
		if b.Heads <= 0 {
			return fmt.Errorf("config %q %s[%d] type=cross_attention requires heads > 0", source, groupName, idx)
		}
		if b.SourceStream == "" {
			return fmt.Errorf("config %q %s[%d] type=cross_attention requires source_stream", source, groupName, idx)
		}
	}
	if blockTypeKey(b) == "mlp" {
		switch strings.ToLower(strings.TrimSpace(b.Activation)) {
		case "", "silu", "gelu", "relu", "leaky_relu_sq":
			// valid
		default:
			return fmt.Errorf("config %q %s[%d] type=mlp has invalid activation %q", source, groupName, idx, b.Activation)
		}
		if b.LeakySlope < 0 {
			return fmt.Errorf("config %q %s[%d] type=mlp has invalid leaky_slope=%g (must be >= 0)", source, groupName, idx, b.LeakySlope)
		}
	}
	return nil
}

func validateKVSources(cfg *ArchConfig, source string) error {
	for i, b := range cfg.Blocks {
		if blockTypeKey(b) != "plain" || b.KVSource <= 0 {
			continue
		}
		srcIdx := b.KVSource - 1
		if srcIdx < 0 || srcIdx >= len(cfg.Blocks) {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (must reference an earlier block)", source, i, b.KVSource)
		}
		if srcIdx >= i {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (must reference an earlier block)", source, i, b.KVSource)
		}
		src := cfg.Blocks[srcIdx]
		if blockTypeKey(src) != "plain" {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (blocks[%d] is type=%q, want plain)", source, i, b.KVSource, srcIdx, src.Type)
		}

		wantKVHeads, err := normalizePlainKVHeads(b.Heads, b.KVHeads)
		if err != nil {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d: %w", source, i, b.KVSource, err)
		}
		gotKVHeads, err := normalizePlainKVHeads(src.Heads, src.KVHeads)
		if err != nil {
			return fmt.Errorf("config %q blocks[%d] kv_source=%d references invalid source block: %w", source, i, b.KVSource, err)
		}
		if src.Heads != b.Heads {
			return fmt.Errorf("config %q blocks[%d] type=plain has incompatible kv_source=%d (heads=%d, source heads=%d)", source, i, b.KVSource, b.Heads, src.Heads)
		}
		if gotKVHeads != wantKVHeads {
			return fmt.Errorf("config %q blocks[%d] type=plain has incompatible kv_source=%d (kv_heads=%d, source kv_heads=%d)", source, i, b.KVSource, wantKVHeads, gotKVHeads)
		}
	}
	return nil
}

func validateBlockRopeDims(b BlockSpec, modelDim int, source, groupName string, idx int) error {
	if b.RopeDims == 0 {
		return nil
	}
	if b.RopeDims < 0 {
		return fmt.Errorf("config %q %s[%d] has invalid rope_dims=%d (must be > 0 when set)", source, groupName, idx, b.RopeDims)
	}
	if b.RopeDims%2 != 0 {
		return fmt.Errorf("config %q %s[%d] has invalid rope_dims=%d (must be even)", source, groupName, idx, b.RopeDims)
	}
	if b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] has rope_dims=%d but heads must be > 0", source, groupName, idx, b.RopeDims)
	}
	if modelDim%b.Heads != 0 {
		return fmt.Errorf("config %q %s[%d] has rope_dims=%d but model_dim=%d is not divisible by heads=%d", source, groupName, idx, b.RopeDims, modelDim, b.Heads)
	}
	headDim := modelDim / b.Heads
	if b.RopeDims > headDim {
		return fmt.Errorf("config %q %s[%d] has invalid rope_dims=%d (must be <= head_dim=%d)", source, groupName, idx, b.RopeDims, headDim)
	}
	return nil
}

// validateCustomBlockSpec validates a custom block's weights and ops.
func validateCustomBlockSpec(b BlockSpec, source, groupName string, idx int) error {
	if b.Name == "" {
		return fmt.Errorf("config %q %s[%d] type=custom requires a name", source, groupName, idx)
	}
	if len(b.Weights) == 0 {
		return fmt.Errorf("config %q %s[%d] custom block %q must declare at least one weight", source, groupName, idx, b.Name)
	}
	if len(b.Ops) == 0 {
		return fmt.Errorf("config %q %s[%d] custom block %q must declare at least one op", source, groupName, idx, b.Name)
	}
	for wi, w := range b.Weights {
		if w.Name == "" {
			return fmt.Errorf("config %q %s[%d] custom block %q weight[%d] missing name", source, groupName, idx, b.Name, wi)
		}
		if len(w.Shape) == 0 {
			return fmt.Errorf("config %q %s[%d] custom block %q weight %q missing shape", source, groupName, idx, b.Name, w.Name)
		}
	}
	for oi, op := range b.Ops {
		if op.Op == "" {
			return fmt.Errorf("config %q %s[%d] custom block %q op[%d] missing op name", source, groupName, idx, b.Name, oi)
		}
		if op.Output == "" && len(op.Outputs) == 0 {
			return fmt.Errorf("config %q %s[%d] custom block %q op[%d] missing output(s)", source, groupName, idx, b.Name, oi)
		}
	}
	return nil
}
