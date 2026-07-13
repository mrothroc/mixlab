package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
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
	CharVocabSize            int          `json:"char_vocab_size,omitempty"`
	CharDim                  int          `json:"char_dim,omitempty"`
	CharMaxPerToken          int          `json:"char_max_per_token,omitempty"`
	PositionalEmbedding      string       `json:"positional_embedding,omitempty"`
	MaxPositions             int          `json:"max_positions,omitempty"`
	EmbeddingDropout         float32      `json:"embedding_dropout,omitempty"`
	HFExportFormat           string       `json:"hf_export_format,omitempty"`
	BigramVocabSize          int          `json:"bigram_vocab_size,omitempty"`
	BigramDim                int          `json:"bigram_dim,omitempty"`
	TrigramVocabSize         int          `json:"trigram_vocab_size,omitempty"`
	TrigramDim               int          `json:"trigram_dim,omitempty"`
	LogitSoftcap             float32      `json:"logit_softcap,omitempty"`
	Dropout                  float32      `json:"dropout,omitempty"`
	AttnDropout              float32      `json:"attn_dropout,omitempty"`
	HiddenDropout            float32      `json:"hidden_dropout,omitempty"`
	MLMHead                  string       `json:"mlm_head,omitempty"`
	NormType                 string       `json:"norm_type,omitempty"`
	NormEps                  float32      `json:"norm_eps,omitempty"`
	NormAffine               *bool        `json:"norm_affine,omitempty"`
	NormPlacement            string       `json:"norm_placement,omitempty"`
	FFNInternalNorm          bool         `json:"ffn_internal_norm,omitempty"`
	LayerAggregation         string       `json:"layer_aggregation,omitempty"`
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

	SourcePath string `json:"-"`

	recurrencePhasesSet           bool
	executionOrderSet             bool
	recurrencePhaseActivationsSet bool
	attnDropoutSet                bool
	hiddenDropoutSet              bool

	CharFeatureIDs    []int32 `json:"-"`
	CharFeatureSource string  `json:"-"`
}

// DataSpec holds data-loader behavior.
type DataSpec struct {
	NoShardShuffle bool `json:"no_shard_shuffle,omitempty"`
}

// Types and validation helpers for recurrence_phases live in
// arch/config_recurrence_phases.go.

// BlockSpec describes a single model block.
type BlockSpec struct {
	Type                              string       `json:"type"`
	Name                              string       `json:"name,omitempty"` // custom block name (required for type=custom)
	WeightGroup                       string       `json:"weight_group,omitempty"`
	ParallelResidual                  *bool        `json:"parallel_residual,omitempty"` // enable/disable parallel residual for this block pair start
	ParallelGroup                     int          `json:"parallel_group,omitempty"`    // explicit heterogeneous parallel group length when set on the first block.
	ResidualScaleInit                 *float64     `json:"residual_scale_init,omitempty"`
	NumExperts                        int          `json:"num_experts,omitempty"`  // moe: number of routed FFN experts.
	TopK                              int          `json:"top_k,omitempty"`        // moe: number of experts selected per token; defaults to min(2,num_experts).
	ExpertBlock                       *BlockSpec   `json:"expert_block,omitempty"` // moe: FFN expert block spec (swiglu/geglu/mlp).
	Router                            string       `json:"router,omitempty"`       // moe: router type; v1 supports "linear".
	LoadBalanceLossWeight             float64      `json:"load_balance_loss_weight,omitempty"`
	Heads                             int          `json:"heads"`
	KVHeads                           int          `json:"kv_heads,omitempty"`
	KVSource                          int          `json:"kv_source,omitempty"`                           // -1 or 0 = compute own KV; positive = reuse KV from block N (1-indexed)
	RopeDims                          int          `json:"rope_dims,omitempty"`                           // RoPE rotation dims per head; 0 or head_dim = full RoPE
	RopeConvention                    string       `json:"rope_convention,omitempty"`                     // plain: "", "adjacent_pair", or "half_rotation".
	RelativeAttention                 string       `json:"relative_attention,omitempty"`                  // plain: "", "none", or "deberta_p2c_c2p".
	RelativeAttentionWindow           int          `json:"relative_attention_window,omitempty"`           // plain relative attention bucket size; defaults to 128.
	RelativeAttentionParameterization string       `json:"relative_attention_parameterization,omitempty"` // plain relative attention layout: per_block_projections or shared_qk_reuse.
	RelativeAttentionEmbeddingNorm    string       `json:"relative_attention_embedding_norm,omitempty"`   // plain shared_qk_reuse: "", "none", or "layernorm" for affine shared relative embedding LayerNorm.
	QKGain                            float64      `json:"qk_gain,omitempty"`                             // per-head learnable QK scaling; 0 disables
	QKNorm                            bool         `json:"qk_norm,omitempty"`                             // plain: learned RMSNorm on Q/K heads before attention scores.
	DifferentialAttention             bool         `json:"differential_attention,omitempty"`              // plain: DIFF Transformer two-softmax differential attention.
	DifferentialLambdaInit            *float64     `json:"differential_lambda_init,omitempty"`            // plain differential attention: optional lambda_init override.
	AttnBias                          bool         `json:"attn_bias,omitempty"`                           // plain: add learned biases to Q/K/V/O projections.
	AttnValueGate                     bool         `json:"attn_value_gate,omitempty"`                     // plain: widen V projection to value+GELU gate and gate the attention output before WO.
	AttnPostNorm                      string       `json:"attn_post_norm,omitempty"`                      // plain: "", "inherit", "none", "after_outproj", or "before_outproj".
	XSA                               bool         `json:"xsa,omitempty"`                                 // enable V-orthogonal projection after attention
	WindowSize                        int          `json:"window_size,omitempty"`                         // plain: sliding causal attention width; 0 = full causal attention
	AttentionMask                     string       `json:"attention_mask,omitempty"`                      // plain: "causal", "bidirectional", or "none"; empty resolves from training objective.
	SkipAttention                     bool         `json:"skip_attention,omitempty"`                      // plain: bypass attention while preserving weight layout.
	SparseAttnGate                    bool         `json:"sparse_attn_gate,omitempty"`                    // plain: narrow per-head output gate over the first gate_window head channels.
	FFNActivation                     string       `json:"ffn_activation,omitempty"`                      // plain: "silu" (default), "geglu", or "swiglu" feed-forward tail.
	FFNPreNorm                        bool         `json:"ffn_pre_norm,omitempty"`                        // plain: add a second pre-FFN norm after attention residual.
	FFNBias                           bool         `json:"ffn_bias,omitempty"`                            // plain: add learned biases to FFN up/down projections.
	InnerDim                          int          `json:"inner_dim,omitempty"`                           // Mamba inner dimension; defaults to model_dim.
	ChunkSize                         int          `json:"chunk_size,omitempty"`                          // ttt_mlp: inner-update mini-batch width; defaults to 16.
	InnerHiddenMult                   float64      `json:"inner_hidden_mult,omitempty"`                   // ttt_mlp: inner MLP expansion; defaults to 4.
	InnerLRBase                       float64      `json:"inner_lr_base,omitempty"`                       // ttt_mlp: learned inner-LR ceiling; defaults to 0.1.
	InnerLRInit                       float64      `json:"inner_lr_init,omitempty"`                       // ttt_mlp: inner base LR at outer step zero before per-token scaling; defaults to 0.01.
	InnerLRWarmupSteps                *int         `json:"inner_lr_warmup_steps,omitempty"`               // ttt_mlp: linear inner-LR warmup; defaults to 5000; explicit zero disables it.
	DK                                int          `json:"d_k,omitempty"`                                 // gated_deltanet: key/query dim per head.
	DV                                int          `json:"d_v,omitempty"`                                 // gated_deltanet: value dim per head; defaults to 2*d_k.
	DState                            int          `json:"d_state,omitempty"`                             // hgrn2: matrix-state key/query dim per head; defaults to model_dim/heads.
	KVShare                           *bool        `json:"kv_share,omitempty"`                            // gated_deltanet: share the K/V projection when true (default).
	ScanChunkSize                     *int         `json:"scan_chunk_size,omitempty"`                     // gated_deltanet/mamba3-canonical: chunk size for chunked scan; 0 keeps the naive/full scan.
	StateSize                         int          `json:"state_size,omitempty"`                          // Mamba-3 canonical state expansion; defaults to 16.
	NGroups                           int          `json:"n_groups,omitempty"`                            // Mamba-3 canonical grouped/MIMO axis; defaults to 4.
	DTRank                            int          `json:"dt_rank,omitempty"`                             // Mamba-3 canonical low-rank dt/lambda/theta rank; defaults to max(inner/16,1).
	ConvKernel                        int          `json:"conv_kernel,omitempty"`                         // Mamba-3 canonical causal conv width; defaults to 4.
	UseConv                           *bool        `json:"use_conv,omitempty"`                            // Mamba-3 canonical short conv toggle; defaults to true.
	DTMin                             float64      `json:"dt_min,omitempty"`                              // Mamba-3 canonical dt init lower bound; defaults to 0.001.
	DTMax                             float64      `json:"dt_max,omitempty"`                              // Mamba-3 canonical dt init upper bound; defaults to 0.1.
	NumLatents                        int          `json:"num_latents,omitempty"`                         // Perceiver/bottleneck latent count.
	SourceStream                      string       `json:"source_stream,omitempty"`                       // cross_attention: stream providing K/V.
	Decay                             float64      `json:"decay,omitempty"`                               // RetNet: initial decay rate in (0,1); defaults to 0.95.
	Activation                        string       `json:"activation,omitempty"`                          // mlp: "silu" (default), "gelu", "relu", "leaky_relu_sq".
	LeakySlope                        float64      `json:"leaky_slope,omitempty"`                         // mlp leaky_relu_sq negative slope; defaults to 0.5.
	Weights                           []WeightSpec `json:"weights,omitempty"`                             // custom block weight declarations
	Ops                               []OpSpec     `json:"ops,omitempty"`                                 // custom block operation sequence

	loadBalanceLossWeightSet bool
	chunkSizeSet             bool
	innerHiddenMultSet       bool
	innerLRBaseSet           bool
	innerLRInitSet           bool
}

// UnmarshalJSON records MoE scalar field presence so an explicit
// load_balance_loss_weight: 0 can disable the auxiliary loss while an omitted
// field still receives the default.
func (b *BlockSpec) UnmarshalJSON(data []byte) error {
	type Alias BlockSpec
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var alias Alias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	*b = BlockSpec(alias)
	_, b.loadBalanceLossWeightSet = raw["load_balance_loss_weight"]
	_, b.chunkSizeSet = raw["chunk_size"]
	_, b.innerHiddenMultSet = raw["inner_hidden_mult"]
	_, b.innerLRBaseSet = raw["inner_lr_base"]
	_, b.innerLRInitSet = raw["inner_lr_init"]
	return nil
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
	Temperature        float64  `json:"temperature,omitempty"`

	temperatureSet bool `json:"-"`
}

// Data2VecSpec configures optional online EMA representation distillation.
type Data2VecSpec struct {
	LossWeight      float64 `json:"loss_weight,omitempty"`
	EMATau          float64 `json:"ema_tau,omitempty"`
	EMATauStart     float64 `json:"ema_tau_start,omitempty"`
	EMATauEnd       float64 `json:"ema_tau_end,omitempty"`
	EMATauRampSteps int     `json:"ema_tau_ramp_steps,omitempty"`
	TopKLayers      int     `json:"top_k_layers,omitempty"`
	SmoothL1Beta    float64 `json:"smooth_l1_beta,omitempty"`
	TargetNorm      string  `json:"target_norm,omitempty"`
	TargetNormEps   float64 `json:"target_norm_eps,omitempty"`
	MaskSource      string  `json:"mask_source,omitempty"`
	MaskProb        float64 `json:"mask_prob,omitempty"`
	PredictorHidden int     `json:"predictor_hidden_dim,omitempty"`

	lossWeightSet    bool
	emaTauSet        bool
	emaTauStartSet   bool
	emaTauEndSet     bool
	topKLayersSet    bool
	smoothL1BetaSet  bool
	targetNormEpsSet bool
}

// DiffusionSpec and its block-diffusion validation live in diffusion.go.

// ExampleFramingSpec configures loader-side fixed example framing for raw
// continuous token streams.
type ExampleFramingSpec struct {
	ContentLen int `json:"content_len"`
	BosID      int `json:"bos_id"`
	EosID      int `json:"eos_id"`

	contentLenSet bool
	bosIDSet      bool
	eosIDSet      bool
}

// UnmarshalJSON records token-id field presence so token id 0 can be used
// intentionally instead of being confused with an omitted field.
func (e *ExampleFramingSpec) UnmarshalJSON(data []byte) error {
	type Alias ExampleFramingSpec
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var alias Alias
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&alias); err != nil {
		return err
	}
	*e = ExampleFramingSpec(alias)
	_, e.contentLenSet = raw["content_len"]
	_, e.bosIDSet = raw["bos_id"]
	_, e.eosIDSet = raw["eos_id"]
	return nil
}

// TrainingSpec holds training hyperparameters.
type TrainingSpec struct {
	Steps                             int                          `json:"steps"`
	LR                                float64                      `json:"lr"`
	Phases                            []TrainingPhase              `json:"phases,omitempty"`
	Objective                         string                       `json:"objective,omitempty"`
	ExportHead                        string                       `json:"export_head,omitempty"`
	DiffusionHead                     string                       `json:"diffusion_head,omitempty"`
	Heads                             []MultiheadHeadSpec          `json:"heads,omitempty"`
	Diffusion                         *DiffusionSpec               `json:"diffusion,omitempty"`
	MLMMaskProb                       float64                      `json:"mlm_mask_prob,omitempty"`
	MLMMaskTokenID                    int                          `json:"mlm_mask_token_id,omitempty"`
	MLMMaskTokenProb                  float64                      `json:"mlm_mask_token_prob,omitempty"`
	MLMRandomTokenProb                float64                      `json:"mlm_random_token_prob,omitempty"`
	MLMKeptUnchangedProb              float64                      `json:"mlm_kept_unchanged_prob,omitempty"`
	MLMMaskProbSchedule               [][]float64                  `json:"mlm_mask_prob_schedule,omitempty"`
	MLMMaskProbScheduleMode           string                       `json:"mlm_mask_prob_schedule_mode,omitempty"`
	HybridCLMFraction                 float64                      `json:"hybrid_clm_fraction,omitempty"`
	HybridCLMFractionSchedule         [][]float64                  `json:"hybrid_clm_fraction_schedule,omitempty"`
	HybridCLMFractionScheduleMode     string                       `json:"hybrid_clm_fraction_schedule_mode,omitempty"`
	HybridSecondaryObjective          string                       `json:"hybrid_secondary_objective,omitempty"`
	HybridMixGranularity              string                       `json:"hybrid_mix_granularity,omitempty"`
	AttentionSegmentMask              string                       `json:"attention_segment_mask,omitempty"`
	AttentionSegmentBoundaryTokenID   int                          `json:"attention_segment_boundary_token_id,omitempty"`
	ExampleFraming                    *ExampleFramingSpec          `json:"example_framing,omitempty"`
	RTD                               *RTDSpec                     `json:"rtd,omitempty"`
	MinimalPair                       *MinimalPairSpec             `json:"minimal_pair,omitempty"`
	Invariance                        *InvarianceSpec              `json:"invariance,omitempty"`
	PLLMargin                         *PLLMarginSpec               `json:"pll_margin,omitempty"`
	WordStructuralObjective           *WordStructuralObjectiveSpec `json:"word_structural_objective,omitempty"`
	Distillation                      *DistillationSpec            `json:"distillation,omitempty"`
	Data2Vec                          *Data2VecSpec                `json:"data2vec,omitempty"`
	ZLoss                             float64                      `json:"z_loss,omitempty"`
	SeqLenSchedule                    [][]int                      `json:"seq_len_schedule,omitempty"`
	WarmupSteps                       int                          `json:"warmup_steps,omitempty"`
	WarmupRatio                       float64                      `json:"warmup_ratio,omitempty"`
	HoldSteps                         int                          `json:"hold_steps,omitempty"`
	WarmdownSteps                     int                          `json:"warmdown_steps,omitempty"`
	TargetValLoss                     float64                      `json:"target_val_loss,omitempty"`
	EarlyStop                         *EarlyStopSpec               `json:"early_stop,omitempty"`
	FirstByteMask                     bool                         `json:"first_byte_mask,omitempty"`
	FirstByteMaskValid                []int32                      `json:"-"`
	RecurrenceActivationFrac          float64                      `json:"recurrence_activation_frac,omitempty"`
	RecurrenceActivationStep          int                          `json:"recurrence_activation_step,omitempty"`
	TTTSteps                          int                          `json:"ttt_steps,omitempty"`
	TTTMode                           string                       `json:"ttt_mode,omitempty"`
	TTTLR                             float64                      `json:"ttt_lr,omitempty"`
	TTTRank                           int                          `json:"ttt_rank,omitempty"`
	HardwareTFLOPs                    float64                      `json:"hardware_tflops,omitempty"` // peak hardware TFLOPS (e.g., 400 for M1 Max, 312 for A100)
	GradClip                          float32                      `json:"grad_clip"`
	WeightDecay                       float32                      `json:"weight_decay"`
	CautiousWeightDecay               bool                         `json:"cautious_weight_decay,omitempty"`
	CautiousWeightDecayActivationFrac float64                      `json:"cautious_weight_decay_activation_frac,omitempty"`
	Beta1                             float32                      `json:"beta1"`
	Beta2                             float32                      `json:"beta2"`
	Epsilon                           float32                      `json:"epsilon"`
	LAMBBeta1                         float32                      `json:"lamb_beta1,omitempty"`
	LAMBBeta2                         float32                      `json:"lamb_beta2,omitempty"`
	LAMBEps                           float32                      `json:"lamb_eps,omitempty"`
	LAMBTrustRatioCap                 float32                      `json:"lamb_trust_ratio_cap"`
	Seed                              int64                        `json:"seed"`
	BatchTokens                       int                          `json:"batch_tokens"`
	ShuffleChunkTokens                int                          `json:"shuffle_chunk_tokens,omitempty"`
	EmbedLR                           float32                      `json:"embed_lr,omitempty"`
	MatrixLR                          float32                      `json:"matrix_lr,omitempty"`
	ScalarLR                          float32                      `json:"scalar_lr,omitempty"`
	HeadLR                            float32                      `json:"head_lr,omitempty"`
	MuonMomentum                      float32                      `json:"muon_momentum,omitempty"`
	MuonBackendSteps                  int                          `json:"muon_backend_steps,omitempty"`
	// NewtonSchulzVariant controls Muon's Newton-Schulz coefficient choice.
	// "" or "fixed" = canonical (3.4445, -4.7750, 2.0315) per iteration.
	// "polar_express" = per-iteration Chebyshev minimax-optimal tuples
	// (You Jiacheng, arXiv:2505.16932).
	NewtonSchulzVariant string  `json:"newton_schulz_variant,omitempty"`
	MuonNesterov        *bool   `json:"muon_nesterov,omitempty"`
	Optimizer           string  `json:"optimizer,omitempty"` // "muon" (default), "muon_eq_r", "normuon", "adamw", or "lamb"
	ComputeDType        string  `json:"compute_dtype,omitempty"`
	QAT                 string  `json:"qat,omitempty"` // "none" (default), "int8", or "int6"
	QATStart            int     `json:"qat_start,omitempty"`
	WeightInit          string  `json:"weight_init,omitempty"`     // "xavier_uniform" (default), "normal", "gptbert", or "gpt2"
	WeightInitStd       float32 `json:"weight_init_std,omitempty"` // std for normal init (default 0.02)
	EmbedWeightDecay    float32 `json:"embed_weight_decay"`
	MatrixWeightDecay   float32 `json:"matrix_weight_decay"`
	ScalarWeightDecay   float32 `json:"scalar_weight_decay"`
	HeadWeightDecay     float32 `json:"head_weight_decay"`
	// MinLRFraction sets the minimum LR as a fraction of peak LR.
	// 0 (default) = current behavior (warmdown ends near 0).
	// 0.10 = recommended (LR never drops below 10% of peak).
	// Used as an absolute floor across both cosine decay and warmdown phases.
	MinLRFraction float32 `json:"min_lr_fraction,omitempty"`
	SWAStart      int     `json:"swa_start,omitempty"`
	SWADecay      float32 `json:"swa_decay,omitempty"`
	SWAInterval   int     `json:"swa_interval,omitempty"`

	mlmMaskProbSet                     bool
	mlmMaskTokenIDSet                  bool
	mlmReplacementProbSet              bool
	hybridCLMFractionSet               bool
	attentionSegmentBoundaryTokenIDSet bool
	warmupStepsSet                     bool
	warmupRatioSet                     bool
	holdStepsSet                       bool
	weightDecaySet                     bool
	embedWeightDecaySet                bool
	matrixWeightDecaySet               bool
	scalarWeightDecaySet               bool
	headWeightDecaySet                 bool
	lambBeta1Set                       bool
	lambBeta2Set                       bool
	lambEpsSet                         bool
	lambTrustRatioCapSet               bool
	swaDecaySet                        bool
	swaIntervalSet                     bool
}

// EarlyStopSpec controls optional validation-loss early stopping beyond the
// legacy target_val_loss threshold.
type EarlyStopSpec struct {
	Metric   string  `json:"metric,omitempty"`
	Patience int     `json:"patience,omitempty"`
	MinDelta float64 `json:"min_delta,omitempty"`
	MinSteps int     `json:"min_steps,omitempty"`
	ValGT    float64 `json:"val_gt,omitempty"`
	AtStep   int     `json:"at_step,omitempty"`
}

// WarmupStepsConfigured reports whether training.warmup_steps was provided.
func (t TrainingSpec) WarmupStepsConfigured() bool {
	return t.warmupStepsSet || t.WarmupSteps != 0
}

// WarmupRatioConfigured reports whether training.warmup_ratio was provided.
func (t TrainingSpec) WarmupRatioConfigured() bool {
	return t.warmupRatioSet || t.WarmupRatio != 0
}

// HoldStepsConfigured reports whether training.hold_steps was provided.
func (t TrainingSpec) HoldStepsConfigured() bool {
	return t.holdStepsSet || t.HoldSteps != 0
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

// EffectiveComputeDType returns the training compute dtype, defaulting to fp32.
func (t TrainingSpec) EffectiveComputeDType() string {
	dtype := strings.ToLower(strings.TrimSpace(t.ComputeDType))
	if dtype == "" {
		return "float32"
	}
	return dtype
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
		LAMBBeta1:         0.9,
		LAMBBeta2:         0.999,
		LAMBEps:           1e-6,
		LAMBTrustRatioCap: 10,
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
	t.HybridMixGranularity = t.EffectiveHybridMixGranularity()
	t.AttentionSegmentMask = t.EffectiveAttentionSegmentMask()
	if t.Data2Vec != nil {
		t.Data2Vec.applyDefaults()
	}
	if t.RTD != nil {
		t.RTD.applyDefaults(t.MLMMaskProb)
	}
	if t.MinimalPair != nil {
		t.MinimalPair.applyDefaults()
	}
	if t.Invariance != nil {
		t.Invariance.applyDefaults()
	}
	if t.PLLMargin != nil {
		t.PLLMargin.applyDefaults()
	}
	if t.WordStructuralObjective != nil {
		t.WordStructuralObjective.applyDefaults(t.MLMMaskTokenID)
	}
	if !t.weightDecaySet && t.WeightDecay == 0 {
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
	if !t.lambBeta1Set && t.LAMBBeta1 == 0 {
		t.LAMBBeta1 = d.LAMBBeta1
	}
	if !t.lambBeta2Set && t.LAMBBeta2 == 0 {
		t.LAMBBeta2 = d.LAMBBeta2
	}
	if !t.lambEpsSet && t.LAMBEps == 0 {
		t.LAMBEps = d.LAMBEps
	}
	if !t.lambTrustRatioCapSet && t.LAMBTrustRatioCap == 0 {
		t.LAMBTrustRatioCap = d.LAMBTrustRatioCap
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
	t.ComputeDType = strings.ToLower(strings.TrimSpace(t.ComputeDType))
	t.WeightInit = strings.ToLower(strings.TrimSpace(t.WeightInit))
	if !t.embedWeightDecaySet && t.EmbedWeightDecay == 0 {
		t.EmbedWeightDecay = t.WeightDecay
	}
	if !t.matrixWeightDecaySet && t.MatrixWeightDecay == 0 {
		t.MatrixWeightDecay = t.WeightDecay
	}
	if !t.scalarWeightDecaySet && t.ScalarWeightDecay == 0 {
		t.ScalarWeightDecay = t.WeightDecay
	}
	if !t.headWeightDecaySet && t.HeadWeightDecay == 0 {
		t.HeadWeightDecay = t.WeightDecay
	}
	if !t.swaDecaySet && t.SWADecay == 0 {
		t.SWADecay = d.SWADecay
	}
	if !t.swaIntervalSet && t.SWAInterval <= 0 {
		t.SWAInterval = d.SWAInterval
	}
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
	if err := validatePositionalAndExportConfig(cfg, source); err != nil {
		return nil, err
	}
	if cfg.CharVocabSize < 0 {
		return nil, fmt.Errorf("config %q has invalid char_vocab_size=%d (must be >= 0)", source, cfg.CharVocabSize)
	}
	if cfg.CharDim < 0 {
		return nil, fmt.Errorf("config %q has invalid char_dim=%d (must be >= 0)", source, cfg.CharDim)
	}
	if cfg.CharMaxPerToken < 0 {
		return nil, fmt.Errorf("config %q has invalid char_max_per_token=%d (must be >= 0)", source, cfg.CharMaxPerToken)
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
	if cfg.CharVocabSize > 0 {
		if cfg.CharVocabSize < 257 {
			return nil, fmt.Errorf("config %q has invalid char_vocab_size=%d (must be 0 to disable or >= 257 when enabled)", source, cfg.CharVocabSize)
		}
		if cfg.CharDim == 0 {
			cfg.CharDim = cfg.ModelDim
		}
		if cfg.CharMaxPerToken == 0 {
			cfg.CharMaxPerToken = 16
		}
		if cfg.CharMaxPerToken <= 0 {
			return nil, fmt.Errorf("config %q has invalid char_max_per_token=%d (must be > 0 when char features are enabled)", source, cfg.CharMaxPerToken)
		}
	} else {
		cfg.CharDim = 0
		cfg.CharMaxPerToken = 0
		cfg.CharFeatureIDs = nil
		cfg.CharFeatureSource = ""
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
	if cfg.AttnDropout < 0 || cfg.AttnDropout > 1 {
		return nil, fmt.Errorf("config %q has invalid attn_dropout=%g (must be in [0,1])", source, cfg.AttnDropout)
	}
	if cfg.HiddenDropout < 0 || cfg.HiddenDropout > 1 {
		return nil, fmt.Errorf("config %q has invalid hidden_dropout=%g (must be in [0,1])", source, cfg.HiddenDropout)
	}
	cfg.NormType = normalizeNormType(cfg.NormType)
	switch cfg.NormType {
	case NormTypeRMSNorm, NormTypeLayerNorm:
	default:
		return nil, fmt.Errorf("config %q has invalid norm_type=%q (must be \"rmsnorm\" or \"layernorm\")", source, cfg.NormType)
	}
	if cfg.NormEps < 0 {
		return nil, fmt.Errorf("config %q has invalid norm_eps=%g (must be > 0)", source, cfg.NormEps)
	}
	if cfg.NormEps == 0 {
		cfg.NormEps = 1e-5
	}
	if cfg.NormAffine != nil && !*cfg.NormAffine && cfg.NormType == NormTypeRMSNorm {
		return nil, fmt.Errorf("config %q norm_affine=false requires norm_type=\"layernorm\" in this release", source)
	}
	cfg.NormPlacement = normalizeNormPlacement(cfg.NormPlacement)
	switch cfg.NormPlacement {
	case NormPlacementPre, NormPlacementPost, NormPlacementSandwich:
	default:
		return nil, fmt.Errorf("config %q has invalid norm_placement=%q (must be \"pre\", \"post\", or \"sandwich\")", source, cfg.NormPlacement)
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
		if err := validateBlockPositionalEmbedding(cfg, b, source, "blocks", i); err != nil {
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
	if err := validateSharedRelativeAttention(cfg, source); err != nil {
		return nil, err
	}
	if err := validateDifferentialAttention(cfg, source); err != nil {
		return nil, err
	}
	if err := validateParallelResidual(cfg, source); err != nil {
		return nil, err
	}
	if err := validateNormPolicy(cfg, source); err != nil {
		return nil, err
	}
	if err := validateLayerAggregation(cfg, source); err != nil {
		return nil, err
	}

	cfg.Training.ApplyDefaults()
	if err := validateTTTMLPPolicy(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingRecipeKnobs(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingObjective(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingAttentionSegmentMask(cfg, source); err != nil {
		return nil, err
	}
	if err := validateMLMHead(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingDistillation(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingData2Vec(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingExampleFraming(cfg, source); err != nil {
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
	case "", "adamw", "muon", "muon_eq_r", "normuon", "lamb":
	default:
		return nil, fmt.Errorf("config %q has invalid training.optimizer=%q (must be \"adamw\", \"muon\", \"muon_eq_r\", \"normuon\", or \"lamb\")", source, cfg.Training.Optimizer)
	}
	switch cfg.Training.WeightInit {
	case "", "xavier_uniform", "normal", "gptbert", "gpt2":
	default:
		return nil, fmt.Errorf("config %q has invalid training.weight_init=%q (must be \"xavier_uniform\", \"normal\", \"gptbert\", or \"gpt2\")", source, cfg.Training.WeightInit)
	}
	switch cfg.Training.EffectiveComputeDType() {
	case "float32", "bf16":
	default:
		return nil, fmt.Errorf("config %q has invalid training.compute_dtype=%q (must be \"float32\" or \"bf16\")", source, cfg.Training.ComputeDType)
	}
	if cfg.Training.LAMBBeta1 < 0 || cfg.Training.LAMBBeta1 >= 1 {
		return nil, fmt.Errorf("config %q has invalid training.lamb_beta1=%g (must be in [0,1))", source, cfg.Training.LAMBBeta1)
	}
	if cfg.Training.LAMBBeta2 < 0 || cfg.Training.LAMBBeta2 >= 1 {
		return nil, fmt.Errorf("config %q has invalid training.lamb_beta2=%g (must be in [0,1))", source, cfg.Training.LAMBBeta2)
	}
	if cfg.Training.LAMBEps <= 0 {
		return nil, fmt.Errorf("config %q has invalid training.lamb_eps=%g (must be > 0)", source, cfg.Training.LAMBEps)
	}
	if cfg.Training.LAMBTrustRatioCap < 0 || math.IsNaN(float64(cfg.Training.LAMBTrustRatioCap)) || math.IsInf(float64(cfg.Training.LAMBTrustRatioCap), 0) {
		return nil, fmt.Errorf("config %q has invalid training.lamb_trust_ratio_cap=%g (must be finite and >= 0; 0 disables capping)", source, cfg.Training.LAMBTrustRatioCap)
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
	if cfg.Training.SWAInterval <= 0 {
		return nil, fmt.Errorf("config %q has invalid training.swa_interval=%d (must be > 0)", source, cfg.Training.SWAInterval)
	}
	if cfg.Training.WarmupSteps < 0 {
		return nil, fmt.Errorf("config %q has invalid training.warmup_steps=%d (must be >= 0)", source, cfg.Training.WarmupSteps)
	}
	if cfg.Training.WarmupRatio < 0 || cfg.Training.WarmupRatio > 1 {
		return nil, fmt.Errorf("config %q has invalid training.warmup_ratio=%g (must be in [0,1])", source, cfg.Training.WarmupRatio)
	}
	if cfg.Training.WarmupStepsConfigured() && cfg.Training.WarmupRatioConfigured() {
		return nil, fmt.Errorf("config %q cannot set both training.warmup_steps and training.warmup_ratio", source)
	}
	if cfg.Training.HoldSteps < 0 {
		return nil, fmt.Errorf("config %q has invalid training.hold_steps=%d (must be >= 0)", source, cfg.Training.HoldSteps)
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
