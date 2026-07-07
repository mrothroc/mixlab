package arch

import (
	"encoding/json"
	"fmt"
	"strings"
)

// DiffusionSpec configures block-diffusion corruption and sampling knobs.
type DiffusionSpec struct {
	BlockSize            int     `json:"block_size,omitempty"`
	StepsPerBlock        int     `json:"steps_per_block,omitempty"`
	MinMaskFraction      float64 `json:"min_mask_fraction,omitempty"`
	MaxMaskFraction      float64 `json:"max_mask_fraction,omitempty"`
	ConfidenceThreshold  float64 `json:"confidence_threshold,omitempty"`
	CommitFloor          int     `json:"commit_floor,omitempty"`
	TimestepConditioning string  `json:"timestep_conditioning,omitempty"`
	TimestepConditionDim int     `json:"timestep_conditioning_dim,omitempty"`

	blockSizeSet            bool
	stepsPerBlockSet        bool
	minMaskFractionSet      bool
	maxMaskFractionSet      bool
	confidenceThresholdSet  bool
	commitFloorSet          bool
	timestepConditionDimSet bool
}

func (d *DiffusionSpec) UnmarshalJSON(data []byte) error {
	type alias DiffusionSpec
	var raw alias
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*d = DiffusionSpec(raw)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	_, d.blockSizeSet = fields["block_size"]
	_, d.stepsPerBlockSet = fields["steps_per_block"]
	_, d.minMaskFractionSet = fields["min_mask_fraction"]
	_, d.maxMaskFractionSet = fields["max_mask_fraction"]
	_, d.confidenceThresholdSet = fields["confidence_threshold"]
	_, d.commitFloorSet = fields["commit_floor"]
	_, d.timestepConditionDimSet = fields["timestep_conditioning_dim"]
	return nil
}

func defaultDiffusionBlockSize(seqLen int) int {
	if seqLen <= 0 {
		return 0
	}
	if seqLen >= 16 && seqLen%16 == 0 {
		return 16
	}
	return seqLen
}

func (d *DiffusionSpec) applyDefaults(seqLen int) {
	if d == nil {
		return
	}
	if !d.blockSizeSet && d.BlockSize == 0 {
		d.BlockSize = defaultDiffusionBlockSize(seqLen)
	}
	if !d.stepsPerBlockSet && d.StepsPerBlock == 0 {
		d.StepsPerBlock = d.BlockSize
	}
	if !d.minMaskFractionSet && d.MinMaskFraction == 0 {
		d.MinMaskFraction = 0.05
	}
	if !d.maxMaskFractionSet && d.MaxMaskFraction == 0 {
		d.MaxMaskFraction = 1.0
	}
	if !d.confidenceThresholdSet && d.ConfidenceThreshold == 0 {
		d.ConfidenceThreshold = 0.8
	}
	if !d.commitFloorSet && d.CommitFloor == 0 {
		d.CommitFloor = 1
	}
	d.TimestepConditioning = normalizeDiffusionTimestepConditioning(d.TimestepConditioning)
	if d.TimestepConditioning == DiffusionTimestepConditioningAdaLN && !d.timestepConditionDimSet && d.TimestepConditionDim == 0 {
		d.TimestepConditionDim = 128
	}
}

const (
	DiffusionTimestepConditioningNone  = "none"
	DiffusionTimestepConditioningAdaLN = "adaln"
)

func normalizeDiffusionTimestepConditioning(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "none", "off", "disabled", "false":
		return DiffusionTimestepConditioningNone
	case "adaln", "ada_ln", "ada-layernorm", "ada_layernorm":
		return DiffusionTimestepConditioningAdaLN
	default:
		return strings.ToLower(strings.TrimSpace(raw))
	}
}

func validateBlockDiffusionObjective(cfg *ArchConfig, source string) error {
	t := &cfg.Training
	if t.Diffusion == nil {
		t.Diffusion = &DiffusionSpec{}
	}
	t.Diffusion.applyDefaults(cfg.SeqLen)
	d := t.Diffusion

	if d.BlockSize <= 0 {
		return fmt.Errorf("config %q has invalid training.diffusion.block_size=%d (must be > 0)", source, d.BlockSize)
	}
	if d.BlockSize > cfg.SeqLen {
		return fmt.Errorf("config %q has invalid training.diffusion.block_size=%d (must be <= seq_len=%d)", source, d.BlockSize, cfg.SeqLen)
	}
	if cfg.SeqLen%d.BlockSize != 0 {
		return fmt.Errorf("config %q has invalid training.diffusion.block_size=%d (must divide seq_len=%d in v1)", source, d.BlockSize, cfg.SeqLen)
	}
	if d.StepsPerBlock <= 0 {
		return fmt.Errorf("config %q has invalid training.diffusion.steps_per_block=%d (must be > 0)", source, d.StepsPerBlock)
	}
	if !finiteInClosedRange(d.MinMaskFraction, 0, 1) {
		return fmt.Errorf("config %q has invalid training.diffusion.min_mask_fraction=%g (must be in [0,1])", source, d.MinMaskFraction)
	}
	if !finiteInClosedRange(d.MaxMaskFraction, 0, 1) || d.MaxMaskFraction <= 0 {
		return fmt.Errorf("config %q has invalid training.diffusion.max_mask_fraction=%g (must be in (0,1])", source, d.MaxMaskFraction)
	}
	if d.MinMaskFraction > d.MaxMaskFraction {
		return fmt.Errorf("config %q has invalid training.diffusion mask fraction range [%g,%g] (min_mask_fraction must be <= max_mask_fraction)", source, d.MinMaskFraction, d.MaxMaskFraction)
	}
	if !finiteInClosedRange(d.ConfidenceThreshold, 0, 1) {
		return fmt.Errorf("config %q has invalid training.diffusion.confidence_threshold=%g (must be in [0,1])", source, d.ConfidenceThreshold)
	}
	if d.CommitFloor <= 0 || d.CommitFloor > d.BlockSize {
		return fmt.Errorf("config %q has invalid training.diffusion.commit_floor=%d (must be in [1,block_size=%d])", source, d.CommitFloor, d.BlockSize)
	}
	switch d.TimestepConditioning {
	case DiffusionTimestepConditioningNone, DiffusionTimestepConditioningAdaLN:
	default:
		return fmt.Errorf("config %q has invalid training.diffusion.timestep_conditioning=%q (must be \"none\" or \"adaln\")", source, d.TimestepConditioning)
	}
	if d.TimestepConditioning == DiffusionTimestepConditioningAdaLN && d.TimestepConditionDim <= 0 {
		return fmt.Errorf("config %q has invalid training.diffusion.timestep_conditioning_dim=%d (must be > 0 for adaln)", source, d.TimestepConditionDim)
	}
	if t.DistillationKLEffectiveActive() {
		return fmt.Errorf("config %q block_diffusion objective paths cannot be combined with training.distillation in v1", source)
	}
	if t.Data2VecActive() {
		return fmt.Errorf("config %q block_diffusion objective paths cannot be combined with training.data2vec in v1", source)
	}

	hasPlain := false
	for i, block := range cfg.Blocks {
		switch blockTypeKey(block) {
		case "plain":
			hasPlain = true
		case "swiglu", "geglu", "mlp", "moe":
		default:
			return fmt.Errorf("config %q blocks[%d].type=%q cannot be combined with block_diffusion objective paths in v1; supported blocks are plain self-attention plus position-wise FFN/MoE blocks", source, i, block.Type)
		}
	}
	if !hasPlain {
		return fmt.Errorf("config %q a block_diffusion objective path requires at least one type=plain block", source)
	}
	return nil
}
