package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

func validateHFContinuousS4DComposition(cfg *ArchConfig) error {
	if cfg == nil || !cfg.LinearFramesEnabled() {
		return unsupportedHFExport("input_adapter.kind", "continuous S4D export requires linear_frames")
	}
	if !cfg.ClassificationEnabled() || cfg.Training.Classification == nil {
		return unsupportedHFExport("training.objective", "linear_frames HF export requires a native classification checkpoint")
	}
	if cfg.EffectiveNormSpec().Type == arch.NormTypeBatchNorm {
		return unsupportedHFExport("norm_type", "continuous S4D HF export does not yet carry BatchNorm running statistics")
	}
	if cfg.EffectiveLayerAggregation() != "none" {
		return unsupportedHFExport("layer_aggregation", "continuous S4D HF export does not support layer aggregation in v1")
	}
	if cfg.FFNInternalNorm {
		return unsupportedHFExport("ffn_internal_norm", "continuous S4D-only exports have no FFN internal norm")
	}
	switch cfg.EffectiveInputAdapterNorm() {
	case arch.InputAdapterNormNone, arch.InputAdapterNormLayerNorm:
	default:
		return unsupportedHFExport("input_adapter.norm", fmt.Sprintf("unsupported continuous input norm %q", cfg.EffectiveInputAdapterNorm()))
	}
	switch cfg.EffectiveNormPlacement() {
	case arch.NormPlacementPre, arch.NormPlacementPost, arch.NormPlacementPostResidual, arch.NormPlacementSandwich:
	default:
		return unsupportedHFExport("norm_placement", fmt.Sprintf("unsupported S4D norm placement %q", cfg.EffectiveNormPlacement()))
	}
	if len(cfg.Blocks) == 0 {
		return unsupportedHFExport("blocks", "continuous S4D export requires at least one s4d block")
	}
	for i, block := range cfg.Blocks {
		if !strings.EqualFold(strings.TrimSpace(block.Type), "s4d") {
			return unsupportedHFExport(
				fmt.Sprintf("blocks[%d].type", i),
				"continuous S4D export v1 supports sequential s4d-only stacks",
			)
		}
	}
	return nil
}

func hfS4DBlockEntry(cfg *ArchConfig, block BlockSpec) map[string]any {
	stateSize := block.StateSize
	if stateSize <= 0 {
		stateSize = 64
	}
	nSSM := block.NSSM
	if nSSM <= 0 {
		nSSM = cfg.ModelDim
	}
	discretization := strings.ToLower(strings.TrimSpace(block.Discretization))
	if discretization == "" {
		discretization = arch.S4DDiscretizationZOH
	}
	entry := map[string]any{
		"type":             "s4d",
		"state_size":       stateSize,
		"n_ssm":            nSSM,
		"bidirectional":    block.Bidirectional,
		"discretization":   discretization,
		"trainable_b":      block.TrainableB,
		"output_transform": hfS4DOutputTransform(block),
		"tie_dropout":      cfg.TieDropout || block.TieDropout,
	}
	if block.FreqScale != nil {
		entry["freq_scale"] = *block.FreqScale
	}
	if block.S4DSobolevFilterEnabled() {
		lr := arch.DefaultS4DSobolevLR
		if block.SobolevFilter.LearningRate != nil {
			lr = *block.SobolevFilter.LearningRate
		}
		sobolev := map[string]any{
			"beta_init":     block.SobolevFilter.BetaInit,
			"learning_rate": lr,
			"trainable":     arch.EffectiveS4DSobolevTrainable(block),
			"weight_decay":  arch.EffectiveS4DSobolevWeightDecay(block),
			"granularity":   arch.EffectiveS4DSobolevGranularity(block),
		}
		if lo, hi, bounded := arch.S4DSobolevBounds(block); bounded {
			sobolev["bounds"] = []float64{lo, hi}
		}
		entry["sobolev_filter"] = sobolev
	}
	return entry
}

func hfS4DOutputTransform(block BlockSpec) string {
	if strings.EqualFold(strings.TrimSpace(block.OutputTransform), "glu") {
		return "glu"
	}
	return "none"
}

func hfFinalNormOverride(cfg *ArchConfig) *bool {
	if cfg == nil || cfg.EffectiveFinalNorm() {
		return nil
	}
	value := false
	return &value
}

func hfS4DWeightNames(block BlockSpec, normPlacement string, norm arch.NormSpec) []hfBlockWeightName {
	names := make([]hfBlockWeightName, 0, 16)
	if normPlacement == arch.NormPlacementPre || normPlacement == arch.NormPlacementSandwich {
		names = appendHFNormWeightNames(names, "s4d_norm", "norm", norm)
	}
	names = append(names,
		hfBlockWeightName{mixlab: "s4d_log_dt", hf: "log_dt"},
		hfBlockWeightName{mixlab: "s4d_log_A_real", hf: "log_A_real"},
		hfBlockWeightName{mixlab: "s4d_A_imag", hf: "A_imag"},
	)
	if block.TrainableB {
		names = append(names,
			hfBlockWeightName{mixlab: "s4d_B_real", hf: "B_real"},
			hfBlockWeightName{mixlab: "s4d_B_imag", hf: "B_imag"},
		)
	}
	names = append(names,
		hfBlockWeightName{mixlab: "s4d_C_real", hf: "C_real"},
		hfBlockWeightName{mixlab: "s4d_C_imag", hf: "C_imag"},
	)
	if block.Bidirectional {
		names = append(names,
			hfBlockWeightName{mixlab: "s4d_C_backward_real", hf: "C_backward_real"},
			hfBlockWeightName{mixlab: "s4d_C_backward_imag", hf: "C_backward_imag"},
		)
	}
	names = append(names, hfBlockWeightName{mixlab: "s4d_D", hf: "D"})
	if block.S4DSobolevFilterEnabled() {
		names = append(names, hfBlockWeightName{mixlab: "s4d_sobolev_beta", hf: "sobolev_beta"})
	}
	if hfS4DOutputTransform(block) == "glu" {
		names = append(names,
			hfBlockWeightName{mixlab: "s4d_out_proj", hf: "out_proj.weight"},
			hfBlockWeightName{mixlab: "s4d_out_bias", hf: "out_proj.bias"},
		)
	}
	if normPlacement == arch.NormPlacementPost || normPlacement == arch.NormPlacementSandwich {
		names = appendHFNormWeightNames(names, "s4d_post_norm", "post_norm", norm)
	}
	if normPlacement == arch.NormPlacementPostResidual {
		names = appendHFNormWeightNames(names, "s4d_post_residual_norm", "post_residual_norm", norm)
	}
	return names
}

func appendHFNormWeightNames(names []hfBlockWeightName, mixlabBase, hfBase string, norm arch.NormSpec) []hfBlockWeightName {
	switch norm.Type {
	case arch.NormTypeRMSNorm:
		return append(names, hfBlockWeightName{mixlab: mixlabBase + "_scale", hf: hfBase + ".weight"})
	case arch.NormTypeLayerNorm:
		if !norm.Affine {
			return names
		}
		return append(names,
			hfBlockWeightName{mixlab: mixlabBase + "_scale", hf: hfBase + ".weight"},
			hfBlockWeightName{mixlab: mixlabBase + "_bias", hf: hfBase + ".bias"},
		)
	default:
		return names
	}
}
