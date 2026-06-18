package arch

import (
	"fmt"
	"strings"
)

const (
	NormTypeRMSNorm   = "rmsnorm"
	NormTypeLayerNorm = "layernorm"

	NormPlacementPre      = "pre"
	NormPlacementPost     = "post"
	NormPlacementSandwich = "sandwich"
)

type NormSpec struct {
	Type   string
	Eps    float32
	Affine bool
}

func defaultNormSpec() NormSpec {
	return NormSpec{Type: NormTypeRMSNorm, Eps: 1e-5, Affine: true}
}

func normalizeNormType(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "rmsnorm", "rms_norm", "rms":
		return NormTypeRMSNorm
	case "layernorm", "layer_norm", "layer":
		return NormTypeLayerNorm
	default:
		return strings.ToLower(strings.TrimSpace(v))
	}
}

func normalizeNormPlacement(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "pre", "prenorm", "pre_norm":
		return NormPlacementPre
	case "post", "postnorm", "post_norm":
		return NormPlacementPost
	case "sandwich", "sandwich_norm":
		return NormPlacementSandwich
	default:
		return strings.ToLower(strings.TrimSpace(v))
	}
}

func (cfg *ArchConfig) EffectiveNormSpec() NormSpec {
	spec := defaultNormSpec()
	if cfg == nil {
		return spec
	}
	spec.Type = normalizeNormType(cfg.NormType)
	if cfg.NormEps > 0 {
		spec.Eps = cfg.NormEps
	}
	if cfg.NormAffine != nil {
		spec.Affine = *cfg.NormAffine
	}
	return spec
}

func (cfg *ArchConfig) EffectiveNormPlacement() string {
	if cfg == nil {
		return NormPlacementPre
	}
	return normalizeNormPlacement(cfg.NormPlacement)
}

func normSpecOrDefault(spec NormSpec) NormSpec {
	if spec.Type == "" && spec.Eps == 0 && !spec.Affine {
		return defaultNormSpec()
	}
	if spec.Type == "" {
		spec.Type = NormTypeRMSNorm
	}
	spec.Type = normalizeNormType(spec.Type)
	if spec.Eps == 0 {
		spec.Eps = 1e-5
	}
	return spec
}

func normPlacementOrDefault(v string) string {
	return normalizeNormPlacement(v)
}

func normWeights(name string, dim int, spec NormSpec) []WeightMeta {
	spec = normSpecOrDefault(spec)
	switch spec.Type {
	case NormTypeRMSNorm:
		rmsName := name
		if name != "final_norm" && !strings.HasSuffix(name, "_scale") {
			rmsName = name + "_scale"
		}
		return []WeightMeta{{Name: rmsName, Shape: []int{dim}, IsNormScale: true, InitOne: true}}
	case NormTypeLayerNorm:
		if !spec.Affine {
			return nil
		}
		return []WeightMeta{
			{Name: name + "_scale", Shape: []int{dim}, IsNormScale: true, InitOne: true},
			{Name: name + "_bias", Shape: []int{dim}, InitZero: true},
		}
	default:
		return []WeightMeta{{Name: name, Shape: []int{dim}, IsNormScale: true, InitOne: true}}
	}
}

func emitNormIR(prog *Program, x string, wi int, output string, spec NormSpec) (int, error) {
	spec = normSpecOrDefault(spec)
	switch spec.Type {
	case NormTypeRMSNorm:
		if !spec.Affine {
			return wi, fmt.Errorf("norm_affine=false is not supported with rmsnorm")
		}
		prog.RMSNorm(x, weightName(wi), output, spec.Eps)
		return wi + 1, nil
	case NormTypeLayerNorm:
		if spec.Affine {
			prog.LayerNorm(x, weightName(wi), weightName(wi+1), output, spec.Eps)
			return wi + 2, nil
		}
		prog.LayerNormNoAffine(x, output, spec.Eps)
		return wi, nil
	default:
		return wi, fmt.Errorf("unsupported norm_type=%q", spec.Type)
	}
}

func emitNamedNormIR(prog *Program, x string, wi int, output string, spec NormSpec) (int, error) {
	return emitNormIR(prog, x, wi, output, spec)
}

func isDefaultNormConfig(cfg *ArchConfig) bool {
	if cfg == nil {
		return true
	}
	spec := cfg.EffectiveNormSpec()
	return spec.Type == NormTypeRMSNorm && spec.Eps == 1e-5 && spec.Affine && cfg.EffectiveNormPlacement() == NormPlacementPre && !cfg.FFNInternalNorm
}
