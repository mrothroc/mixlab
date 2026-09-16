package arch

import (
	"fmt"
	"math"
	"strings"
)

type PatchImageSpec struct {
	Height   int `json:"height"`
	Width    int `json:"width"`
	Channels int `json:"channels"`
}

type PatchAugmentSpec struct {
	HFlip         bool      `json:"hflip,omitempty"`
	RandomCropPad int       `json:"random_crop_pad,omitempty"`
	PadValue      []float32 `json:"pad_value,omitempty"`
}

func (cfg *ArchConfig) LinearPatchesEnabled() bool {
	return cfg != nil && cfg.EffectiveInputAdapterKind() == InputAdapterLinearPatches
}

func (cfg *ArchConfig) ContinuousInputEnabled() bool {
	return cfg != nil && (cfg.LinearFramesEnabled() || cfg.LinearPatchesEnabled())
}

func validatePatchAdapter(cfg *ArchConfig, source string) error {
	s := cfg.InputAdapter
	if s == nil {
		return nil
	}
	if !cfg.LinearPatchesEnabled() {
		if s.Image != nil || s.Patch != 0 || s.Coords != "" || s.Augment != nil {
			return fmt.Errorf("config %q image/patch/coords/augment require input_adapter.kind=linear_patches", source)
		}
		return nil
	}
	if s.Image == nil || s.Patch <= 0 || s.Image.Height <= 0 || s.Image.Width <= 0 || s.Image.Channels <= 0 {
		return fmt.Errorf("config %q linear_patches requires positive image height/width/channels and patch", source)
	}
	h, w, c, p := s.Image.Height, s.Image.Width, s.Image.Channels, s.Patch
	if h%p != 0 || w%p != 0 {
		return fmt.Errorf("config %q image height/width must be divisible by patch", source)
	}
	if h > math.MaxInt32/w || h*w > math.MaxInt32/c {
		return fmt.Errorf("config %q patch image exceeds int32 indexing", source)
	}
	t, f := (h/p)*(w/p), p*p*c
	if cfg.SeqLen != t || (s.FeatureDim != 0 && s.FeatureDim != f) {
		return fmt.Errorf("config %q patch geometry requires seq_len=%d and feature_dim=%d", source, t, f)
	}
	s.FeatureDim = f
	s.Coords = strings.ToLower(strings.TrimSpace(s.Coords))
	if s.Coords == "" {
		s.Coords = "none"
	}
	if s.Coords != "none" && s.Coords != "learned_xy" {
		return fmt.Errorf("config %q input_adapter.coords must be none or learned_xy", source)
	}
	if cfg.Training.LengthBucketsChangeShape(cfg.SeqLen) || len(cfg.Training.SeqLenSchedule) > 0 {
		return fmt.Errorf("config %q linear_patches requires fixed geometry, not length bucketing or seq_len schedules", source)
	}
	if a := s.Augment; a != nil {
		if a.RandomCropPad < 0 || a.RandomCropPad > (math.MaxInt32-max(h, w))/2 {
			return fmt.Errorf("config %q random_crop_pad must be non-negative and fit int32 geometry", source)
		}
		if len(a.PadValue) != 0 && len(a.PadValue) != c {
			return fmt.Errorf("config %q pad_value must contain exactly %d channels", source, c)
		}
		for _, v := range a.PadValue {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("config %q pad_value must be finite", source)
			}
		}
	}
	return nil
}

func patchCoordinateWeightShapes(cfg *ArchConfig) []WeightMeta {
	if !cfg.LinearPatchesEnabled() || cfg.InputAdapter.Coords != "learned_xy" {
		return nil
	}
	s := cfg.InputAdapter
	return []WeightMeta{
		{Name: "input_adapter_coord_x", Shape: []int{s.Image.Width / s.Patch, cfg.ModelDim}},
		{Name: "input_adapter_coord_y", Shape: []int{s.Image.Height / s.Patch, cfg.ModelDim}},
	}
}

func emitPatchCoordinates(prog *Program, state string, spec *InputAdapterSpec, wi, b, d int) (string, int) {
	if spec == nil || spec.Kind != InputAdapterLinearPatches || spec.Coords != "learned_xy" {
		return state, wi
	}
	h, w := spec.Image.Height/spec.Patch, spec.Image.Width/spec.Patch
	prog.Reshape(weightName(wi), []int{1, 1, w, d}, "patch_x")
	prog.Reshape(weightName(wi+1), []int{1, h, 1, d}, "patch_y")
	prog.Add("patch_x", "patch_y", "patch_xy")
	prog.Reshape("patch_xy", []int{1, h * w, d}, "patch_coords")
	prog.Reshape(state, []int{b, h * w, d}, "patch_projected")
	prog.Add("patch_projected", "patch_coords", "patch_positioned")
	return "patch_positioned", wi + 2
}
