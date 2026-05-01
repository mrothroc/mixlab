package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

const (
	backoutTensorName       = "x_backout"
	backoutLambdaWeightName = "backout_lambda"
	defaultBackoutLambda    = -1.0
)

// BackoutSpec controls final-latent residual subtraction. The builder captures
// the residual stream after SaveLayer and subtracts a learned scalar-weighted
// copy immediately before the final RMSNorm.
type BackoutSpec struct {
	SaveLayer  int     `json:"save_layer"`
	LambdaInit float32 `json:"lambda_init"`

	saveLayerSet  bool
	lambdaInitSet bool
}

// UnmarshalJSON records field presence so save_layer can remain a valid zero
// value while still being required when the backout block is present.
func (b *BackoutSpec) UnmarshalJSON(data []byte) error {
	var raw struct {
		SaveLayer  *int     `json:"save_layer"`
		LambdaInit *float32 `json:"lambda_init"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return err
	}
	if raw.SaveLayer != nil {
		b.SaveLayer = *raw.SaveLayer
		b.saveLayerSet = true
	}
	if raw.LambdaInit != nil {
		b.LambdaInit = *raw.LambdaInit
		b.lambdaInitSet = true
	}
	return nil
}

func (b *BackoutSpec) effectiveLambdaInit() float32 {
	if b == nil {
		return 0
	}
	if b.lambdaInitSet || b.LambdaInit != 0 {
		return b.LambdaInit
	}
	return defaultBackoutLambda
}

func validateBackout(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Backout == nil {
		return nil
	}
	if !cfg.Backout.saveLayerSet {
		return fmt.Errorf("config %q backout.save_layer is required", source)
	}
	if err := validateBackoutBounds(cfg.Backout, len(cfg.Blocks), cfg.UNet, source); err != nil {
		return err
	}
	cfg.Backout.LambdaInit = cfg.Backout.effectiveLambdaInit()
	cfg.Backout.lambdaInitSet = true
	return nil
}

func validateBackoutBounds(spec *BackoutSpec, numBlocks int, unet bool, source string) error {
	if spec == nil {
		return nil
	}
	if numBlocks <= 0 {
		return fmt.Errorf("config %q backout requires at least one block", source)
	}
	if spec.SaveLayer < 0 || spec.SaveLayer >= numBlocks {
		return fmt.Errorf("config %q has invalid backout.save_layer=%d (must be in [0,%d))", source, spec.SaveLayer, numBlocks)
	}
	if spec.SaveLayer == numBlocks-1 {
		return fmt.Errorf("config %q has invalid backout.save_layer=%d (must be < len(blocks)-1 to avoid a no-op)", source, spec.SaveLayer)
	}
	lambda := float64(spec.effectiveLambdaInit())
	if math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		return fmt.Errorf("config %q has invalid backout.lambda_init=%g (must be finite)", source, spec.effectiveLambdaInit())
	}
	if unet {
		return fmt.Errorf("config %q backout is not supported with unet", source)
	}
	return nil
}

func backoutWeightShapes(spec *BackoutSpec) []WeightMeta {
	if spec == nil {
		return nil
	}
	return []WeightMeta{{
		Name:      backoutLambdaWeightName,
		Shape:     []int{1},
		InitValue: spec.effectiveLambdaInit(),
	}}
}

type backoutBuildPlan struct {
	enabled     bool
	saveLayer   int
	weightIndex int
	captured    bool
}

func newBackoutBuildPlan(spec *BackoutSpec, numBlocks int, unet bool, source string) (backoutBuildPlan, error) {
	if spec == nil {
		return backoutBuildPlan{}, nil
	}
	if err := validateBackoutBounds(spec, numBlocks, unet, source); err != nil {
		return backoutBuildPlan{}, err
	}
	return backoutBuildPlan{
		enabled:   true,
		saveLayer: spec.SaveLayer,
	}, nil
}

func (p *backoutBuildPlan) setWeightIndex(idx int) {
	if p == nil || !p.enabled {
		return
	}
	p.weightIndex = idx
}

func (p *backoutBuildPlan) captureAfterBlock(prog *Program, blockIdx int, stream string) {
	if p == nil || !p.enabled || p.captured || blockIdx != p.saveLayer {
		return
	}
	prog.ScalarMul(stream, 1.0, backoutTensorName)
	p.captured = true
}

func (p *backoutBuildPlan) applyBeforeFinalNorm(prog *Program, stream string) error {
	if p == nil || !p.enabled {
		return nil
	}
	if !p.captured {
		return fmt.Errorf("backout save_layer=%d was not emitted", p.saveLayer)
	}
	scaled := backoutTensorName + "_scaled"
	prog.Mul(backoutTensorName, weightName(p.weightIndex), scaled)
	prog.Sub(stream, scaled, stream)
	return nil
}
