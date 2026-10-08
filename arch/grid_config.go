package arch

import (
	"fmt"
	"math"
	"reflect"
)

const ObjectiveDenseRegression = "dense_regression"
const InputAdapterGrid = "grid"

// DenseRegressionSpec selects a spatial graph output and reporting units.
type DenseRegressionSpec struct {
	Output         string   `json:"output"`
	TargetChannels int      `json:"target_channels"`
	MetricScale    *float64 `json:"metric_scale,omitempty"`
}

func (cfg *ArchConfig) GridEnabled() bool {
	return cfg != nil && cfg.EffectiveInputAdapterKind() == InputAdapterGrid
}

func (s *DenseRegressionSpec) EffectiveMetricScale() float64 {
	if s == nil || s.MetricScale == nil {
		return 1
	}
	return *s.MetricScale
}

// rejectGridExtras makes new sequence-only fields fail closed in spatial mode.
// Private bookkeeping fields are not a public configuration surface.
func rejectGridExtras(value any, allowed map[string]bool, scope string) error {
	v := reflect.ValueOf(value)
	t := v.Type()
	var defaults reflect.Value
	if _, ok := value.(TrainingSpec); ok {
		d := TrainingSpec{}
		d.ApplyDefaults()
		d.MLMMaskProbScheduleMode = "step"
		d.HybridCLMFractionScheduleMode = "step"
		defaults = reflect.ValueOf(d)
	}
	for i := 0; i < v.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() || allowed[f.Name] || f.Tag.Get("json") == "-" {
			continue
		}
		if !v.Field(i).IsZero() {
			if defaults.IsValid() && reflect.DeepEqual(v.Field(i).Interface(), defaults.Field(i).Interface()) {
				continue
			}
			return fmt.Errorf("%s.%s is unsupported for dense_regression", scope, f.Name)
		}
	}
	return nil
}

func validateGridConfig(cfg *ArchConfig, source string) (*ArchConfig, error) {
	if !cfg.GridEnabled() || cfg.DenseRegression == nil || cfg.Training.Objective != ObjectiveDenseRegression {
		return nil, fmt.Errorf("config %q dense_regression requires input_adapter.kind=grid, dense_regression, and training.objective=dense_regression", source)
	}
	if err := rejectGridExtras(*cfg, map[string]bool{"Name": true, "InputAdapter": true, "DenseRegression": true, "Blocks": true, "Training": true}, "config"); err != nil {
		return nil, err
	}
	if err := rejectGridExtras(*cfg.InputAdapter, map[string]bool{"Kind": true, "Channels": true, "Height": true, "Width": true}, "input_adapter"); err != nil {
		return nil, err
	}
	i := cfg.InputAdapter
	if err := cfg.Training.GridLoader.validate(); err != nil {
		return nil, err
	}
	if a := cfg.Training.GridAugmentation; a != nil && a.Dihedral && i.Height != i.Width {
		return nil, fmt.Errorf("grid_augmentation.dihedral requires square input geometry")
	}
	if _, err := gridSize([]int{cfg.Training.BatchSize, i.Height, i.Width, i.Channels}); err != nil {
		return nil, fmt.Errorf("grid dimensions/batch_size: %w", err)
	}
	if cfg.DenseRegression.TargetChannels <= 0 {
		return nil, fmt.Errorf("dense_regression.target_channels must be positive")
	}
	scale := cfg.DenseRegression.EffectiveMetricScale()
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return nil, fmt.Errorf("dense_regression.metric_scale must be finite and positive")
	}
	if len(cfg.Blocks) != 1 || cfg.Blocks[0].Type != "custom" || cfg.Blocks[0].Name == "" {
		return nil, fmt.Errorf("dense_regression requires one named custom block")
	}
	if err := rejectGridExtras(cfg.Blocks[0], map[string]bool{"Type": true, "Name": true, "Weights": true, "Ops": true}, "blocks[0]"); err != nil {
		return nil, err
	}
	if cfg.Training.batchTokensSet || cfg.Training.BatchTokens != 0 {
		return nil, fmt.Errorf("dense_regression uses batch_size, not batch_tokens")
	}
	// The common defaults include sequence-only knobs; validate explicit policy
	// before applying them, then leave those defaults inert in the grid runner.
	allowed := map[string]bool{"GridLoader": true, "GridAugmentation": true, "InitFrom": true, "InitAllowMissing": true, "Freeze": true, "Phases": true}
	for _, name := range []string{"Objective", "BatchSize", "Steps", "Seed", "LR", "Optimizer", "WeightDecay", "WeightDecayPolicy", "EmbedLR", "MatrixLR", "ScalarLR", "HeadLR", "EmbedWeightDecay", "MatrixWeightDecay", "ScalarWeightDecay", "HeadWeightDecay", "Beta1", "Beta2", "Epsilon", "LAMBBeta1", "LAMBBeta2", "LAMBEps", "LAMBTrustRatioCap", "GradClip", "WarmupSteps", "WarmupRatio", "HoldSteps", "WarmdownSteps", "MinLRFraction", "LRScheduleSteps", "WeightInit", "WeightInitStd", "ComputeDType", "ValEverySteps", "ValExamples", "EarlyStop", "TargetValLoss"} {
		allowed[name] = true
	}
	if err := rejectGridExtras(cfg.Training, allowed, "training"); err != nil {
		return nil, err
	}
	if cfg.Training.Optimizer != "adamw" && cfg.Training.Optimizer != "lamb" {
		return nil, fmt.Errorf("dense_regression requires optimizer adamw or lamb")
	}
	if cfg.Training.Steps < 0 || cfg.Training.ValEverySteps < 0 || cfg.Training.ValExamples != 0 {
		return nil, fmt.Errorf("dense_regression requires nonnegative steps/val_every_steps and full-split validation (val_examples=0)")
	}
	cfg.Training.ApplyDefaults()
	if err := validateCommonTrainingSettings(cfg, source); err != nil {
		return nil, err
	}
	if err := validateTrainingRecipeKnobs(cfg, source); err != nil {
		return nil, err
	}
	// Do not serialize inert language-model defaults into a grid checkpoint.
	v := reflect.ValueOf(&cfg.Training).Elem()
	for n := 0; n < v.NumField(); n++ {
		f := v.Type().Field(n)
		if f.IsExported() && !allowed[f.Name] && f.Tag.Get("json") != "-" {
			v.Field(n).SetZero()
		}
	}
	if _, _, err := buildGridGraph(cfg, true); err != nil {
		return nil, fmt.Errorf("config %q: %w", source, err)
	}
	return cfg, nil
}

// Keep shapes bounded before any allocation or conversion to MLX int dimensions.
func gridSize(shape []int) (int, error) {
	n := int64(1)
	for _, d := range shape {
		if d <= 0 || int64(d) > math.MaxInt32 || n > math.MaxInt32/int64(d) {
			return 0, fmt.Errorf("invalid or overflowing shape %v", shape)
		}
		n *= int64(d)
	}
	return int(n), nil
}
