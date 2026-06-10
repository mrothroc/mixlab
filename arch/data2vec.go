package arch

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const (
	Data2VecTargetNormLayer   = "layer_norm"
	Data2VecTargetNormNone    = "none"
	Data2VecMaskSourceObject  = "objective"
	data2VecDefaultLossWeight = 1.0
	data2VecDefaultTau        = 0.999
	data2VecDefaultTopK       = 8
	data2VecDefaultBeta       = 1.0
	data2VecDefaultNormEps    = 1e-5
)

func (d *Data2VecSpec) UnmarshalJSON(data []byte) error {
	type alias Data2VecSpec
	var raw alias
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*d = Data2VecSpec(raw)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	_, d.lossWeightSet = fields["loss_weight"]
	_, d.emaTauSet = fields["ema_tau"]
	_, d.emaTauStartSet = fields["ema_tau_start"]
	_, d.emaTauEndSet = fields["ema_tau_end"]
	_, d.topKLayersSet = fields["top_k_layers"]
	_, d.smoothL1BetaSet = fields["smooth_l1_beta"]
	_, d.targetNormEpsSet = fields["target_norm_eps"]
	return nil
}

func (d *Data2VecSpec) applyDefaults() {
	if d == nil {
		return
	}
	if !d.lossWeightSet {
		d.LossWeight = data2VecDefaultLossWeight
	}
	if !d.emaTauSet {
		d.EMATau = data2VecDefaultTau
	}
	if !d.emaTauStartSet {
		d.EMATauStart = d.EMATau
	}
	if !d.emaTauEndSet {
		d.EMATauEnd = d.EMATau
	}
	if !d.topKLayersSet {
		d.TopKLayers = data2VecDefaultTopK
	}
	if !d.smoothL1BetaSet {
		d.SmoothL1Beta = data2VecDefaultBeta
	}
	d.TargetNorm = normalizeData2VecTargetNorm(d.TargetNorm)
	if !d.targetNormEpsSet {
		d.TargetNormEps = data2VecDefaultNormEps
	}
	d.MaskSource = strings.ToLower(strings.TrimSpace(d.MaskSource))
	if d.MaskSource == "" {
		d.MaskSource = Data2VecMaskSourceObject
	}
}

// Data2VecActive reports whether data2vec should alter weight layout or graph
// emission. A nil spec is disabled; loss_weight=0 is treated as intentionally
// disabled after validation/defaulting.
func (t TrainingSpec) Data2VecActive() bool {
	return t.Data2Vec != nil && t.Data2Vec.LossWeight > 0
}

func normalizeData2VecTargetNorm(norm string) string {
	switch strings.ToLower(strings.TrimSpace(norm)) {
	case "", Data2VecTargetNormLayer, "instance_norm", "feature_norm":
		return Data2VecTargetNormLayer
	case Data2VecTargetNormNone:
		return Data2VecTargetNormNone
	default:
		return strings.ToLower(strings.TrimSpace(norm))
	}
}

func validateTrainingData2Vec(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.Data2Vec == nil {
		return nil
	}
	d := cfg.Training.Data2Vec
	if !finiteNonNegative(d.LossWeight) {
		return fmt.Errorf("config %q has invalid training.data2vec.loss_weight=%g (must be finite and >= 0)", source, d.LossWeight)
	}
	if d.LossWeight == 0 {
		return nil
	}
	if !finiteInClosedRange(d.EMATau, 0, 1) {
		return fmt.Errorf("config %q has invalid training.data2vec.ema_tau=%g (must be in [0,1])", source, d.EMATau)
	}
	if !finiteInClosedRange(d.EMATauStart, 0, 1) {
		return fmt.Errorf("config %q has invalid training.data2vec.ema_tau_start=%g (must be in [0,1])", source, d.EMATauStart)
	}
	if !finiteInClosedRange(d.EMATauEnd, 0, 1) {
		return fmt.Errorf("config %q has invalid training.data2vec.ema_tau_end=%g (must be in [0,1])", source, d.EMATauEnd)
	}
	if d.EMATauRampSteps < 0 {
		return fmt.Errorf("config %q has invalid training.data2vec.ema_tau_ramp_steps=%d (must be >= 0)", source, d.EMATauRampSteps)
	}
	if d.TopKLayers <= 0 || d.TopKLayers > len(cfg.Blocks) {
		return fmt.Errorf("config %q has invalid training.data2vec.top_k_layers=%d (must be in [1,%d])", source, d.TopKLayers, len(cfg.Blocks))
	}
	if !finitePositive(d.SmoothL1Beta) {
		return fmt.Errorf("config %q has invalid training.data2vec.smooth_l1_beta=%g (must be finite and > 0)", source, d.SmoothL1Beta)
	}
	if d.TargetNorm != Data2VecTargetNormLayer && d.TargetNorm != Data2VecTargetNormNone {
		return fmt.Errorf("config %q has invalid training.data2vec.target_norm=%q (must be \"layer_norm\", \"instance_norm\", \"feature_norm\", or \"none\")", source, d.TargetNorm)
	}
	if !finitePositive(d.TargetNormEps) {
		return fmt.Errorf("config %q has invalid training.data2vec.target_norm_eps=%g (must be finite and > 0)", source, d.TargetNormEps)
	}
	if d.MaskSource != Data2VecMaskSourceObject {
		return fmt.Errorf("config %q has invalid training.data2vec.mask_source=%q (v1 supports only \"objective\")", source, d.MaskSource)
	}
	if d.MaskProb < 0 || d.MaskProb > 1 || math.IsNaN(d.MaskProb) || math.IsInf(d.MaskProb, 0) {
		return fmt.Errorf("config %q has invalid training.data2vec.mask_prob=%g (must be in [0,1])", source, d.MaskProb)
	}
	if d.PredictorHidden < 0 {
		return fmt.Errorf("config %q has invalid training.data2vec.predictor_hidden_dim=%d (must be >= 0)", source, d.PredictorHidden)
	}

	switch cfg.Training.EffectiveObjective() {
	case ObjectiveMLM, ObjectiveMNTP:
	case ObjectiveHybrid:
		if cfg.Training.HybridCLMFraction >= 1 {
			return fmt.Errorf("config %q has training.data2vec enabled but hybrid_clm_fraction=%g leaves no masked objective steps", source, cfg.Training.HybridCLMFraction)
		}
	default:
		return fmt.Errorf("config %q has training.data2vec enabled but objective=%q is not a masked objective", source, cfg.Training.Objective)
	}
	if cfg.Training.Distillation != nil {
		return fmt.Errorf("config %q cannot combine training.data2vec with training.distillation in v1", source)
	}
	if cfg.MTP != nil && cfg.MTP.EffectiveN() > 1 {
		return fmt.Errorf("config %q cannot combine training.data2vec with top-level mtp in v1", source)
	}
	if cfg.Training.FirstByteMask {
		return fmt.Errorf("config %q cannot combine training.data2vec with training.first_byte_mask in v1", source)
	}
	if len(cfg.RecurrencePhases) > 0 {
		return fmt.Errorf("config %q cannot combine training.data2vec with recurrence_phases in v1", source)
	}
	if len(cfg.Recurrence) > 0 {
		return fmt.Errorf("config %q cannot combine training.data2vec with recurrence weight sharing in v1", source)
	}
	if cfg.UNet {
		return fmt.Errorf("config %q cannot combine training.data2vec with unet in v1", source)
	}
	if cfg.ParallelResidual {
		return fmt.Errorf("config %q cannot combine training.data2vec with parallel_residual in v1", source)
	}
	return nil
}

func data2VecWeightShapes(modelDim int, spec *Data2VecSpec) []WeightMeta {
	if spec == nil || spec.LossWeight <= 0 {
		return nil
	}
	if spec.PredictorHidden > 0 {
		return []WeightMeta{
			{Name: "data2vec_pred_1", Shape: []int{modelDim, spec.PredictorHidden}},
			{Name: "data2vec_pred_2", Shape: []int{spec.PredictorHidden, modelDim}},
		}
	}
	return []WeightMeta{{Name: "data2vec_pred", Shape: []int{modelDim, modelDim}}}
}

func data2VecWeightCount(spec *Data2VecSpec) int {
	if spec == nil || spec.LossWeight <= 0 {
		return 0
	}
	if spec.PredictorHidden > 0 {
		return 2
	}
	return 1
}

type data2VecHiddenCapture struct {
	topK   int
	prefix string
	total  int
	names  []string
}

func newData2VecHiddenCapture(topK, total int, prefix string) *data2VecHiddenCapture {
	if topK <= 0 {
		return nil
	}
	if prefix == "" {
		prefix = "data2vec"
	}
	return &data2VecHiddenCapture{topK: topK, prefix: prefix, total: total}
}

func (c *data2VecHiddenCapture) captureAfterBlock(prog *Program, blockIdx int, stream string) {
	if c == nil || prog == nil {
		return
	}
	if blockIdx < c.total-c.topK {
		return
	}
	name := fmt.Sprintf("%s_layer_%02d_hidden_flat", c.prefix, blockIdx)
	prog.ScalarMul(stream, 1.0, name)
	c.names = append(c.names, name)
}

func (c *data2VecHiddenCapture) declareOutputs(prog *Program, rows, modelDim int) {
	if c == nil || prog == nil {
		return
	}
	for _, name := range c.names {
		prog.DeclareOutput(name, TensorFloat32, []int{rows, modelDim})
	}
}

func finiteNonNegative(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0
}

func finitePositive(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v > 0
}

func finiteInClosedRange(v, min, max float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= min && v <= max
}
