package train

import "github.com/mrothroc/mixlab/gpu"

// TrainOptions holds optional parameters for runTrain.
type TrainOptions struct {
	SafetensorsPath     string // If set, export weights after training
	SafetensorsLoad     string // If set, load weights from safetensors file before training
	Resume              string // If set, restore a complete resumable checkpoint bundle
	Quantize            string // Quantization mode: "none", "int8", or "int6"
	QuantMethod         string // Quantization clipping method: "quantile" or "sdclip"
	QuantK              float32
	QuantKEmbed         float32
	DoFullEval          bool     // If true, run full BPB evaluation after training
	LUTDir              string   // Directory containing BPB lookup tables
	CheckpointDir       string   // Directory for periodic safetensors checkpoints
	CheckpointEvery     int      // Save checkpoint every N steps; 0 disables
	LogEvery            int      // Print progress every N steps; 0 uses default/env cadence
	ValEvery            int      // Run validation every N steps; 0 uses default/env cadence
	Timing              bool     // If true, print per-step timing breakdown at log intervals
	PProfAddr           string   // If set, serve live pprof and Mixlab telemetry on this address
	TelemetryOut        string   // If set, write periodic telemetry snapshots as JSONL
	SWAStartOverride    *int     // If set, overrides training.swa_start
	SWADecayOverride    *float32 // If set, overrides training.swa_decay
	SWAIntervalOverride *int     // If set, overrides training.swa_interval
	telemetry           *telemetryRuntime

	// OptimizerOverride lets callers customize the optimizer plan that RunArch
	// builds before the GPU trainer is created.
	//
	// The callback receives the auto-generated default plan and the weight shapes
	// that plan was built from, and returns a replacement plan. The returned plan
	// must cover every weight exactly once; RunArch validates this and returns an
	// error otherwise.
	//
	// Example:
	//
	//	OptimizerOverride: func(spec gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
	//		frozen := len(spec.Groups)
	//		spec.Groups = append(spec.Groups, gpu.OptimizerGroup{
	//			Kind: gpu.OptimizerAdamW,
	//			LR:   0,
	//		})
	//		for i, shape := range shapes {
	//			if shape.Name == "embed" {
	//				spec.Weights[i].GroupIndex = frozen
	//			}
	//		}
	//		return spec, nil
	//	}
	//
	// Most callers should leave this nil.
	OptimizerOverride func(defaultPlan gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error)
}
