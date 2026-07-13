package train

import (
	"fmt"
	"sort"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

// declaredComponentLossOutputs returns public scalar loss diagnostics. The IR
// declaration is the contract: inactive/no-op objectives do not declare an
// output and therefore cannot appear in telemetry.
func declaredComponentLossOutputs(prog *arch.Program) []string {
	if prog == nil {
		return nil
	}
	outputs := make([]string, 0)
	for _, out := range prog.Outputs {
		if out.DType != arch.TensorFloat32 || len(out.Shape) != 1 || out.Shape[0] != 1 {
			continue
		}
		if out.Name != "eval_loss" && (strings.HasSuffix(out.Name, "_loss") || strings.Contains(out.Name, "_ttt_")) {
			outputs = append(outputs, out.Name)
		}
	}
	return outputs
}

func formatTrainingExtraDiagnostics(values map[string]float64) string {
	if len(values) == 0 {
		return ""
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%.5g", key, values[key]))
	}
	return strings.Join(parts, " ")
}

func splitTrainingDiagnostics(values map[string]float64) (map[string]float64, map[string]float64) {
	if len(values) == 0 {
		return nil, nil
	}
	losses := make(map[string]float64)
	extra := make(map[string]float64)
	for name, value := range values {
		if strings.Contains(name, "_ttt_") && !strings.HasSuffix(name, "_loss") {
			extra[name] = value
			continue
		}
		losses[name] = value
	}
	if len(losses) == 0 {
		losses = nil
	}
	if len(extra) == 0 {
		extra = nil
	}
	return losses, extra
}

func readTrainingStepComponentLosses(trainer GPUTrainer, enabled bool) (map[string]float64, error) {
	if !enabled {
		return nil, nil
	}
	type componentLossReader interface {
		ReadComponentLossesGPU() (map[string]float64, error)
	}
	reader, ok := trainer.(componentLossReader)
	if !ok {
		return nil, nil
	}
	return reader.ReadComponentLossesGPU()
}

func enableTrainingStepComponentLossCapture(trainer GPUTrainer) error {
	type componentLossCaptureConfigurer interface {
		EnableComponentLossCapture() error
	}
	configurer, ok := trainer.(componentLossCaptureConfigurer)
	if !ok {
		return nil
	}
	if err := configurer.EnableComponentLossCapture(); err != nil {
		return fmt.Errorf("enable training-step component loss capture: %w", err)
	}
	return nil
}
