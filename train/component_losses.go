package train

import (
	"fmt"
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
		if out.Name != "eval_loss" && strings.HasSuffix(out.Name, "_loss") {
			outputs = append(outputs, out.Name)
		}
	}
	return outputs
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
