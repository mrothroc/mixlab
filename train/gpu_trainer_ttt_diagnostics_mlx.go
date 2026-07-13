//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

// EvaluateTTTDiagnosticsGPU samples TTT scan diagnostics with live weights and
// no outer gradient. The named evaluator compiles and caches this graph
// independently from the loss-only optimizer step.
func (t *mlxGPUTrainer) EvaluateTTTDiagnosticsGPU(
	batch objectiveBatch,
	batchSize, seqLen int,
) (map[string]float64, error) {
	if len(t.tttDiagnosticOutputs) == 0 {
		return nil, nil
	}
	if _, err := t.EvaluateObjectiveGPUWithOutputs(batch, batchSize, seqLen, t.tttDiagnosticOutputs); err != nil {
		return nil, fmt.Errorf("evaluate TTT diagnostics: %w", err)
	}
	result := make(map[string]float64, len(t.tttDiagnosticOutputs))
	for _, name := range t.tttDiagnosticOutputs {
		out, err := gpu.TrainerReadOutput(t.handle, name, []int{1})
		if err != nil {
			return nil, fmt.Errorf("read TTT diagnostic %q: %w", name, err)
		}
		if len(out) != 1 {
			return nil, fmt.Errorf("TTT diagnostic %q returned %d values, want 1", name, len(out))
		}
		result[name] = float64(out[0])
	}
	return result, nil
}
