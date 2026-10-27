package train

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// Print once at setup, not per step. MLX evaluates lazily on a worker thread;
// this source-name map lets its backend signature notices be attributed without
// introducing training state or source-name dependencies into the MLX patch.
func writeGridConvolutionContext(w io.Writer, cfg *ArchConfig, policy gpu.PrecisionPolicy) {
	if !cfg.GridEnabled() || policy.Backend != "cuda" || policy.TF32Allowed || policy.ComputeDType != "float32" {
		return
	}
	ops, err := arch.GridConvolutionDiagnostics(cfg)
	if err != nil {
		_, _ = fmt.Fprintf(w, "[grid convolution context] unavailable: %v\n", err)
		return
	}
	for _, op := range ops {
		_, _ = fmt.Fprintf(w, "[grid convolution context] precision=strict-fp32 %s\n", op)
	}
}

func annotateGridPreflightError(err error, cfg *ArchConfig, plan mlxMemoryLimitPlan) error {
	if err == nil {
		return nil
	}
	err = annotateGridMemoryError(err, "preflight before full validation", cfg, plan)
	backend := gpu.PrecisionBackend()
	if plan.DedicatedDevice {
		backend = "cuda"
	}
	policy, policyErr := gpu.ResolvePrecisionPolicy(cfg.Training.EffectiveComputeDType(), backend, cfg.Training.AllowTF32, os.LookupEnv)
	precision := fmt.Sprintf("compute=%s allow_tf32=%t", policy.ComputeDType, policy.TF32Allowed)
	if policyErr != nil {
		precision = fmt.Sprintf("precision unavailable: %v", policyErr)
	}
	ops, shapeErr := arch.GridConvolutionDiagnostics(cfg)
	if shapeErr != nil {
		return fmt.Errorf("grid execution preflight (%s; graph context unavailable: %v): %w", precision, shapeErr, err)
	}
	return fmt.Errorf("grid execution preflight (%s); convolution candidates: %s: %w", precision, strings.Join(ops, "; "), err)
}
