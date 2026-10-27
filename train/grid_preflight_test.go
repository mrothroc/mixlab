package train

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestGridExecutionPreflightContext(t *testing.T) {
	cfg := gridPhaseConfig(t)
	no := false
	cfg.Training.AllowTF32 = &no
	t.Setenv("MLX_ENABLE_TF32", "0")
	t.Setenv("NVIDIA_TF32_OVERRIDE", "0")
	cfg.Blocks[0].Ops[0].Params["dilation"] = 2
	cfg.Blocks[0].Ops[0].Params["padding"] = 2
	cause := errors.New("No execution plans support the graph")
	err := annotateGridPreflightError(cause, cfg, mlxMemoryLimitPlan{DedicatedDevice: true})
	for _, text := range []string{"execution preflight", "compute=float32 allow_tf32=false", "convolution candidates:", "op=conv2d", "input=[2 8 8 2]", "weight=", "output=", "dilation=2", cfg.Blocks[0].Name + "." + cfg.Blocks[0].Ops[0].Output} {
		if !strings.Contains(err.Error(), text) {
			t.Fatalf("missing %q in %v", text, err)
		}
	}
	if !errors.Is(err, cause) {
		t.Fatal("lost original backend error")
	}
	if annotateGridPreflightError(nil, cfg, mlxMemoryLimitPlan{}) != nil {
		t.Fatal("nil error changed")
	}
}

func TestGridStrictConvolutionSourceContext(t *testing.T) {
	cfg := gridPhaseConfig(t)
	for _, backend := range []string{"metal", "cuda"} {
		for _, tf32 := range []bool{true, false} {
			var out bytes.Buffer
			writeGridConvolutionContext(&out, cfg, gpu.PrecisionPolicy{Backend: backend, ComputeDType: "float32", TF32Allowed: tf32})
			active := backend == "cuda" && !tf32
			if (out.Len() > 0) != active {
				t.Fatalf("backend=%s tf32=%v: %s", backend, tf32, &out)
			}
			if active && !strings.Contains(out.String(), cfg.Blocks[0].Name+"."+cfg.Blocks[0].Ops[0].Output) {
				t.Fatal("missing source op name")
			}
		}
	}
}
