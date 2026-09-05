//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"testing"
)

func TestMamba3DebugScanFallbackTrainingFlags(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, disableCompiled := range []string{"0", "1"} {
		for _, allowFallback := range []string{"0", "1"} {
			t.Run(fmt.Sprintf("disable_compiled=%s/allow_fallback=%s", disableCompiled, allowFallback), func(t *testing.T) {
				t.Setenv("MIXLAB_FORCE_COMPILED_STEP", "0")
				t.Setenv("MIXLAB_DISABLE_COMPILED_STEP", "0")
				t.Setenv("MIXLAB_DISABLE_MAMBA3_COMPILED_STEP", disableCompiled)
				t.Setenv("MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE", "1")
				t.Setenv("MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK", allowFallback)
				// Exercise the composed scan on Metal too, not a second native path.
				t.Setenv("MIXLAB_MAMBA3_DISABLE_METAL_PRIMITIVE", "1")
				cfg, err := ParseArchConfig([]byte(`{
					"model_dim":8,"vocab_size":8,"seq_len":4,
					"blocks":[{"type":"mamba3-canonical","inner_dim":8,
						"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false}],
					"training":{"optimizer":"adamw","batch_tokens":4,"lr":0.001,"grad_clip":1}
				}`), "mamba3-debug-flags")
				if err != nil {
					t.Fatal(err)
				}
				prog, err := BuildIRProgramFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				trainer, err := initGPUTrainer(prog, cfg, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				defer trainer.CloseTrainer()
				for step := 0; step < 2; step++ {
					err := trainer.SubmitStepGPU([]int{0, 1, 2, 3}, []int{1, 2, 3, 4}, 1, 4, 0.001)
					if allowFallback == "0" {
						if err == nil {
							t.Fatal("canonical training allowed CUDA scan fallback without opt-in")
						}
						for _, want := range []string{
							"requires BOTH MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE=1 and MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK=1",
							"MIXLAB_DISABLE_MAMBA3_COMPILED_STEP=1 changes training-step compilation only",
							"Standalone scan/eval", "MIXLAB_MAMBA3_DISABLE_METAL_PRIMITIVE=1",
						} {
							if !strings.Contains(err.Error(), want) {
								t.Errorf("error %q missing %q", err, want)
							}
						}
						return
					}
					if err != nil {
						t.Fatalf("advertised flag combination rejected: %v", err)
					}
					loss, err := trainer.CollectLossGPU()
					if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
						t.Fatalf("debug fallback step %d: loss=%g err=%v", step, loss, err)
					}
				}
			})
		}
	}
}
