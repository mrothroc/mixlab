package gpu

import (
	"strings"
	"testing"
)

func TestMamba3DebugGuidanceMatchesTrainingGuard(t *testing.T) {
	policy := readRepositoryFile(t, "mamba3_debug_policy.h")
	for _, flag := range []string{
		"MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE",
		"MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK",
	} {
		if !strings.Contains(policy, flag) {
			t.Fatalf("shared fallback guidance missing %s", flag)
		}
	}
	for _, file := range []string{"mamba3_cuda_primitive.cpp", "ir_trainer.cpp"} {
		source := readRepositoryFile(t, file)
		if !strings.Contains(source, `#include "mamba3_debug_policy.h"`) ||
			!strings.Contains(source, "mamba3_cuda_scan_fallback_guidance()") ||
			!strings.Contains(source, "kMamba3DisableCUDAScan") {
			t.Errorf("%s no longer shares the advertised scan-disable policy", file)
		}
		if strings.Contains(source, "set MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE=1 only") {
			t.Errorf("%s advertises an incomplete canonical-training fallback command", file)
		}
	}
	trainer := readRepositoryFile(t, "ir_trainer.cpp")
	if !strings.Contains(trainer, "env_truthy(kMamba3AllowMLXScanFallback)") {
		t.Fatal("training guard no longer checks the advertised opt-in flag")
	}
	if !strings.Contains(trainer, "scan primitive selection is unchanged") {
		t.Fatal("compiled-step fallback must distinguish scan primitive selection")
	}
}
