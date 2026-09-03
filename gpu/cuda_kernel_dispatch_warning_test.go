package gpu

import (
	"strings"
	"testing"
)

// Falling back to NVRTC is correct but silent -- it only shows up as slower
// startup, so a GPU whose architecture is missing from the build can go
// unnoticed indefinitely. These guard the contract that makes the fallback
// self-diagnosing: the generator publishes what it built for, and the
// dispatcher reports that alongside the GPU that could not use it.
func TestCUDAKernelRegistryPublishesBuiltArchitectures(t *testing.T) {
	script := readRepositoryFile(t, "cuda_kernels/generate_registry.sh")
	for _, required := range []string{
		"ARCH_LIST=",
		"kEmbeddedCudaKernelArchitectures",
	} {
		if !strings.Contains(script, required) {
			t.Errorf("generate_registry.sh no longer emits %q", required)
		}
	}
	// The empty-header path compiles on every non-CUDA build, so it has to
	// declare the symbol too or the dispatcher stops building off Linux.
	if strings.Count(script, "kEmbeddedCudaKernelArchitectures") < 2 {
		t.Error("generate_registry.sh must declare the architecture list in both the populated and empty headers")
	}
	placeholder := readRepositoryFile(t, "cuda_kernels/registry_generated.h")
	if !strings.Contains(placeholder, "kEmbeddedCudaKernelArchitectures") {
		t.Error("checked-in registry placeholder does not declare kEmbeddedCudaKernelArchitectures")
	}
}

func TestCUDAKernelDispatchWarnsOnceOnPrecompiledFallback(t *testing.T) {
	dispatch := readRepositoryFile(t, "cuda_kernel_dispatch.cpp")
	for _, required := range []string{
		"PRECOMPILED CUDA KERNELS UNUSABLE ON THIS GPU",
		"kEmbeddedCudaKernelArchitectures",
		"compute_capability_major",
		"g_precompiled_cuda_warning_emitted",
		"generate_registry.sh",
	} {
		if !strings.Contains(dispatch, required) {
			t.Errorf("precompiled-fallback warning no longer includes %q", required)
		}
	}
	if !strings.Contains(dispatch, "first_failure = !g_precompiled_cuda_warning_emitted") {
		t.Error("warning must fire once per process, not once per kernel")
	}
}
