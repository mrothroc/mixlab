package gpu

import (
	"os"
	"strings"
	"testing"
)

func TestTTTMLPCUDAKernelIsInBuildRegistry(t *testing.T) {
	list, err := os.ReadFile("cuda_kernels/cuda_kernels.list")
	if err != nil {
		t.Fatalf("read CUDA kernel list: %v", err)
	}
	if !strings.Contains(string(list), "gpu/cuda_kernels/ttt_mlp_causal_conv.cu") {
		t.Fatal("CUDA kernel list is missing ttt_mlp_causal_conv.cu")
	}
	source, err := os.ReadFile("cuda_kernels/ttt_mlp_causal_conv.cu")
	if err != nil {
		t.Fatalf("read TTT CUDA kernel: %v", err)
	}
	for _, required := range []string{
		`extern "C" __global__ void ttt_mlp_causal_conv`,
		"const float* history",
		"const float* weight",
		"batch_size * sequence_width",
	} {
		if !strings.Contains(string(source), required) {
			t.Fatalf("TTT CUDA kernel is missing %q", required)
		}
	}
	primitive, err := os.ReadFile("ttt_mlp_cuda_primitive.cpp")
	if err != nil {
		t.Fatalf("read TTT CUDA primitive: %v", err)
	}
	for _, required := range []string{
		"mx::contiguous(x)",
		"mx::contiguous(history)",
		"mx::contiguous(weight)",
	} {
		if !strings.Contains(string(primitive), required) {
			t.Fatalf("TTT CUDA primitive is missing %q", required)
		}
	}
}
