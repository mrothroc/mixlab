package gpu

import (
	"strings"
	"testing"
)

func TestS4DSobolevCUDAKernelRegistryContract(t *testing.T) {
	list := readRepositoryFile(t, "cuda_kernels/cuda_kernels.list")
	for _, kernel := range []string{
		"s4d_sobolev_filter_forward",
		"s4d_sobolev_filter_vjp_product",
		"s4d_sobolev_filter_vjp_beta",
	} {
		entry := "gpu/cuda_kernels/" + kernel + ".cu"
		if !strings.Contains(list, entry) {
			t.Errorf("CUDA kernel registry is missing %q", entry)
		}
		source := readRepositoryFile(t, "cuda_kernels/"+kernel+".cu")
		if !strings.Contains(source, `extern "C" __global__ void `+kernel) {
			t.Errorf("%s does not export its registered kernel symbol", entry)
		}
	}
}

func TestS4DSobolevCUDAUsesLazyBackwardPrimitive(t *testing.T) {
	primitive := readRepositoryFile(t, "s4d_sobolev_cuda_primitive.cpp")
	for _, required := range []string{
		"S4DSobolevForwardCUDAPrimitive",
		"S4DSobolevBackwardCUDAPrimitive",
		"mx::array::make_arrays",
		"s4d_sobolev_filter_vjp_product",
		"s4d_sobolev_filter_vjp_beta",
		"precompiled_cuda_kernel_available",
	} {
		if !strings.Contains(primitive, required) {
			t.Errorf("S4D Sobolev CUDA primitive is missing %q", required)
		}
	}
	if strings.Contains(primitive, "mx::real(") {
		t.Fatal("S4D Sobolev CUDA VJP must keep complex-to-real reduction inside CUDA")
	}
}
