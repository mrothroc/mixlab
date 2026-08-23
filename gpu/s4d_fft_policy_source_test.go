package gpu

import (
	"strings"
	"testing"
)

func TestS4DBidirectionalFFTPolicyContract(t *testing.T) {
	source := readRepositoryFile(t, "ir.cpp")
	start := strings.Index(source, "mx::array s4d_fft_convolution_bidirectional(")
	if start < 0 {
		t.Fatal("could not isolate s4d_fft_convolution_bidirectional")
	}
	end := strings.Index(source[start:], "mx::array s4d_recurrent(")
	if end < 0 {
		t.Fatal("could not isolate s4d_fft_convolution_bidirectional")
	}
	body := source[start : start+end]
	for _, required := range []string{
		"s4d_linear_convolution_fft_length(T)",
		": 2 * T",
		"mx::slice(kernel, {0, 0}, {D, T})",
		"mx::slice(kernel, {0, T}, {D, 2 * T})",
		"mx::zeros({D, fft_len - 2 * T}",
		"mx::fft::rfft(x, fft_len, 1)",
		"mx::fft::rfft(fft_kernel, fft_len, 1)",
		"mx::fft::irfft(product, fft_len, 1)",
	} {
		if !strings.Contains(body, required) {
			t.Errorf("bidirectional S4D FFT path is missing %q", required)
		}
	}
	for _, forbidden := range []string{
		"mx::fft::rfft(x, 2 * T, 1)",
		"mx::fft::rfft(fft_kernel, 2 * T, 1)",
		"mx::fft::irfft(product, 2 * T, 1)",
		"mx::zeros({D, T}",
	} {
		if strings.Contains(body, forbidden) {
			t.Errorf("bidirectional S4D FFT path retains obsolete expression %q", forbidden)
		}
	}
}

func TestS4DBidirectionalMetalKernelHasExplicitBackward(t *testing.T) {
	source := readRepositoryFile(t, "s4d_kernel_metal_primitive.cpp")
	for _, required := range []string{
		"s4d_bidirectional_kernel_forward_metal",
		"s4d_bidirectional_kernel_backward_metal",
		"S4DBidirectionalKernelForwardMetalPrimitive",
		"S4DBidirectionalKernelBackwardMetalPrimitive",
		"mx::custom_vjp",
		"MIXLAB_S4D_DISABLE_METAL_KERNEL_PRIMITIVE",
	} {
		if !strings.Contains(source, required) {
			t.Errorf("S4D Metal kernel primitive is missing %q", required)
		}
	}
}

func TestS4DBidirectionalSharesKernelPowers(t *testing.T) {
	source := readRepositoryFile(t, "ir.cpp")
	start := strings.Index(source, "S4DResult s4d_forward_advanced(")
	if start < 0 {
		t.Fatal("could not isolate s4d_forward_advanced")
	}
	end := strings.Index(source[start:], "struct TTTMLPScanResult")
	if end < 0 {
		t.Fatal("could not isolate s4d_forward_advanced")
	}
	body := source[start : start+end]
	if got := strings.Count(body, "s4d_materialize_kernel_powers(discrete, T)"); got != 1 {
		t.Fatalf("bidirectional S4D power-basis builds=%d want 1", got)
	}
	if got := strings.Count(body, "s4d_materialize_kernel_from_powers("); got != 2 {
		t.Fatalf("bidirectional S4D directional projections=%d want 2", got)
	}
}
