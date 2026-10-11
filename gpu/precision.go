package gpu

import "fmt"

// PrecisionPolicy describes permission for reduced-precision FP32 arithmetic,
// not the algorithm selected for every operation. BF16 work may also contain
// float32 operations to which this policy applies.
type PrecisionPolicy struct {
	ComputeDType string `json:"compute_dtype"`
	Backend      string `json:"backend"`
	TF32Allowed  bool   `json:"tf32_allowed"`
}

// ResolvePrecisionPolicy is pure: it does not initialize MLX or mutate env.
func ResolvePrecisionPolicy(dtype, backend string, requested *bool, lookup func(string) (string, bool)) (PrecisionPolicy, error) {
	p := PrecisionPolicy{ComputeDType: dtype, Backend: backend}
	// MLX gates reduced-precision float32 paths with one switch on both GPU
	// backends: TF32 on CUDA, and neural-accelerator matmul/attention on Metal.
	if backend != "cuda" && backend != "metal" {
		if requested != nil {
			return p, fmt.Errorf("training.allow_tf32 requires a GPU backend (backend=%s); omit it on this backend", backend)
		}
		return p, nil
	}
	read := func(name string) (*bool, error) {
		s, ok := lookup(name)
		if !ok {
			return nil, nil
		}
		if s != "0" && s != "1" {
			return nil, fmt.Errorf("%s must be 0 or 1, got %q", name, s)
		}
		v := s == "1"
		return &v, nil
	}
	mlx, err := read("MLX_ENABLE_TF32")
	if err != nil {
		return p, err
	}
	var nvidia *bool
	if backend == "cuda" {
		if nvidia, err = read("NVIDIA_TF32_OVERRIDE"); err != nil {
			return p, err
		}
	}
	p.TF32Allowed = true
	if mlx != nil {
		p.TF32Allowed = *mlx
	}
	if requested != nil {
		if mlx != nil && *mlx != *requested {
			return p, fmt.Errorf("training.allow_tf32 conflicts with MLX_ENABLE_TF32")
		}
		p.TF32Allowed = *requested
	}
	if nvidia != nil && !*nvidia {
		if requested != nil && *requested {
			return p, fmt.Errorf("training.allow_tf32=true conflicts with NVIDIA_TF32_OVERRIDE=0")
		}
		p.TF32Allowed = false
	}
	return p, nil
}

func (p PrecisionPolicy) String() string {
	if p.Backend == "cuda" {
		if p.TF32Allowed {
			return fmt.Sprintf("compute=%s (cuda: TF32 allowed for eligible float32 matmuls/convolutions; MLX_ENABLE_TF32=0 for strict FP32)", p.ComputeDType)
		}
		return fmt.Sprintf("compute=%s (cuda: TF32 disabled; strict FP32 for float32 matmuls/convolutions)", p.ComputeDType)
	}
	if p.Backend == "metal" {
		if p.TF32Allowed {
			return fmt.Sprintf("compute=%s (metal: reduced precision allowed for float32 matmuls/attention on GPUs with neural accelerators; MLX_ENABLE_TF32=0 for strict FP32)", p.ComputeDType)
		}
		return fmt.Sprintf("compute=%s (metal: reduced precision disabled; strict FP32 for float32 matmuls/attention)", p.ComputeDType)
	}
	return fmt.Sprintf("compute=%s (%s: no GPU precision policy)", p.ComputeDType, p.Backend)
}
