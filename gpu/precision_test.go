package gpu

import (
	"strings"
	"testing"
)

func TestPrecisionPolicy(t *testing.T) {
	yes, no := true, false
	for _, tc := range []struct {
		name, backend string
		request       *bool
		env           map[string]string
		want          bool
		err           string
	}{
		{"cuda_default", "cuda", nil, nil, true, ""},
		{"strict", "cuda", &no, nil, false, ""},
		{"enabled", "cuda", &yes, nil, true, ""},
		{"legacy_optout", "cuda", nil, map[string]string{"MLX_ENABLE_TF32": "0"}, false, ""},
		{"nvidia_optout", "cuda", nil, map[string]string{"NVIDIA_TF32_OVERRIDE": "0", "MLX_ENABLE_TF32": "1"}, false, ""},
		{"nvidia_one_not_force", "cuda", &no, map[string]string{"NVIDIA_TF32_OVERRIDE": "1"}, false, ""},
		{"mlx_conflict", "cuda", &no, map[string]string{"MLX_ENABLE_TF32": "1"}, false, "conflicts"},
		{"nvidia_conflict", "cuda", &yes, map[string]string{"NVIDIA_TF32_OVERRIDE": "0"}, false, "conflicts"},
		{"invalid_env", "cuda", nil, map[string]string{"MLX_ENABLE_TF32": "true"}, false, "must be 0 or 1"},
		{"empty_env", "cuda", nil, map[string]string{"NVIDIA_TF32_OVERRIDE": ""}, false, "must be 0 or 1"},
		// Apple GPUs with neural accelerators route float32 matmul/attention
		// through reduced precision under the same MLX switch.
		{"metal_default", "metal", nil, nil, true, ""},
		{"metal_strict", "metal", &no, nil, false, ""},
		{"metal_env_optout", "metal", nil, map[string]string{"MLX_ENABLE_TF32": "0"}, false, ""},
		{"metal_conflict", "metal", &yes, map[string]string{"MLX_ENABLE_TF32": "0"}, false, "conflicts"},
		{"metal_ignores_nvidia", "metal", nil, map[string]string{"NVIDIA_TF32_OVERRIDE": "junk"}, true, ""},
		{"cpu_default", "unavailable", nil, nil, false, ""},
		{"cpu_explicit", "unavailable", &yes, nil, false, "requires a GPU"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, dtype := range []string{"float32", "bf16"} {
				p, err := ResolvePrecisionPolicy(dtype, tc.backend, tc.request, func(k string) (string, bool) { v, ok := tc.env[k]; return v, ok })
				if tc.err != "" {
					if err == nil || !strings.Contains(err.Error(), tc.err) {
						t.Fatalf("got %v want %s", err, tc.err)
					}
					continue
				}
				if err != nil || p.TF32Allowed != tc.want || p.ComputeDType != dtype || p.Backend != tc.backend {
					t.Fatalf("policy=%+v err=%v", p, err)
				}
				if !strings.Contains(p.String(), dtype) || !strings.Contains(p.String(), tc.backend) {
					t.Fatal(p.String())
				}
			}
		})
	}
}

// The pure precision contract must accept every policy the resolver emits,
// including the portable build's backend, so digests work on every build.
func TestResolvedPoliciesSatisfyContract(t *testing.T) {
	no := false
	for _, backend := range []string{"cuda", "metal", "unavailable", PrecisionBackend()} {
		for _, dtype := range []string{"float32", "bf16"} {
			for _, request := range []*bool{nil, &no} {
				if backend != "cuda" && backend != "metal" && request != nil {
					continue
				}
				p, err := ResolvePrecisionPolicy(dtype, backend, request, func(string) (string, bool) { return "", false })
				if err != nil {
					t.Fatal(err)
				}
				if err = p.Validate(); err != nil {
					t.Fatalf("%+v: %v", p, err)
				}
				if _, err = p.Digest(); err != nil {
					t.Fatalf("%+v: %v", p, err)
				}
			}
		}
	}
}
