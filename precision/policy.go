// Package precision owns the pure resolved execution policy. It does not resolve
// devices or environment settings, or decide admission and resume compatibility.
package precision

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

type Policy struct {
	ComputeDType string `json:"compute_dtype"`
	Backend      string `json:"backend"`
	TF32Allowed  bool   `json:"tf32_allowed"`
}

func (p Policy) Validate() error {
	if p.ComputeDType != "float32" && p.ComputeDType != "bf16" {
		return fmt.Errorf("unsupported compute dtype %q", p.ComputeDType)
	}
	// "unavailable" is what a build without a GPU runtime resolves; it never
	// permits reduced precision.
	if p.Backend != "metal" && p.Backend != "cuda" && p.Backend != "unavailable" {
		return fmt.Errorf("unsupported backend %q", p.Backend)
	}
	if p.Backend == "unavailable" && p.TF32Allowed {
		return fmt.Errorf("a policy without a GPU backend cannot allow TF32")
	}
	return nil
}

func (p Policy) Digest() (string, error) {
	if err := p.Validate(); err != nil {
		return "", err
	}
	b, err := json.Marshal(p)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

func (p Policy) String() string {
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
