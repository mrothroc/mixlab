//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/internal/strictfp32test"
)

// runInStrictFP32Process makes a float32 reference test compare strict FP32
// arithmetic; it returns true in the child, where the caller runs its body.
func runInStrictFP32Process(t *testing.T) bool {
	t.Helper()
	return strictfp32test.Run(t, func() error {
		strict := false
		p, err := gpu.ConfigurePrecision("float32", &strict)
		if err == nil && p.TF32Allowed {
			err = fmt.Errorf("policy %+v still allows TF32", p)
		}
		return err
	})
}
