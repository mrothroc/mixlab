//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

const strictFP32ChildEnv = "MIXLAB_TEST_STRICT_FP32_CHILD"

// runInStrictFP32Process makes a float32 reference test compare strict FP32
// arithmetic. The TF32 policy is frozen per process and allowed by default, so
// the parent reruns the test in a child with TF32 disabled and reports its
// result. It returns true in the child, where the caller runs its body.
func runInStrictFP32Process(t *testing.T) bool {
	t.Helper()
	if os.Getenv(strictFP32ChildEnv) == "1" {
		strict := false
		if p, err := ConfigurePrecision("float32", &strict); err != nil || p.TF32Allowed {
			t.Fatalf("strict FP32 child policy %+v: %v", p, err)
		}
		return true
	}
	cmd := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v", "-test.count=1")
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "MLX_ENABLE_TF32=") && !strings.HasPrefix(e, "NVIDIA_TF32_OVERRIDE=") &&
			!strings.HasPrefix(e, strictFP32ChildEnv+"=") {
			cmd.Env = append(cmd.Env, e)
		}
	}
	cmd.Env = append(cmd.Env, strictFP32ChildEnv+"=1")
	out, err := cmd.CombinedOutput()
	t.Logf("strict FP32 child:\n%s", out)
	if err != nil {
		t.Fatalf("strict FP32 child: %v", err)
	}
	if strings.Contains(string(out), "--- SKIP: "+t.Name()) {
		t.Skip("strict FP32 child skipped")
	}
	return false
}
