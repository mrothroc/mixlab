// Package strictfp32test runs float32 reference tests under strict FP32.
//
// The GPU TF32 policy is frozen per process and allowed by default, so a test
// that compares against strict-FP32 fixtures reruns itself in a child process
// with TF32 disabled.
package strictfp32test

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

const childEnv = "MIXLAB_TEST_STRICT_FP32_CHILD"

// Run returns true in the strict-FP32 child, after configure has frozen the
// precision policy; the caller then runs its body. In the parent it reruns
// t in a child with TF32 disabled, reports the child's result, and returns
// false.
func Run(t *testing.T, configure func() error) bool {
	t.Helper()
	if os.Getenv(childEnv) == "1" {
		if err := configure(); err != nil {
			t.Fatalf("strict FP32 child policy: %v", err)
		}
		return true
	}
	cmd := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v", "-test.count=1")
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "MLX_ENABLE_TF32=") && !strings.HasPrefix(e, "NVIDIA_TF32_OVERRIDE=") &&
			!strings.HasPrefix(e, childEnv+"=") {
			cmd.Env = append(cmd.Env, e)
		}
	}
	// Set the variable rather than only configuring the bridge: Go reads its own
	// copy of the environment, so code that resolves the policy from a config
	// must see TF32 disabled too.
	cmd.Env = append(cmd.Env, childEnv+"=1", "MLX_ENABLE_TF32=0")
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
