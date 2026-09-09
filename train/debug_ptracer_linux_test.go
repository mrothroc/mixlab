//go:build linux

package train

import "testing"

func TestDebugPtracerRejectsBroadOrUnrelatedAuthorization(t *testing.T) {
	for _, value := range []string{"-1", "0", "1", "invalid", "999999999"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("MIXLAB_DEBUG_PTRACER_PID", value)
			if err := allowDebugParent(); err == nil {
				t.Fatal("must not authorize arbitrary debuggers")
			}
		})
	}
	t.Setenv("MIXLAB_DEBUG_PTRACER_PID", "")
	if err := allowDebugParent(); err != nil {
		t.Fatal(err)
	}
}
