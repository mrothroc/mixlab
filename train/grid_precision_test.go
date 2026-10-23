package train

import (
	"strings"
	"testing"
)

func TestTorchStatePrecisionNotice(t *testing.T) {
	yes, no := true, false
	for _, value := range []*bool{nil, &no, &yes} {
		cfg := &ArchConfig{Name: "grid"}
		cfg.Training.AllowTF32 = value
		got := torchStatePrecisionNotice(cfg)
		want := "unset (runtime default/environment)"
		if value != nil {
			want = "false"
			if *value {
				want = "true"
			}
		}
		for _, s := range []string{"storage=float32", "configured_allow_tf32=" + want, "data-only export; no GPU compute executed"} {
			if !strings.Contains(got, s) {
				t.Fatalf("notice %q missing %q", got, s)
			}
		}
	}
}
