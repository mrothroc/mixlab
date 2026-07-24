package train

import (
	"strings"
	"testing"
)

func TestRCEquivarianceRejectsHFExportBeforeWriting(t *testing.T) {
	cfg := objectiveTestConfig()
	cfg.RCEquivariant = true
	err := validateHFExportConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), "rc_equivariant") {
		t.Fatalf("HF export error=%v", err)
	}
}
