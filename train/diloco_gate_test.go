package train

import (
	"github.com/mrothroc/mixlab/arch"
	"strings"
	"testing"
)

func TestDiLoCoRuntimeUnavailable(t *testing.T) {
	cfg := &arch.ArchConfig{}
	cfg.Training.Distributed = &arch.DistributedSpec{Mode: "diloco"}
	_, err := runTrain(cfg, "no-data-should-be-opened", TrainOptions{})
	if err == nil || !strings.Contains(err.Error(), "validation only") {
		t.Fatalf("runtime did not fail closed: %v", err)
	}
}
