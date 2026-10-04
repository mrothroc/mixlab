//go:build mlx && cgo && (darwin || linux)

package train

import (
	"sort"
	"testing"
	"time"
)

func TestExecutionContractTTTBaseline(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	config, checkpoint, cfg := newTTTMLPInferenceFixture(t)
	start := time.Now()
	session, err := NewTTTMLPInferenceSession(config, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	construction := time.Since(start)
	state, err := session.NewState()
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	// Warm every offset variant before measuring. Decode reads output back and
	// therefore synchronizes completion at each sample's measurement boundary.
	start = time.Now()
	for i := 0; i < 8; i++ {
		if _, err := session.Decode(state, 1); err != nil {
			t.Fatal(err)
		}
	}
	warmup := time.Since(start)
	variants := session.Stats().ProgramVariants
	samples := make([]time.Duration, 32)
	for i := range samples {
		start = time.Now()
		if _, err := session.Decode(state, 1); err != nil {
			t.Fatal(err)
		}
		samples[i] = time.Since(start)
	}
	if session.Stats().ProgramVariants != variants {
		t.Fatal("steady decode constructed new programs/contracts")
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	t.Logf("FP32 B=1 D=%d vocab=%d blocks=%d construction=%s warmup_8_tokens=%s synchronized_decode_n=32 min=%s median=%s p95=%s", cfg.ModelDim, cfg.VocabSize, len(cfg.Blocks), construction, warmup, samples[0], samples[16], samples[30])
}
