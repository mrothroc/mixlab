package train

import (
	"math"
	"strings"
	"testing"

	arch "github.com/mrothroc/mixlab/arch"
)

type fakeSelectedWeightReader struct {
	indexes []int
	weights [][]float32
}

func (f *fakeSelectedWeightReader) ReadWeightsGPU(indexes []int) ([][]float32, error) {
	f.indexes = append([]int(nil), indexes...)
	return f.weights, nil
}

func TestS4DSobolevDiagnosticsUseTargetedReadbackAndEffectiveBeta(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":4,"vocab_size":8,"seq_len":4,
		"blocks":[{"type":"s4d","state_size":4,"sobolev_filter":{"bounds":[-2,2]}}],
		"training":{"batch_tokens":4}
	}`), "sobolev-diagnostics")
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := arch.CollectS4DSobolevWeightBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	reader := &fakeSelectedWeightReader{weights: [][]float32{{-2, -0.5, 0.5, 2}}}
	diagnostics, err := sampleS4DSobolevDiagnostics(reader, bindings)
	if err != nil {
		t.Fatal(err)
	}
	if len(reader.indexes) != 1 || reader.indexes[0] != bindings[0].WeightIndex {
		t.Fatalf("indexes=%v", reader.indexes)
	}
	got := diagnostics.Blocks[0]
	if got.Count != 4 || got.CountAbsGT1 != 2 || got.CountAbsGT2 != 0 || !got.Bounded {
		t.Fatalf("diagnostics=%+v", got)
	}
	if math.Abs(got.Minimum-(-2*math.Tanh(2))) > 1e-6 || math.Abs(got.Maximum-2*math.Tanh(2)) > 1e-6 {
		t.Fatalf("effective range=[%g,%g]", got.Minimum, got.Maximum)
	}
	line := formatS4DSobolevDiagnostics(diagnostics)
	for _, want := range []string{"p01=", "p50=", "p99=", "|beta|>1=2", "nyquist="} {
		if !strings.Contains(line, want) {
			t.Fatalf("line missing %q: %s", want, line)
		}
	}
}
