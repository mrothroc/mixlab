//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestInferenceSessionEvalTokens(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	sess, err := NewInferenceSession(fixture.configPath, fixture.weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	defer sess.Close()

	if got := sess.Config(); got == nil {
		t.Fatal("Config() = nil")
	} else {
		if got.Name != fixture.cfg.Name {
			t.Fatalf("Config().Name = %q, want %q", got.Name, fixture.cfg.Name)
		}
		if got.SeqLen != fixture.cfg.SeqLen {
			t.Fatalf("Config().SeqLen = %d, want %d", got.SeqLen, fixture.cfg.SeqLen)
		}
	}

	nlls, err := sess.EvalTokens(fixture.evalTokens[:fixture.cfg.Training.BatchTokens+1])
	if err != nil {
		t.Fatalf("EvalTokens: %v", err)
	}
	if len(nlls) != fixture.cfg.Training.BatchTokens {
		t.Fatalf("len(nlls) = %d, want %d", len(nlls), fixture.cfg.Training.BatchTokens)
	}
	for i, nll := range nlls {
		if math.IsNaN(float64(nll)) || math.IsInf(float64(nll), 0) {
			t.Fatalf("nll[%d] is non-finite: %v", i, nll)
		}
	}
}

func TestInferenceSessionCloseIdempotent(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	sess, err := NewInferenceSession(fixture.configPath, fixture.weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("Close first call: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("Close second call: %v", err)
	}
}

func TestInferenceSessionEvalAfterClose(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	sess, err := NewInferenceSession(fixture.configPath, fixture.weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := sess.EvalTokens(fixture.evalTokens[:fixture.cfg.Training.BatchTokens+1]); err == nil {
		t.Fatal("EvalTokens after Close succeeded, want error")
	}
}

type inferenceSessionFixture struct {
	cfg         *ArchConfig
	configPath  string
	weightsPath string
	evalTokens  []uint16
}

func newInferenceSessionFixture(t *testing.T) inferenceSessionFixture {
	t.Helper()

	dir := t.TempDir()
	cfg := &ArchConfig{
		Name:      "inference_session_test",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
			{Type: "swiglu"},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Steps = 1
	cfg.Training.LR = 1e-3
	cfg.Training.Seed = 7
	cfg.Training.BatchTokens = 8

	configPath := filepath.Join(dir, "config.json")
	configBlob, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatalf("Marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, configBlob, 0o644); err != nil {
		t.Fatalf("Write config: %v", err)
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	weightsPath := filepath.Join(dir, "weights.safetensors")
	if err := exportSafetensors(weightsPath, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}

	return inferenceSessionFixture{
		cfg:         cfg,
		configPath:  configPath,
		weightsPath: weightsPath,
		evalTokens:  []uint16{1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 0, 1, 2, 3, 4},
	}
}

func writeInferenceShard(t *testing.T, path string, tokens []uint16) {
	t.Helper()

	header := make([]int32, testHeaderInts)
	header[0] = testShardMagic
	header[1] = testShardVersion
	header[2] = int32(len(tokens))
	buf := make([]byte, testHeaderInts*4+len(tokens)*2)
	for i, v := range header {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	off := testHeaderInts * 4
	for i, tok := range tokens {
		binary.LittleEndian.PutUint16(buf[off+i*2:], tok)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("Write shard %s: %v", path, err)
	}
}

func writeInferenceLUTs(t *testing.T, dir string, vocab int) {
	t.Helper()

	base := make([]byte, vocab*2)
	for i := 0; i < vocab; i++ {
		binary.LittleEndian.PutUint16(base[i*2:], 1)
	}
	if err := os.WriteFile(filepath.Join(dir, "bytes_per_token.bin"), base, 0o644); err != nil {
		t.Fatalf("Write bytes_per_token.bin: %v", err)
	}
	leading := make([]byte, vocab)
	if err := os.WriteFile(filepath.Join(dir, "has_leading_space.bin"), leading, 0o644); err != nil {
		t.Fatalf("Write has_leading_space.bin: %v", err)
	}
	boundary := make([]byte, vocab)
	for i := range boundary {
		boundary[i] = 1
	}
	if err := os.WriteFile(filepath.Join(dir, "is_boundary_token.bin"), boundary, 0o644); err != nil {
		t.Fatalf("Write is_boundary_token.bin: %v", err)
	}
}
