//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/logprobs"
)

func TestRunEvalLogprobsParityWithInferenceSession(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	outPath := filepath.Join(dataDir, "eval.bin")
	if err := RunEvalLogprobs(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, outPath); err != nil {
		t.Fatalf("RunEvalLogprobs: %v", err)
	}

	blob, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", outPath, err)
	}
	header, records, err := logprobs.Read(bytes.NewReader(blob))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}
	if header.Magic != logprobs.Magic {
		t.Fatalf("header.Magic = %#x, want %#x", header.Magic, logprobs.Magic)
	}

	sess, err := NewInferenceSession(fixture.configPath, fixture.weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	defer sess.Close()

	directNLLs, err := sess.EvalTokens(fixture.evalTokens)
	if err != nil {
		t.Fatalf("EvalTokens: %v", err)
	}
	if len(records) != len(directNLLs) {
		t.Fatalf("len(records) = %d, want %d", len(records), len(directNLLs))
	}
	for i, rec := range records {
		wantToken := fixture.evalTokens[i+1]
		if rec.TokenID != wantToken {
			t.Fatalf("record[%d].TokenID = %d, want %d", i, rec.TokenID, wantToken)
		}
		if diff := math.Abs(float64(rec.NLL - directNLLs[i])); diff > 1e-6 {
			t.Fatalf("record[%d].NLL = %.8f, want %.8f (diff=%.8g)", i, rec.NLL, directNLLs[i], diff)
		}
	}
}
