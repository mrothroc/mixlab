package train

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestParityCompletePairs(t *testing.T) {
	tests := []struct {
		name        string
		tokenCount  int
		batchTokens int
		want        int
	}{
		{name: "empty", tokenCount: 0, batchTokens: 8, want: 0},
		{name: "one token", tokenCount: 1, batchTokens: 8, want: 0},
		{name: "bad batch", tokenCount: 20, batchTokens: 0, want: 0},
		{name: "exact", tokenCount: 17, batchTokens: 8, want: 16},
		{name: "truncates", tokenCount: 26, batchTokens: 8, want: 24},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parityCompletePairs(tt.tokenCount, tt.batchTokens); got != tt.want {
				t.Fatalf("parityCompletePairs()=%d, want %d", got, tt.want)
			}
		})
	}
}

func TestParitySamplePairs(t *testing.T) {
	tests := []struct {
		name        string
		totalPairs  int
		batchTokens int
		requested   int
		want        int
	}{
		{name: "none", totalPairs: 0, batchTokens: 8, requested: 0, want: 0},
		{name: "default one batch", totalPairs: 32, batchTokens: 8, requested: 0, want: 8},
		{name: "rounds up", totalPairs: 32, batchTokens: 8, requested: 9, want: 16},
		{name: "caps at total", totalPairs: 16, batchTokens: 8, requested: 100, want: 16},
		{name: "sub batch request", totalPairs: 16, batchTokens: 8, requested: 3, want: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := paritySamplePairs(tt.totalPairs, tt.batchTokens, tt.requested); got != tt.want {
				t.Fatalf("paritySamplePairs()=%d, want %d", got, tt.want)
			}
		})
	}
}

func TestParseHFParitySummaryIgnoresWarnings(t *testing.T) {
	out := []byte("warning: noisy dependency output\n{\"hf_loss\":1.25,\"max_logit_diff\":0.0002,\"scored_pairs\":32,\"sample_pairs\":8,\"backbone_hidden_size\":64}\n")
	got, err := parseHFParitySummary(out)
	if err != nil {
		t.Fatalf("parseHFParitySummary: %v", err)
	}
	if got.HFLoss != 1.25 || got.MaxLogitDiff != 0.0002 || got.ScoredPairs != 32 || got.SamplePairs != 8 || got.BackboneHiddenSize != 64 {
		t.Fatalf("unexpected summary: %+v", got)
	}
}

func TestParseHFParitySummaryRejectsMissingOrNonFinite(t *testing.T) {
	if _, err := parseHFParitySummary([]byte("no json here\n")); err == nil {
		t.Fatal("expected missing JSON error")
	}
	if _, err := parseHFParitySummary([]byte(`{"hf_loss":1e999,"max_logit_diff":0,"scored_pairs":1,"sample_pairs":1}`)); err == nil {
		t.Fatal("expected non-finite loss error")
	}
}

func TestEvaluateParityThresholds(t *testing.T) {
	summary := hfParitySummary{HFLoss: 1.02, MaxLogitDiff: 0.002}
	if err := evaluateParityThresholds(1.0, summary, 0.03, 0.003); err != nil {
		t.Fatalf("expected thresholds to pass: %v", err)
	}
	if err := evaluateParityThresholds(1.0, summary, 0.01, 0.003); err == nil {
		t.Fatal("expected loss threshold failure")
	}
	if err := evaluateParityThresholds(1.0, summary, 0.03, 0.001); err == nil {
		t.Fatal("expected logit threshold failure")
	}
	if err := evaluateParityThresholds(math.NaN(), summary, 0.03, 0.003); err == nil {
		t.Fatal("expected non-finite native loss failure")
	}
	if err := evaluateParityThresholds(1.0, summary, 0.03, 0); err != nil {
		t.Fatalf("expected disabled logit gate to pass: %v", err)
	}
}

func TestWriteParityFiles(t *testing.T) {
	dir := t.TempDir()
	tokensPath := filepath.Join(dir, "tokens.bin")
	if err := writeParityTokens(tokensPath, []uint16{1, 2, 3}); err != nil {
		t.Fatalf("writeParityTokens: %v", err)
	}
	tokData, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatalf("read tokens: %v", err)
	}
	if got := binary.LittleEndian.Uint32(tokData[0:4]); got != parityTokensMagic {
		t.Fatalf("token magic=%#x, want %#x", got, parityTokensMagic)
	}
	if got := binary.LittleEndian.Uint32(tokData[4:8]); got != parityFileVersion {
		t.Fatalf("token version=%d, want %d", got, parityFileVersion)
	}
	if got := binary.LittleEndian.Uint64(tokData[8:16]); got != 3 {
		t.Fatalf("token count=%d, want 3", got)
	}

	logitsPath := filepath.Join(dir, "logits.bin")
	if err := writeParityLogits(logitsPath, []float32{1, 2, 3, 4, 5, 6}, 2, 3); err != nil {
		t.Fatalf("writeParityLogits: %v", err)
	}
	logitsData, err := os.ReadFile(logitsPath)
	if err != nil {
		t.Fatalf("read logits: %v", err)
	}
	if got := binary.LittleEndian.Uint32(logitsData[0:4]); got != parityLogitsMagic {
		t.Fatalf("logits magic=%#x, want %#x", got, parityLogitsMagic)
	}
	if got := binary.LittleEndian.Uint64(logitsData[8:16]); got != 2 {
		t.Fatalf("logit pairs=%d, want 2", got)
	}
	if got := binary.LittleEndian.Uint64(logitsData[16:24]); got != 3 {
		t.Fatalf("logit vocab=%d, want 3", got)
	}
	if err := writeParityLogits(filepath.Join(dir, "bad.bin"), []float32{1}, 2, 3); err == nil {
		t.Fatal("expected shape mismatch")
	}
}
