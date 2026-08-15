package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEvalModeRejectsNegativeClassificationBatchCapBeforeGPUSetup(t *testing.T) {
	err := runEvalModeWithOptions("", "", "", EvalModeOptions{ValBatches: -1})
	if err == nil || !strings.Contains(err.Error(), "-val-batches") {
		t.Fatalf("error=%v", err)
	}
}

func TestResolveEvalShardPatternUsesExplicitValSplit(t *testing.T) {
	dir := t.TempDir()
	valPattern := filepath.Join(dir, "train_*.bin")
	if err := os.WriteFile(filepath.Join(dir, "train_00000.bin"), []byte("fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	selection, err := resolveEvalShardPattern("", valPattern)
	if err != nil {
		t.Fatal(err)
	}
	if selection.Pattern != valPattern || !selection.Explicit || selection.sourceLabel() != "explicit -val" {
		t.Fatalf("selection=%+v", selection)
	}
}

func TestResolveEvalShardPatternPreservesLegacyTrainToValDerivation(t *testing.T) {
	dir := t.TempDir()
	trainPattern := filepath.Join(dir, "train_*.bin")
	valPattern := filepath.Join(dir, "val_*.bin")
	if err := os.WriteFile(filepath.Join(dir, "val_00000.bin"), []byte("fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	selection, err := resolveEvalShardPattern(trainPattern, "")
	if err != nil {
		t.Fatal(err)
	}
	if selection.Pattern != valPattern || selection.Explicit || selection.sourceLabel() != "derived from -train" {
		t.Fatalf("selection=%+v want pattern=%q", selection, valPattern)
	}
}

func TestResolveEvalShardPatternExplainsMissingDerivedSplit(t *testing.T) {
	dir := t.TempDir()
	trainPattern := filepath.Join(dir, "train_*.bin")
	if err := os.WriteFile(filepath.Join(dir, "train_00000.bin"), []byte("fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := resolveEvalShardPattern(trainPattern, "")
	if err == nil || !strings.Contains(err.Error(), "pass -val") || !strings.Contains(err.Error(), "-val-split 0") {
		t.Fatalf("error=%v", err)
	}
}
