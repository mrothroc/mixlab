package train

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

// TestPrepareScript runs scripts/prepare.py on a small text file and verifies
// the output shards load correctly via data.LoadDataShard.
func TestPrepareScript(t *testing.T) {
	// Check python3 and tokenizers are available.
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found, skipping prepare test")
	}
	cmd := exec.Command("python3", "-c", "import tokenizers")
	if err := cmd.Run(); err != nil {
		t.Skip("python3 tokenizers library not available, skipping prepare test")
	}

	// Find prepare.py relative to the test file.
	scriptPath := filepath.Join("scripts", "prepare.py")
	if _, err := os.Stat(scriptPath); err != nil {
		// Package tests run from ./train, so walk to the repo-root script path.
		scriptPath = filepath.Join("..", "scripts", "prepare.py")
		if _, err := os.Stat(scriptPath); err != nil {
			t.Fatalf("cannot find prepare.py: %v", err)
		}
	}

	// Create temp directory with a small test corpus.
	tmpDir := t.TempDir()
	inputFile := filepath.Join(tmpDir, "test_corpus.txt")
	outputDir := filepath.Join(tmpDir, "shards")

	// Write a small but non-trivial corpus (repeat to get enough tokens).
	corpus := "The quick brown fox jumps over the lazy dog. " +
		"Pack my box with five dozen liquor jugs. " +
		"How vexingly quick daft zebras jump! " +
		"The five boxing wizards jump quickly. "
	// Repeat to ensure we have enough text for tokenization.
	repeated := ""
	for i := 0; i < 100; i++ {
		repeated += corpus
	}
	if err := os.WriteFile(inputFile, []byte(repeated), 0644); err != nil {
		t.Fatalf("writing test corpus: %v", err)
	}

	// Run prepare.py with small vocab and token count per shard.
	prepCmd := exec.Command("python3", scriptPath,
		"--input", inputFile,
		"--output", outputDir,
		"--vocab-size", "256",
		"--val-split", "0.1",
		"--tokens-per-shard", "500",
	)
	prepCmd.Stdout = os.Stdout
	prepCmd.Stderr = os.Stderr

	if err := prepCmd.Run(); err != nil {
		t.Fatalf("prepare.py failed: %v", err)
	}

	// Verify training shards exist and are loadable.
	trainPattern := filepath.Join(outputDir, "train_*.bin")
	trainFiles, err := filepath.Glob(trainPattern)
	if err != nil || len(trainFiles) == 0 {
		t.Fatalf("no training shards found matching %s", trainPattern)
	}

	totalTrainTokens := 0
	for _, f := range trainFiles {
		toks, err := data.LoadDataShard(f)
		if err != nil {
			t.Errorf("loading train shard %s: %v", f, err)
			continue
		}
		if len(toks) == 0 {
			t.Errorf("train shard %s has 0 tokens", f)
		}
		totalTrainTokens += len(toks)
		t.Logf("train shard %s: %d tokens", filepath.Base(f), len(toks))
	}

	// Verify validation shards exist and are loadable.
	valPattern := filepath.Join(outputDir, "val_*.bin")
	valFiles, err := filepath.Glob(valPattern)
	if err != nil || len(valFiles) == 0 {
		t.Fatalf("no validation shards found matching %s", valPattern)
	}

	totalValTokens := 0
	for _, f := range valFiles {
		toks, err := data.LoadDataShard(f)
		if err != nil {
			t.Errorf("loading val shard %s: %v", f, err)
			continue
		}
		if len(toks) == 0 {
			t.Errorf("val shard %s has 0 tokens", f)
		}
		totalValTokens += len(toks)
		t.Logf("val shard %s: %d tokens", filepath.Base(f), len(toks))
	}

	t.Logf("Total: %d train tokens, %d val tokens across %d+%d shards",
		totalTrainTokens, totalValTokens, len(trainFiles), len(valFiles))

	if totalTrainTokens == 0 {
		t.Fatal("no training tokens produced")
	}
	if totalValTokens == 0 {
		t.Fatal("no validation tokens produced")
	}

	// Verify tokenizer.json was saved.
	tokenizerPath := filepath.Join(outputDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); err != nil {
		t.Errorf("tokenizer.json not found: %v", err)
	}

	// Verify the Loader can read the shards end-to-end.
	loader, err := data.NewLoader(trainPattern, 42, 64)
	if err != nil {
		t.Fatalf("NewLoader failed: %v", err)
	}
	x, y, err := loader.NextBatch(128, 64)
	if err != nil {
		t.Fatalf("NextBatch failed: %v", err)
	}
	if len(x) != 128 || len(y) != 128 {
		t.Errorf("batch size mismatch: got x=%d y=%d, want 128", len(x), len(y))
	}
}

// TestFindPrepareScript verifies the script locator logic.
func TestFindPrepareScript(t *testing.T) {
	// When run from repository root/, should find scripts/prepare.py.
	scriptPath, err := findPrepareScript()
	if err != nil {
		t.Skipf("prepare.py not found (expected when running from a different directory): %v", err)
	}
	if _, err := os.Stat(scriptPath); err != nil {
		t.Errorf("findPrepareScript returned %q but file doesn't exist: %v", scriptPath, err)
	}
}
