package train

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
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
	pairPath := filepath.Join(outputDir, "pairs.train.jsonl")
	pairReportPath := filepath.Join(outputDir, "pairs.report.json")
	pairSamplePath := filepath.Join(outputDir, "pairs.samples.jsonl")

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
		"--wwm-compatible-tokenizer",
		"--val-split", "0.1",
		"--tokens-per-shard", "500",
		"--char-vocab-size", "257",
		"--char-max-per-token", "8",
		"--minimal-pair-out", pairPath,
		"--minimal-pair-corruptions", "agreement,attractor,word_order,npi_licensor,quantifier_scope,filler_gap",
		"--minimal-pair-weights", "agreement=1,attractor=2,word_order=1,npi_licensor=1,quantifier_scope=1,filler_gap=1",
		"--minimal-pair-morphology", "induced",
		"--minimal-pair-max-pairs", "12",
		"--minimal-pair-seed", "7",
		"--minimal-pair-report-out", pairReportPath,
		"--minimal-pair-sample-out", pairSamplePath,
		"--minimal-pair-sample-count", "3",
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
	tokenizerBlob, err := os.ReadFile(tokenizerPath)
	if err != nil {
		t.Fatal(err)
	}
	var tokenizerDoc struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(tokenizerBlob, &tokenizerDoc); err != nil {
		t.Fatal(err)
	}
	actualVocab := 0
	for _, id := range tokenizerDoc.Model.Vocab {
		if id+1 > actualVocab {
			actualVocab = id + 1
		}
	}
	wordStart, eligible, scheme, err := mlmWordBoundaryLUTFromTokenizer(tokenizerPath, actualVocab, 4)
	if err != nil {
		t.Fatalf("prepared tokenizer is not WWM-compatible: %v", err)
	}
	if scheme != "bytelevel" || len(wordStart) != actualVocab || eligible[4] != 0 {
		t.Fatalf("prepared WWM metadata scheme=%q starts=%d mask_eligible=%d", scheme, len(wordStart), eligible[4])
	}
	externalOutputDir := filepath.Join(tmpDir, "external-tokenizer-shards")
	externalPrepCmd := exec.Command("python3", scriptPath,
		"--input", inputFile,
		"--output", externalOutputDir,
		"--tokenizer-path", tokenizerPath,
		"--wwm-compatible-tokenizer",
		"--val-split", "0.1",
		"--tokens-per-shard", "10000",
	)
	if output, err := externalPrepCmd.CombinedOutput(); err != nil {
		t.Fatalf("prepare.py with external tokenizer failed: %v\n%s", err, output)
	}
	copiedTokenizerBlob, err := os.ReadFile(filepath.Join(externalOutputDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(copiedTokenizerBlob, tokenizerBlob) {
		t.Fatal("prepare did not preserve the external tokenizer.json byte-for-byte")
	}
	charPath := filepath.Join(outputDir, "char_features.bin")
	charBlob, err := os.ReadFile(charPath)
	if err != nil {
		t.Fatalf("char_features.bin not found: %v", err)
	}
	if len(charBlob) < charFeatureHeaderInts*4 {
		t.Fatalf("char_features.bin too small: %d bytes", len(charBlob))
	}
	header := func(i int) int32 {
		return int32(binary.LittleEndian.Uint32(charBlob[i*4 : i*4+4]))
	}
	if header(0) != charFeatureMagic || header(1) != charFeatureVersion {
		t.Fatalf("bad char feature header magic/version: %d/%d", header(0), header(1))
	}
	if header(3) != 257 || header(4) != 8 {
		t.Fatalf("bad char feature config: char_vocab=%d max=%d", header(3), header(4))
	}
	pairBlob, err := os.ReadFile(pairPath)
	if err != nil {
		t.Fatalf("minimal pair artifact not found: %v", err)
	}
	records, err := decodeMinimalPairJSONL(bytes.NewReader(pairBlob), pairPath, 256)
	if err != nil {
		t.Fatalf("decode minimal pair artifact: %v", err)
	}
	if len(records) == 0 {
		t.Fatal("minimal pair artifact has no records")
	}
	if records[0].Family == "" {
		t.Fatalf("minimal pair record missing family: %+v", records[0])
	}
	reportBlob, err := os.ReadFile(pairReportPath)
	if err != nil {
		t.Fatalf("minimal pair report not found: %v", err)
	}
	var report map[string]any
	if err := json.Unmarshal(reportBlob, &report); err != nil {
		t.Fatalf("decode minimal pair report: %v", err)
	}
	if report["written"].(float64) <= 0 {
		t.Fatalf("minimal pair report has no written records: %v", report)
	}
	if _, ok := report["family_weights"].(map[string]any)["attractor"]; !ok {
		t.Fatalf("minimal pair report missing family weights: %v", report)
	}
	sampleBlob, err := os.ReadFile(pairSamplePath)
	if err != nil {
		t.Fatalf("minimal pair sample dump not found: %v", err)
	}
	if !bytes.Contains(sampleBlob, []byte(`"clean_text"`)) || !bytes.Contains(sampleBlob, []byte(`"corrupt_text"`)) {
		t.Fatalf("minimal pair sample dump missing text fields: %s", sampleBlob)
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
