package train

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
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
	manifest, err := data.LoadDatasetManifest(filepath.Join(outputDir, data.DatasetManifestFilename))
	if err != nil {
		t.Fatalf("load prepared dataset manifest: %v", err)
	}
	if manifest.Modality != "text" || manifest.Representation != data.DatasetRepresentationDiscreteTokens || manifest.TokenDType != data.DatasetTokenDTypeUint16 {
		t.Fatalf("prepared manifest representation fields=%+v", manifest)
	}
	if trainSplit := manifest.Splits["train"]; trainSplit.Tokens != int64(totalTrainTokens) || trainSplit.Shards != len(trainFiles) {
		t.Fatalf("manifest train split=%+v, want tokens=%d shards=%d", trainSplit, totalTrainTokens, len(trainFiles))
	}
	if valSplit := manifest.Splits["val"]; valSplit.Tokens != int64(totalValTokens) || valSplit.Shards != len(valFiles) {
		t.Fatalf("manifest val split=%+v, want tokens=%d shards=%d", valSplit, totalValTokens, len(valFiles))
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
	if manifest.VocabSize != actualVocab || manifest.Artifacts.Tokenizer != "tokenizer.json" || manifest.SpecialTokenIDs["[MASK]"] != 4 {
		t.Fatalf("prepared manifest tokenizer fields=%+v actual_vocab=%d", manifest, actualVocab)
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

func TestPreparePerRecordTextProducesIndependentMaskedRows(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy, tokenizers").Run(); err != nil {
		t.Skip("python3 numpy/tokenizers libraries not available")
	}
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "records.jsonl")
	outDir := filepath.Join(dir, "prepared")
	input := "{\"text\":\"CCO\"}\n{\"text\":\"NCC\"}\n{\"text\":\"CO\"}\n{\"text\":\"CN\"}\n"
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatal(err)
	}
	err := runPrepare(PrepareOptions{
		Input: inputPath, Output: outDir, InputFormat: "text", VocabSize: 32, ValSplit: 0.25,
		TextFieldName: "text", FramePerRecord: true, RecordSeqLen: 8,
		RecordPADID: 0, RecordBOSID: 1, RecordEOSID: 2, RecordOverflow: "error",
	})
	if err != nil {
		t.Fatalf("prepare per-record text: %v", err)
	}
	manifest, err := data.LoadDatasetManifest(filepath.Join(outDir, data.DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.ShardFormat != data.DatasetShardFormatSequenceV1 || manifest.EffectiveSequenceLayout() != data.DatasetSequenceLayoutOneRecordRow || manifest.RecordSeqLen != 8 {
		t.Fatalf("manifest=%+v", manifest)
	}
	if manifest.Splits["train"].Sequences != 3 || manifest.Splits["val"].Sequences != 1 {
		t.Fatalf("record split=%+v", manifest.Splits)
	}
	trainFiles, _ := filepath.Glob(filepath.Join(outDir, "train_*.bin"))
	if len(trainFiles) == 0 {
		t.Fatal("no record-oriented train shard")
	}
	records, err := data.LoadSequenceShard(trainFiles[0])
	if err != nil {
		t.Fatal(err)
	}
	loader, err := data.NewLoaderWithOptions(filepath.Join(outDir, "train_*.bin"), 7, data.LoaderOptions{NoShardShuffle: true})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := loader.NextBatchDetailed(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	contentLen := len(records[0])
	if batch.X[0] != 1 || batch.X[contentLen+1] != 2 {
		t.Fatalf("row framing x=%v record=%v", batch.X, records[0])
	}
	for i, token := range records[0] {
		if batch.X[i+1] != int(token) {
			t.Fatalf("record token %d=%d, row=%v", i, token, batch.X)
		}
	}
	for i, active := range batch.LossMask {
		want := float32(0)
		if i <= contentLen {
			want = 1
		}
		if active != want {
			t.Fatalf("loss_mask[%d]=%g want=%g (content_len=%d)", i, active, want, contentLen)
		}
	}
}

func TestPrepareLabeledJSONLProducesClassificationManifestAndAtomicLabels(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy, tokenizers").Run(); err != nil {
		t.Skip("python3 numpy/tokenizers libraries not available")
	}
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "classification.jsonl")
	outDir := filepath.Join(dir, "prepared")
	input := "" +
		"{\"id\":\"a\",\"sequence\":\"alpha one\",\"label\":0}\n" +
		"{\"id\":\"b\",\"sequence\":\"beta two\",\"label\":1}\n" +
		"{\"id\":\"c\",\"sequence\":\"alpha three\",\"label\":0}\n" +
		"{\"id\":\"d\",\"sequence\":\"beta four\",\"label\":1}\n" +
		"{\"id\":\"e\",\"sequence\":\"alpha five\",\"label\":0}\n" +
		"{\"id\":\"f\",\"sequence\":\"beta six\",\"label\":1}\n"
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatal(err)
	}
	err := runPrepare(PrepareOptions{
		Input: inputPath, Output: outDir, InputFormat: "text", VocabSize: 32, ValSplit: 0.34,
		TextFieldName: "sequence", LabelFieldName: "label", RecordSeqLen: 12,
		RecordPADID: 0, RecordBOSID: 1, RecordEOSID: 2, RecordOverflow: "truncate",
	})
	if err != nil {
		t.Fatalf("prepare labeled JSONL: %v", err)
	}
	manifest, err := data.LoadDatasetManifest(filepath.Join(outDir, data.DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.ShardFormat != data.DatasetShardFormatLabeledSequenceV1 ||
		manifest.Task == nil || manifest.Task.NumLabels != 2 ||
		manifest.Task.Type != data.DatasetTaskSingleLabelClassification {
		t.Fatalf("classification manifest=%+v", manifest)
	}
	if !reflect.DeepEqual(manifest.Splits["train"].ClassCounts, map[string]int64{"0": 2, "1": 2}) ||
		!reflect.DeepEqual(manifest.Splits["val"].ClassCounts, map[string]int64{"0": 1, "1": 1}) {
		t.Fatalf("class counts=%+v", manifest.Splits)
	}
	trainFiles, _ := filepath.Glob(filepath.Join(outDir, "train_*.bin"))
	records, labels, err := data.LoadLabeledSequenceShard(trainFiles[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 4 || !reflect.DeepEqual(labels, []int32{0, 1, 0, 1}) {
		t.Fatalf("prepared records/labels=%d/%v", len(records), labels)
	}
}

func TestPreparePerRecordOverflowPolicies(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	scriptPath, err := filepath.Abs(filepath.Join("..", "scripts", "prepare.py"))
	if err != nil {
		t.Fatal(err)
	}
	recordScriptPath := filepath.Join(filepath.Dir(scriptPath), "prepare_records.py")
	program := `import runpy
m = runpy.run_path(` + strconv.Quote(recordScriptPath) + `, run_name="prepare_records_module")
class Enc:
    def __init__(self, ids): self.ids = ids
class Tok:
    def encode(self, text, add_special_tokens=False): return Enc(list(range(len(text))))
records, stats = m["tokenize_text_records"](Tok(), ["abcdef", "xy"], 6, "drop", "train")
assert len(records) == 1 and stats["dropped"] == 1 and list(records[0][1]) == [0, 1]
records, stats = m["tokenize_text_records"](Tok(), ["abcdef", "xy"], 6, "truncate", "train")
assert len(records) == 2 and stats["truncated"] == 1 and len(records[0][1]) == 4
try:
    m["tokenize_text_records"](Tok(), ["abcdef"], 6, "error", "train")
except ValueError as exc:
    assert "permits 4" in str(exc)
else:
    raise AssertionError("error overflow policy accepted an overlong record")
`
	if output, err := exec.Command("python3", "-c", program).CombinedOutput(); err != nil {
		t.Fatalf("record overflow helper: %v\n%s", err, output)
	}
}

func TestRunPrepareValidatesPerRecordContractBeforeLaunchingPython(t *testing.T) {
	base := PrepareOptions{
		Input: "records.jsonl", Output: "out", InputFormat: "text", FramePerRecord: true,
		RecordSeqLen: 8, RecordPADID: 0, RecordBOSID: 1, RecordEOSID: 2, RecordOverflow: "error",
	}
	tests := []struct {
		name string
		edit func(*PrepareOptions)
		want string
	}{
		{name: "format", edit: func(o *PrepareOptions) { o.InputFormat = "fasta" }, want: "input-format=text"},
		{name: "seq len", edit: func(o *PrepareOptions) { o.RecordSeqLen = 2 }, want: "record-seq-len >= 3"},
		{name: "missing id", edit: func(o *PrepareOptions) { o.RecordEOSID = -1 }, want: "non-negative"},
		{name: "duplicate ids", edit: func(o *PrepareOptions) { o.RecordEOSID = 1 }, want: "must be distinct"},
		{name: "overflow", edit: func(o *PrepareOptions) { o.RecordOverflow = "wrap" }, want: "record-overflow"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := base
			tt.edit(&opts)
			err := runPrepare(opts)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}
}

func TestPrepareFASTAProducesInspectableBoundarySafeDataset(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	scriptPath := filepath.Join("..", "scripts", "prepare.py")
	if _, err := os.Stat(scriptPath); err != nil {
		scriptPath = filepath.Join("scripts", "prepare.py")
	}
	dir := t.TempDir()
	fasta := filepath.Join(dir, "tiny.fasta")
	outDir := filepath.Join(dir, "prepared")
	input := ">contig_a first\nACGTN\n>contig_b\nRYYA\n>contig_c\nGGTT\n"
	if err := os.WriteFile(fasta, []byte(input), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("python3", scriptPath,
		"--input", fasta, "--output", outDir, "--input-format", "fasta",
		"--nucleotide-alphabet", "dna", "--nucleotide-ambiguous-symbols", "N,R",
		"--nucleotide-invalid-symbol-policy", "error", "--val-split", "0.34",
		"--tokens-per-shard", "6")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("prepare FASTA: %v\n%s", err, output)
	}
	manifest, err := data.LoadDatasetManifest(filepath.Join(outDir, data.DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Modality != "nucleotide" || manifest.ShardFormat != data.DatasetShardFormatSequenceV1 || manifest.VocabSize != 11 {
		t.Fatalf("manifest=%+v", manifest)
	}
	if manifest.Splits["train"].Sequences != 2 || manifest.Splits["val"].Sequences != 1 {
		t.Fatalf("split metadata=%+v", manifest.Splits)
	}
	vocab, err := data.LoadNucleotideVocabulary(filepath.Join(outDir, manifest.Artifacts.Vocabulary))
	if err != nil {
		t.Fatal(err)
	}
	trainFiles, _ := filepath.Glob(filepath.Join(outDir, "train_*.bin"))
	var decoded []string
	for _, path := range trainFiles {
		records, err := data.LoadSequenceShard(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, record := range records {
			ids := make([]int, len(record))
			for i, id := range record {
				ids[i] = int(id)
			}
			sequence, err := vocab.Decode(ids)
			if err != nil {
				t.Fatal(err)
			}
			decoded = append(decoded, sequence)
		}
	}
	if !reflect.DeepEqual(decoded, []string{"ACGTN", "RYYA"}) {
		t.Fatalf("decoded FASTA train records=%v", decoded)
	}
	loader, err := data.NewLoaderWithOptions(filepath.Join(outDir, "train_*.bin"), 1, data.LoaderOptions{NoShardShuffle: true})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := loader.NextBatchDetailed(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(batch.X)-1; i++ {
		if batch.SegmentIDs[i] != batch.SegmentIDs[i+1] && batch.LossMask[i] != 0 {
			t.Fatalf("cross-contig target at position %d: x=%v y=%v mask=%v segments=%v", i, batch.X, batch.Y, batch.LossMask, batch.SegmentIDs)
		}
	}
}

func TestPrepareLabeledFASTAKeepsTSVLabelsAtomic(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	dir := t.TempDir()
	fasta := filepath.Join(dir, "tiny.fasta")
	labelsPath := filepath.Join(dir, "labels.tsv")
	outDir := filepath.Join(dir, "prepared")
	if err := os.WriteFile(fasta, []byte(">a\nACGT\n>b\nTGCA\n>c\nAAAA\n>d\nCCCC\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(labelsPath, []byte("a\t0\nb\t1\nc\t0\nd\t1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := runPrepare(PrepareOptions{
		Input: fasta, Output: outDir, InputFormat: "fasta", ValSplit: 0.25,
		LabelFile: labelsPath, RecordSeqLen: 8, RecordOverflow: "error",
		NucleotideAlphabet: "dna", NucleotideAmbiguous: "N", NucleotideInvalidPolicy: "error",
	})
	if err != nil {
		t.Fatalf("prepare labeled FASTA: %v", err)
	}
	manifest, err := data.LoadDatasetManifest(filepath.Join(outDir, data.DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.ShardFormat != data.DatasetShardFormatLabeledSequenceV1 || manifest.Modality != "nucleotide" ||
		manifest.Task == nil || manifest.Task.NumLabels != 2 {
		t.Fatalf("manifest=%+v", manifest)
	}
	trainFiles, _ := filepath.Glob(filepath.Join(outDir, "train_*.bin"))
	_, labels, err := data.LoadLabeledSequenceShard(trainFiles[0])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(labels, []int32{0, 1}) {
		t.Fatalf("train labels=%v want [0 1]", labels)
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
