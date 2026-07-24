package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

// PrepareOptions holds flags for the prepare command.
type PrepareOptions struct {
	Input                     string
	Output                    string
	InputFormat               string
	VocabSize                 int
	ValSplit                  float64
	TokenizerPath             string
	WWMCompatibleTokenizer    bool
	TextFieldName             string
	LabelFieldName            string
	LabelFile                 string
	FramePerRecord            bool
	RecordSeqLen              int
	RecordPADID               int
	RecordBOSID               int
	RecordEOSID               int
	RecordOverflow            string
	CharVocabSize             int
	CharMaxPerToken           int
	MinimalPairOut            string
	MinimalPairCorruptions    string
	MinimalPairWeights        string
	MinimalPairMorphology     string
	MinimalPairMaxPairs       int
	MinimalPairSeed           int
	MinimalPairReportOut      string
	MinimalPairSampleOut      string
	MinimalPairSampleCount    int
	NucleotideAlphabet        string
	NucleotideAmbiguous       string
	NucleotideInvalidPolicy   string
	NucleotideFraming         string
	NucleotideStreamSeparator string
}

// runPrepare shells out to scripts/prepare.py to tokenize raw text into binary shards.
func runPrepare(opts PrepareOptions) error {
	if opts.Input == "" {
		return fmt.Errorf("-input is required for prepare mode; pass a text file, JSONL, or directory, e.g.: mixlab -mode prepare -input corpus.jsonl -prepare-output-dir data/")
	}
	if opts.Output == "" {
		return fmt.Errorf("-prepare-output-dir (or legacy -output) is required for prepare mode; pass an output directory, e.g.: mixlab -mode prepare -input corpus.jsonl -prepare-output-dir data/")
	}
	inputFormat := strings.ToLower(strings.TrimSpace(opts.InputFormat))
	nucleotideFraming := strings.ToLower(strings.TrimSpace(opts.NucleotideFraming))
	if nucleotideFraming == "" {
		nucleotideFraming = "record"
	}
	switch nucleotideFraming {
	case "record", "stream":
	default:
		return fmt.Errorf("invalid -nucleotide-framing=%q (want record or stream)", opts.NucleotideFraming)
	}
	nucleotideStreamSeparator := strings.ToLower(strings.TrimSpace(opts.NucleotideStreamSeparator))
	if nucleotideStreamSeparator == "" {
		nucleotideStreamSeparator = "eos"
	}
	switch nucleotideStreamSeparator {
	case "eos", "none":
	default:
		return fmt.Errorf("invalid -nucleotide-stream-separator=%q (want eos or none)", opts.NucleotideStreamSeparator)
	}
	if inputFormat != "fasta" && nucleotideFraming != "record" {
		return fmt.Errorf("-nucleotide-framing=%s requires -input-format=fasta", nucleotideFraming)
	}
	if opts.LabelFile != "" && nucleotideFraming != "record" {
		return fmt.Errorf("-label-file requires -nucleotide-framing=record")
	}
	if opts.LabelFieldName != "" && opts.LabelFile != "" {
		return fmt.Errorf("-label-field and -label-file are mutually exclusive")
	}
	if opts.LabelFieldName != "" && inputFormat != "text" {
		return fmt.Errorf("-label-field requires -input-format=text JSONL")
	}
	if opts.LabelFile != "" && inputFormat != "fasta" {
		return fmt.Errorf("-label-file requires -input-format=fasta")
	}
	recordIDsRequired := opts.FramePerRecord || opts.LabelFieldName != ""
	recordLengthRequired := recordIDsRequired || opts.LabelFile != ""
	if opts.FramePerRecord && inputFormat != "text" && opts.LabelFile == "" {
		return fmt.Errorf("-frame-per-record requires -input-format=text")
	}
	if recordLengthRequired {
		if opts.RecordSeqLen < 3 {
			return fmt.Errorf("record-oriented preparation requires -record-seq-len >= 3")
		}
	}
	if recordIDsRequired {
		if inputFormat != "text" {
			return fmt.Errorf("-frame-per-record requires -input-format=text")
		}
		ids := []int{opts.RecordPADID, opts.RecordBOSID, opts.RecordEOSID}
		if ids[0] < 0 || ids[1] < 0 || ids[2] < 0 {
			return fmt.Errorf("record-oriented text preparation requires non-negative -record-pad-id, -record-bos-id, and -record-eos-id")
		}
		if ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2] {
			return fmt.Errorf("-record-pad-id, -record-bos-id, and -record-eos-id must be distinct")
		}
	}
	if recordLengthRequired {
		switch opts.RecordOverflow {
		case "", "error", "drop", "truncate":
		default:
			return fmt.Errorf("invalid -record-overflow=%q (want error, drop, or truncate)", opts.RecordOverflow)
		}
	}

	scriptPath, err := findPrepareScript()
	if err != nil {
		return err
	}

	args := []string{
		scriptPath,
		"--input", opts.Input,
		"--output", opts.Output,
		"--vocab-size", fmt.Sprintf("%d", opts.VocabSize),
		"--val-split", fmt.Sprintf("%g", opts.ValSplit),
	}
	if opts.InputFormat != "" {
		args = append(args, "--input-format", opts.InputFormat)
	}
	if opts.NucleotideAlphabet != "" {
		args = append(args, "--nucleotide-alphabet", opts.NucleotideAlphabet)
	}
	if opts.NucleotideAmbiguous != "" {
		args = append(args, "--nucleotide-ambiguous-symbols", opts.NucleotideAmbiguous)
	}
	if opts.NucleotideInvalidPolicy != "" {
		args = append(args, "--nucleotide-invalid-symbol-policy", opts.NucleotideInvalidPolicy)
	}
	args = append(args, "--nucleotide-framing", nucleotideFraming)
	args = append(args, "--nucleotide-stream-separator", nucleotideStreamSeparator)
	if opts.TokenizerPath != "" {
		args = append(args, "--tokenizer-path", opts.TokenizerPath)
	}
	if opts.WWMCompatibleTokenizer {
		args = append(args, "--wwm-compatible-tokenizer")
	}
	if opts.TextFieldName != "" && opts.TextFieldName != "text" {
		args = append(args, "--text-field", opts.TextFieldName)
	}
	if opts.LabelFieldName != "" {
		args = append(args, "--label-field", opts.LabelFieldName)
	}
	if opts.LabelFile != "" {
		args = append(args, "--label-file", opts.LabelFile)
	}
	if opts.FramePerRecord {
		args = append(args, "--frame-per-record")
	}
	if recordIDsRequired {
		args = append(args,
			"--record-seq-len", fmt.Sprintf("%d", opts.RecordSeqLen),
			"--record-pad-id", fmt.Sprintf("%d", opts.RecordPADID),
			"--record-bos-id", fmt.Sprintf("%d", opts.RecordBOSID),
			"--record-eos-id", fmt.Sprintf("%d", opts.RecordEOSID),
		)
	} else if opts.LabelFile != "" {
		args = append(args, "--record-seq-len", fmt.Sprintf("%d", opts.RecordSeqLen))
	}
	if recordLengthRequired {
		if opts.RecordOverflow != "" {
			args = append(args, "--record-overflow", opts.RecordOverflow)
		}
	}
	if opts.CharVocabSize > 0 {
		args = append(args, "--char-vocab-size", fmt.Sprintf("%d", opts.CharVocabSize))
	}
	if opts.CharMaxPerToken > 0 {
		args = append(args, "--char-max-per-token", fmt.Sprintf("%d", opts.CharMaxPerToken))
	}
	if opts.MinimalPairOut != "" {
		args = append(args, "--minimal-pair-out", opts.MinimalPairOut)
	}
	if opts.MinimalPairCorruptions != "" {
		args = append(args, "--minimal-pair-corruptions", opts.MinimalPairCorruptions)
	}
	if opts.MinimalPairWeights != "" {
		args = append(args, "--minimal-pair-weights", opts.MinimalPairWeights)
	}
	if opts.MinimalPairMorphology != "" {
		args = append(args, "--minimal-pair-morphology", opts.MinimalPairMorphology)
	}
	if opts.MinimalPairMaxPairs > 0 {
		args = append(args, "--minimal-pair-max-pairs", fmt.Sprintf("%d", opts.MinimalPairMaxPairs))
	}
	if opts.MinimalPairSeed != 0 {
		args = append(args, "--minimal-pair-seed", fmt.Sprintf("%d", opts.MinimalPairSeed))
	}
	if opts.MinimalPairReportOut != "" {
		args = append(args, "--minimal-pair-report-out", opts.MinimalPairReportOut)
	}
	if opts.MinimalPairSampleOut != "" {
		args = append(args, "--minimal-pair-sample-out", opts.MinimalPairSampleOut)
	}
	if opts.MinimalPairSampleCount > 0 {
		args = append(args, "--minimal-pair-sample-count", fmt.Sprintf("%d", opts.MinimalPairSampleCount))
	}

	fmt.Printf("Running: python3 %s\n", strings.Join(args, " "))
	cmd := exec.Command("python3", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("prepare.py failed: %w", err)
	}

	// Validate output: check that at least one train shard exists.
	trainPattern := filepath.Join(opts.Output, "train_*.bin")
	matches, _ := filepath.Glob(trainPattern)
	if len(matches) == 0 {
		return fmt.Errorf("no training shards produced in %s", opts.Output)
	}
	fmt.Printf("\nValidation: found %d training shard(s)\n", len(matches))

	valPattern := filepath.Join(opts.Output, "val_*.bin")
	valMatches, _ := filepath.Glob(valPattern)
	fmt.Printf("Validation: found %d validation shard(s)\n", len(valMatches))

	manifestPath := filepath.Join(opts.Output, data.DatasetManifestFilename)
	manifest, err := data.LoadDatasetManifest(manifestPath)
	if err != nil {
		return fmt.Errorf("validate prepared dataset manifest: %w", err)
	}
	fmt.Printf("Validation: dataset modality=%s representation=%s vocab_size=%d\n",
		manifest.Modality, manifest.Representation, manifest.VocabSize)

	return nil
}

// findPrepareScript locates scripts/prepare.py relative to the binary,
// the current working directory, or via MIXLAB_SCRIPTS env var.
func findPrepareScript() (string, error) {
	// 1. Check MIXLAB_SCRIPTS env var.
	if envDir := os.Getenv("MIXLAB_SCRIPTS"); envDir != "" {
		p := filepath.Join(envDir, "prepare.py")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	// 2. Relative to the executable.
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), "scripts", "prepare.py")
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	// 3. Relative to current working directory.
	if wd, err := os.Getwd(); err == nil {
		candidates := []string{
			filepath.Join(wd, "scripts", "prepare.py"),
			filepath.Join(wd, "..", "scripts", "prepare.py"),
			filepath.Join(wd, "cmd", "mixlab", "scripts", "prepare.py"),
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				return p, nil
			}
		}
	}

	return "", fmt.Errorf("cannot find scripts/prepare.py; run from the repository root or set MIXLAB_SCRIPTS=/path/to/mixlab/scripts")
}
