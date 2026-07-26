package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/data"
	prepareassets "github.com/mrothroc/mixlab/scripts"
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

// runPrepare invokes the bundled prepare.py to tokenize raw text into binary shards.
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

	python, err := preparePython(inputFormat)
	if err != nil {
		return err
	}
	script, err := resolvePrepareScript()
	if err != nil {
		return err
	}
	defer script.close()

	args := []string{
		script.path,
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

	fmt.Printf("Running: %s %s\n", python, strings.Join(args, " "))
	cmd := exec.Command(python, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("prepare.py failed using %s scripts: %w", script.source, err)
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

type prepareScriptResolution struct {
	path    string
	source  string
	cleanup func()
}

func (r prepareScriptResolution) close() {
	if r.cleanup != nil {
		r.cleanup()
	}
}

// resolvePrepareScript preserves explicit and legacy filesystem overrides,
// then falls back to the scripts embedded in the installed binary.
func resolvePrepareScript() (prepareScriptResolution, error) {
	executable, _ := os.Executable()
	workingDir, _ := os.Getwd()
	return resolvePrepareScriptAt(os.Getenv("MIXLAB_SCRIPTS"), executable, workingDir)
}

func resolvePrepareScriptAt(envDir, executable, workingDir string) (prepareScriptResolution, error) {
	// 1. Explicit developer override.
	if envDir != "" {
		p := filepath.Join(envDir, "prepare.py")
		if _, err := os.Stat(p); err == nil {
			return prepareScriptResolution{path: p, source: "MIXLAB_SCRIPTS"}, nil
		}
	}

	// 2. Legacy binary-adjacent bundle.
	if executable != "" {
		p := filepath.Join(filepath.Dir(executable), "scripts", "prepare.py")
		if _, err := os.Stat(p); err == nil {
			return prepareScriptResolution{path: p, source: "binary-adjacent"}, nil
		}
	}

	// 3. Legacy source-checkout lookup.
	if workingDir != "" {
		candidates := []string{
			filepath.Join(workingDir, "scripts", "prepare.py"),
			filepath.Join(workingDir, "..", "scripts", "prepare.py"),
			filepath.Join(workingDir, "cmd", "mixlab", "scripts", "prepare.py"),
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				return prepareScriptResolution{path: p, source: "source-checkout"}, nil
			}
		}
	}

	// 4. Installed binaries always carry the canonical prepare bundle.
	tempDir, err := os.MkdirTemp("", "mixlab-prepare-*")
	if err != nil {
		return prepareScriptResolution{}, fmt.Errorf("create temporary directory for embedded prepare scripts: %w", err)
	}
	scriptPath, err := prepareassets.Materialize(tempDir)
	if err != nil {
		_ = os.RemoveAll(tempDir)
		return prepareScriptResolution{}, err
	}
	return prepareScriptResolution{
		path:    scriptPath,
		source:  "embedded",
		cleanup: func() { _ = os.RemoveAll(tempDir) },
	}, nil
}

func preparePython(inputFormat string) (string, error) {
	python, err := exec.LookPath("python3")
	if err != nil {
		return "", fmt.Errorf("prepare requires Python 3 on PATH; install python3 and retry")
	}

	modules := []string{"numpy"}
	install := "numpy"
	if inputFormat != "fasta" {
		modules = append(modules, "tokenizers")
		install += " tokenizers"
	}
	check := "import " + strings.Join(modules, ", ")
	cmd := exec.Command(python, "-c", check)
	if output, err := cmd.CombinedOutput(); err != nil {
		detail := strings.TrimSpace(string(output))
		if detail != "" {
			detail = ": " + detail
		}
		return "", fmt.Errorf(
			"prepare requires Python packages %s; install them in the active environment with `python3 -m pip install %s`%s",
			strings.Join(modules, " and "), install, detail,
		)
	}
	return python, nil
}
