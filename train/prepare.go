package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// PrepareOptions holds flags for the prepare command.
type PrepareOptions struct {
	Input         string
	Output        string
	VocabSize     int
	ValSplit      float64
	TokenizerPath string
	TextFieldName string
}

// runPrepare shells out to scripts/prepare.py to tokenize raw text into binary shards.
func runPrepare(opts PrepareOptions) error {
	if opts.Input == "" {
		return fmt.Errorf("-input is required for prepare mode; pass a text file, JSONL, or directory, e.g.: mixlab -mode prepare -input corpus.jsonl -output data/")
	}
	if opts.Output == "" {
		return fmt.Errorf("-output is required for prepare mode; pass an output directory, e.g.: mixlab -mode prepare -input corpus.jsonl -output data/")
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
	if opts.TokenizerPath != "" {
		args = append(args, "--tokenizer-path", opts.TokenizerPath)
	}
	if opts.TextFieldName != "" && opts.TextFieldName != "text" {
		args = append(args, "--text-field", opts.TextFieldName)
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
