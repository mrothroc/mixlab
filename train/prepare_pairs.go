package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

type PreparePairsOptions struct {
	ConfigPath string
	PairIn     string
	PairOut    string
	VocabSize  int
	MaxLen     int
}

func runPreparePairsWithOptions(opts PreparePairsOptions) error {
	pairIn := strings.TrimSpace(opts.PairIn)
	vocabSize := opts.VocabSize
	maxLen := opts.MaxLen
	if strings.TrimSpace(opts.ConfigPath) != "" {
		cfg, err := LoadArchConfig(opts.ConfigPath)
		if err != nil {
			return err
		}
		vocabSize = cfg.VocabSize
		if maxLen == 0 {
			maxLen = cfg.SeqLen
		}
		if pairIn == "" && cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.Source == arch.MinimalPairSourceJSONL {
			pairIn = resolveConfigRelativePath(cfg.SourcePath, cfg.Training.MinimalPair.Path)
		}
	}
	if pairIn == "" {
		return fmt.Errorf("-pair-in is required for prepare-pairs mode unless -config has training.minimal_pair.source=\"jsonl\"")
	}
	if vocabSize <= 0 {
		return fmt.Errorf("-vocab-size must be > 0 for prepare-pairs mode, or provide -config")
	}
	if maxLen < 0 {
		return fmt.Errorf("-pair-max-len must be >= 0")
	}
	if opts.PairOut != "" && samePath(pairIn, opts.PairOut) {
		return fmt.Errorf("-pair-in and -pair-out must be different paths")
	}
	records, err := loadMinimalPairs(pairIn, arch.MinimalPairSourceJSONL, minimalPairDecodeOptions{
		VocabSize:     vocabSize,
		MaxLen:        maxLen,
		RequireFamily: true,
	})
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return fmt.Errorf("minimal pair file %q has no records", pairIn)
	}
	summary := summarizeMinimalPairs(records)
	if opts.PairOut != "" {
		if err := os.MkdirAll(filepath.Dir(opts.PairOut), 0o755); err != nil && filepath.Dir(opts.PairOut) != "." {
			return fmt.Errorf("create pair output directory: %w", err)
		}
		f, err := os.Create(opts.PairOut)
		if err != nil {
			return fmt.Errorf("create pair output %q: %w", opts.PairOut, err)
		}
		closeOut := true
		defer func() {
			if closeOut {
				_ = f.Close()
			}
		}()
		if err := writeMinimalPairBinary(f, records, vocabSize, maxLen); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			closeOut = false
			return fmt.Errorf("close pair output %q: %w", opts.PairOut, err)
		}
		closeOut = false
		fmt.Printf("wrote minimal-pair binary shard to %s\n", opts.PairOut)
	}
	blob, err := json.Marshal(summary)
	if err != nil {
		return fmt.Errorf("marshal minimal-pair summary: %w", err)
	}
	fmt.Printf("minimal-pair validation summary: %s\n", blob)
	return nil
}
