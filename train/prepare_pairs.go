package train

import (
	"bufio"
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
	energyAggregation := ""
	pairKind := ""
	invarianceSkipIDs := map[int]bool(nil)
	pllMarginSkipIDs := map[int]bool(nil)
	pllMarginMaskTokenID := -1
	if strings.TrimSpace(opts.ConfigPath) != "" {
		cfg, err := LoadArchConfig(opts.ConfigPath)
		if err != nil {
			return err
		}
		vocabSize = cfg.VocabSize
		if maxLen == 0 {
			maxLen = cfg.SeqLen
		}
		configuredPairSources := 0
		if cfg.Training.MinimalPair != nil {
			configuredPairSources++
		}
		if cfg.Training.Invariance != nil {
			configuredPairSources++
		}
		if cfg.Training.PLLMargin != nil {
			configuredPairSources++
		}
		if pairIn == "" && configuredPairSources > 1 {
			return fmt.Errorf("-pair-in is required when config contains multiple pair artifact settings")
		}
		if pairIn == "" && cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.Source == arch.MinimalPairSourceJSONL {
			pairIn = resolveConfigRelativePath(cfg.SourcePath, cfg.Training.MinimalPair.Path)
			pairKind = "minimal"
		}
		if pairIn == "" && cfg.Training.Invariance != nil && cfg.Training.Invariance.Source != arch.InvarianceSourceBinary {
			pairIn = resolveConfigRelativePath(cfg.SourcePath, cfg.Training.Invariance.Path)
			pairKind = "invariance"
		}
		if pairIn == "" && cfg.Training.PLLMargin != nil && cfg.Training.PLLMargin.Source != arch.PLLMarginSourceBinary {
			pairIn = resolveConfigRelativePath(cfg.SourcePath, cfg.Training.PLLMargin.Path)
			pairKind = "pll_margin"
		}
		invarianceSkipIDs = invarianceSkipTokenIDs(cfg)
		pllMarginSkipIDs = pllMarginSkipTokenIDs(cfg)
		if cfg.Training.PLLMargin != nil {
			pllMarginMaskTokenID = cfg.Training.MLMMaskTokenID
		}
		if cfg.Training.MinimalPair != nil {
			energyAggregation = cfg.Training.MinimalPair.EnergyAggregationMode()
		}
	}
	if pairIn == "" {
		return fmt.Errorf("-pair-in is required for prepare-pairs mode unless -config supplies a JSONL training.minimal_pair, training.invariance, or training.pll_margin path")
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
	if pairKind == "" {
		var err error
		pairKind, err = detectPreparePairKind(pairIn)
		if err != nil {
			return err
		}
	}
	if pairKind == "invariance" {
		return runPrepareInvariancePairs(pairIn, opts.PairOut, vocabSize, maxLen, invarianceSkipIDs)
	}
	if pairKind == "pll_margin" {
		return runPreparePLLMarginPairs(pairIn, opts.PairOut, vocabSize, maxLen, pllMarginMaskTokenID, pllMarginSkipIDs)
	}
	records, err := loadMinimalPairs(pairIn, arch.MinimalPairSourceJSONL, minimalPairDecodeOptions{
		VocabSize:         vocabSize,
		MaxLen:            maxLen,
		RequireFamily:     true,
		EnergyAggregation: energyAggregation,
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

func detectPreparePairKind(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open pair input %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal([]byte(line), &fields); err != nil {
			return "", fmt.Errorf("%s line %d: invalid JSON: %w", path, lineNo, err)
		}
		_, viewPos := fields["view_pos"]
		_, viewNeg := fields["view_neg"]
		if viewPos || viewNeg {
			return "pll_margin", nil
		}
		_, viewA := fields["view_a"]
		_, viewB := fields["view_b"]
		if viewA || viewB {
			return "invariance", nil
		}
		return "minimal", nil
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("read pair input %q: %w", path, err)
	}
	return "", fmt.Errorf("pair input %q has no records", path)
}

func runPrepareInvariancePairs(pairIn, pairOut string, vocabSize, maxLen int, skipTokenIDs map[int]bool) error {
	records, err := loadInvariancePairs(pairIn, arch.InvarianceSourceJSONL, invariancePairDecodeOptions{
		VocabSize:     vocabSize,
		MaxLen:        maxLen,
		MaskTokenID:   -1,
		SkipTokenIDs:  skipTokenIDs,
		RequireFamily: true,
	})
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return fmt.Errorf("invariance pair file %q has no records", pairIn)
	}
	if pairOut != "" {
		if err := os.MkdirAll(filepath.Dir(pairOut), 0o755); err != nil && filepath.Dir(pairOut) != "." {
			return fmt.Errorf("create pair output directory: %w", err)
		}
		f, err := os.Create(pairOut)
		if err != nil {
			return fmt.Errorf("create pair output %q: %w", pairOut, err)
		}
		closeOut := true
		defer func() {
			if closeOut {
				_ = f.Close()
			}
		}()
		if err := writeInvariancePairBinary(f, records, vocabSize, maxLen); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			closeOut = false
			return fmt.Errorf("close pair output %q: %w", pairOut, err)
		}
		closeOut = false
		fmt.Printf("wrote invariance-pair binary shard to %s\n", pairOut)
	}
	summary, err := json.Marshal(summarizeInvariancePairs(records))
	if err != nil {
		return fmt.Errorf("marshal invariance-pair summary: %w", err)
	}
	fmt.Printf("invariance-pair validation summary: %s\n", summary)
	return nil
}

func runPreparePLLMarginPairs(pairIn, pairOut string, vocabSize, maxLen, maskTokenID int, skipTokenIDs map[int]bool) error {
	records, err := loadPLLMarginPairs(pairIn, arch.PLLMarginSourceJSONL, pllMarginPairDecodeOptions{
		VocabSize:     vocabSize,
		MaxLen:        maxLen,
		MaskTokenID:   maskTokenID,
		SkipTokenIDs:  skipTokenIDs,
		RequireFamily: true,
	})
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return fmt.Errorf("PLL margin pair file %q has no records", pairIn)
	}
	if pairOut != "" {
		if err := os.MkdirAll(filepath.Dir(pairOut), 0o755); err != nil && filepath.Dir(pairOut) != "." {
			return fmt.Errorf("create pair output directory: %w", err)
		}
		f, err := os.Create(pairOut)
		if err != nil {
			return fmt.Errorf("create pair output %q: %w", pairOut, err)
		}
		closeOut := true
		defer func() {
			if closeOut {
				_ = f.Close()
			}
		}()
		if err := writePLLMarginPairBinary(f, records, vocabSize, maxLen); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			closeOut = false
			return fmt.Errorf("close pair output %q: %w", pairOut, err)
		}
		closeOut = false
		fmt.Printf("wrote PLL margin pair binary shard to %s\n", pairOut)
	}
	summary, err := json.Marshal(summarizePLLMarginPairs(records))
	if err != nil {
		return fmt.Errorf("marshal PLL margin pair summary: %w", err)
	}
	fmt.Printf("PLL margin pair validation summary: %s\n", summary)
	return nil
}
