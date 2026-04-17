package train

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// runArchRace loads all JSON configs from a directory, trains each sequentially,
// and prints ranked results sorted by validation loss (or last training loss
// if validation is unavailable).
func runArchRace(configsDir, trainPattern string, opts TrainOptions) error {
	if configsDir == "" {
		return fmt.Errorf("arch_race mode requires -configs; pass a directory of JSON configs, e.g.: mixlab -mode arch_race -configs examples/ -train 'data/train_*.bin'")
	}
	if trainPattern == "" {
		return fmt.Errorf("arch_race mode requires -train; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}

	configs, err := loadConfigsFromDir(configsDir)
	if err != nil {
		return err
	}
	if len(configs) == 0 {
		return fmt.Errorf("no .json config files found in %q", configsDir)
	}

	// Validate that all configs share vocab_size, seq_len, and batch_tokens
	// so they can use the same validation set.
	base := configs[0]
	for _, cfg := range configs[1:] {
		if cfg.VocabSize != base.VocabSize || cfg.SeqLen != base.SeqLen || cfg.Training.BatchTokens != base.Training.BatchTokens {
			return fmt.Errorf("arch_race requires shared vocab_size/seq_len/batch_tokens across configs; %q differs from %q",
				cfg.Name, base.Name)
		}
	}

	fmt.Printf("=== ARCH RACE: %d configs from %s ===\n\n", len(configs), configsDir)

	results := make([]TrainResult, 0, len(configs))
	for i, cfg := range configs {
		fmt.Printf("\n--- [%d/%d] Training %q ---\n", i+1, len(configs), cfg.Name)
		r, err := runTrain(cfg, trainPattern, opts)
		if err != nil {
			return fmt.Errorf("config %q failed: %w", cfg.Name, err)
		}
		results = append(results, r)
	}

	// Sort by validation loss (preferred) or last training loss.
	sort.Slice(results, func(i, j int) bool {
		return sortKey(results[i]) < sortKey(results[j])
	})

	fmt.Println("\n=== ARCH RACE RESULTS ===")
	fmt.Printf("%-28s %10s %10s %10s %10s %10s %s\n",
		"CONFIG", "FIRST", "LAST", "VAL", "DELTA", "DELTA%", "TIME")
	fmt.Println(strings.Repeat("-", 96))
	for _, r := range results {
		pct := 0.0
		if r.FirstLoss != 0 {
			pct = r.Delta / r.FirstLoss * 100
		}
		valStr := fmt.Sprintf("%10.4f", r.LastValLoss)
		if !r.HasValLoss {
			valStr = fmt.Sprintf("%10s", "n/a")
		}
		fmt.Printf("%-28s %10.4f %10.4f %s %10.4f %9.1f%% %8s\n",
			r.Name, r.FirstLoss, r.LastLoss, valStr, r.Delta, pct,
			r.Elapsed.Round(time.Millisecond))
	}

	return nil
}

// sortKey returns the value to sort results by: validation loss if available,
// otherwise last training loss.
func sortKey(r TrainResult) float64 {
	if r.HasValLoss {
		return r.LastValLoss
	}
	return r.LastLoss
}

// loadConfigsFromDir walks dir and loads all .json files as ArchConfigs,
// returning them sorted by filename.
func loadConfigsFromDir(dir string) ([]*ArchConfig, error) {
	var paths []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".json") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk configs dir %q: %w", dir, err)
	}
	sort.Strings(paths)

	configs := make([]*ArchConfig, 0, len(paths))
	for _, p := range paths {
		cfg, err := LoadArchConfig(p)
		if err != nil {
			return nil, fmt.Errorf("load config %q: %w", p, err)
		}
		configs = append(configs, cfg)
	}
	return configs, nil
}
