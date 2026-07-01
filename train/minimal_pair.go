package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

type minimalPairRecord struct {
	ID      string `json:"id"`
	Clean   []int  `json:"clean"`
	Corrupt []int  `json:"corrupt"`
	Family  string `json:"family,omitempty"`
}

type minimalPairSampler struct {
	records []minimalPairRecord
	path    string
}

func minimalPairActive(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.MultiheadEnabled() && cfg.Training.MinimalPair != nil
}

func newMinimalPairSampler(cfg *ArchConfig) (*minimalPairSampler, error) {
	if !minimalPairActive(cfg) {
		return nil, nil
	}
	path := resolveConfigRelativePath(cfg.SourcePath, cfg.Training.MinimalPair.Path)
	records, err := loadMinimalPairJSONL(path, cfg.VocabSize)
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("minimal pair file %q has no records", path)
	}
	return &minimalPairSampler{records: records, path: path}, nil
}

func resolveConfigRelativePath(configPath, path string) string {
	path = strings.TrimSpace(path)
	if path == "" || filepath.IsAbs(path) || strings.TrimSpace(configPath) == "" {
		return path
	}
	dir := filepath.Dir(configPath)
	if dir == "." || dir == "" {
		return path
	}
	return filepath.Join(dir, path)
}

func loadMinimalPairJSONL(path string, vocabSize int) ([]minimalPairRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open minimal pair file %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	return decodeMinimalPairJSONL(f, path, vocabSize)
}

func decodeMinimalPairJSONL(r io.Reader, source string, vocabSize int) ([]minimalPairRecord, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	var out []minimalPairRecord
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec minimalPairRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("%s line %d: invalid JSON: %w", source, lineNo, err)
		}
		if err := validateMinimalPairRecord(rec, vocabSize); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", source, lineNo, err)
		}
		out = append(out, rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read minimal pair file %q: %w", source, err)
	}
	return out, nil
}

func validateMinimalPairRecord(rec minimalPairRecord, vocabSize int) error {
	if strings.TrimSpace(rec.ID) == "" {
		return fmt.Errorf("id must be non-empty")
	}
	if len(rec.Clean) == 0 {
		return fmt.Errorf("clean tokens must be non-empty")
	}
	if len(rec.Corrupt) == 0 {
		return fmt.Errorf("corrupt tokens must be non-empty")
	}
	for name, toks := range map[string][]int{"clean": rec.Clean, "corrupt": rec.Corrupt} {
		for i, tok := range toks {
			if tok < 0 || tok >= vocabSize {
				return fmt.Errorf("%s[%d]=%d out of range [0,%d)", name, i, tok, vocabSize)
			}
		}
	}
	return nil
}

func maybeAttachMinimalPairs(sampler *minimalPairSampler, cfg *ArchConfig, step int, batch objectiveBatch, rawBatchSize, seqLen int) (objectiveBatch, error) {
	if sampler == nil || !minimalPairActive(cfg) {
		return batch, nil
	}
	if rawBatchSize <= 0 || seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid minimal-pair batch shape rows=%d seq_len=%d", rawBatchSize, seqLen)
	}
	if rawBatchSize%2 != 0 {
		return objectiveBatch{}, fmt.Errorf("minimal-pair energy training requires an even number of sequence rows per batch, got %d", rawBatchSize)
	}
	need := rawBatchSize * seqLen
	if len(batch.x) < need*len(cfg.Training.Heads) || len(batch.y) < need*len(cfg.Training.Heads) || len(batch.lossMask) < need*len(cfg.Training.Heads) {
		return objectiveBatch{}, fmt.Errorf("minimal-pair objective batch too small for heads=%d need=%d", len(cfg.Training.Heads), need)
	}
	totalPairs := rawBatchSize / 2
	activePairs := int(math.Ceil(float64(totalPairs) * cfg.Training.MinimalPair.PairBatchFraction))
	if activePairs < 1 {
		activePairs = 1
	}
	if activePairs > totalPairs {
		activePairs = totalPairs
	}
	pad := minimalPairPadTokenID(cfg)
	for headIdx, head := range cfg.Training.Heads {
		if head.Objective != arch.ObjectiveEnergy {
			continue
		}
		rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0xeb100d5eed000001+uint64(headIdx))
		tokenOffset := headIdx * need
		rowOffset := headIdx * rawBatchSize
		clear(batch.lossMask[tokenOffset : tokenOffset+need])
		for pairIdx := 0; pairIdx < totalPairs; pairIdx++ {
			rec := sampler.records[0]
			active := pairIdx < activePairs
			if active {
				rec = sampler.records[rng.Intn(len(sampler.records))]
			}
			cleanRow := tokenOffset + (2*pairIdx)*seqLen
			corruptRow := cleanRow + seqLen
			fillMinimalPairRow(batch.x[cleanRow:cleanRow+seqLen], batch.y[cleanRow:cleanRow+seqLen], rec.Clean, pad)
			fillMinimalPairRow(batch.x[corruptRow:corruptRow+seqLen], batch.y[corruptRow:corruptRow+seqLen], rec.Corrupt, pad)
			if active {
				batch.lossMask[cleanRow] = 1
				batch.lossMask[corruptRow] = 1
			}
			if len(batch.unmaskedX) >= tokenOffset+need {
				copy(batch.unmaskedX[cleanRow:cleanRow+seqLen], batch.x[cleanRow:cleanRow+seqLen])
				copy(batch.unmaskedX[corruptRow:corruptRow+seqLen], batch.x[corruptRow:corruptRow+seqLen])
			}
		}
		if len(batch.diffusionBlockStart) >= rowOffset+rawBatchSize && len(batch.diffusionBlockEnd) >= rowOffset+rawBatchSize {
			for row := 0; row < rawBatchSize; row++ {
				batch.diffusionBlockStart[rowOffset+row] = 0
				batch.diffusionBlockEnd[rowOffset+row] = int32(seqLen)
			}
		}
	}
	return batch, nil
}

func fillMinimalPairRow(dstX, dstY []int, tokens []int, pad int) {
	for i := range dstX {
		dstX[i] = pad
		dstY[i] = pad
	}
	n := len(tokens)
	if n > len(dstX) {
		n = len(dstX)
	}
	copy(dstX[:n], tokens[:n])
	copy(dstY[:n], tokens[:n])
}

func minimalPairPadTokenID(cfg *ArchConfig) int {
	if cfg != nil && cfg.Training.MLMMaskTokenID >= 0 && cfg.Training.MLMMaskTokenID < cfg.VocabSize {
		return cfg.Training.MLMMaskTokenID
	}
	return 0
}
