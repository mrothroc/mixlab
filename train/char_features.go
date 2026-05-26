package train

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// File format for char_features.bin:
//
//	header: charFeatureHeaderInts little-endian int32s
//	  [0] magic                      (charFeatureMagic)
//	  [1] version                    (charFeatureVersion)
//	  [2] vocab_size                 (must equal config.vocab_size)
//	  [3] char_vocab_size            (must equal config.char_vocab_size)
//	  [4] char_max_per_token         (must equal config.char_max_per_token)
//	  [5..] reserved for tooling; the engine does not interpret these.
//	payload: vocab_size * char_max_per_token little-endian uint16 ids
//	  id 0 is reserved padding (contributes zero in the runtime gather)
//	  other ids must satisfy id < char_vocab_size
//
// The bundled scripts/prepare.py produces this format for HuggingFace
// ByteLevel BPE tokenizers, but the engine is encoding-agnostic — any
// tool that emits a matching header + uint16 payload works.
const (
	charFeaturesFilename        = "char_features.bin"
	charFeatureMagic      int32 = 20260526
	charFeatureVersion    int32 = 1
	charFeatureHeaderInts       = 256
)

func configureCharFeaturesForTraining(cfg *ArchConfig, trainPattern string) (string, error) {
	if cfg == nil || cfg.CharVocabSize <= 0 {
		return "", nil
	}
	if len(cfg.CharFeatureIDs) == cfg.VocabSize*cfg.CharMaxPerToken {
		return cfg.CharFeatureSource, nil
	}
	path, ok, err := charFeaturesPathForTrainPattern(trainPattern)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("char_vocab_size=%d requires %s next to training shards", cfg.CharVocabSize, charFeaturesFilename)
	}
	if err := loadCharFeaturesIntoConfig(cfg, path); err != nil {
		return "", err
	}
	return path, nil
}

func configureCharFeaturesForConfigPath(cfg *ArchConfig, configPath string, extraPaths ...string) error {
	if cfg == nil || cfg.CharVocabSize <= 0 {
		return nil
	}
	if len(cfg.CharFeatureIDs) == cfg.VocabSize*cfg.CharMaxPerToken {
		return nil
	}
	candidates := make([]string, 0, 1+len(extraPaths))
	if configPath != "" {
		candidates = append(candidates, filepath.Join(filepath.Dir(configPath), charFeaturesFilename))
	}
	for _, p := range extraPaths {
		if p == "" {
			continue
		}
		candidates = append(candidates, filepath.Join(filepath.Dir(p), charFeaturesFilename))
	}
	for _, path := range uniqueStrings(candidates) {
		if _, err := os.Stat(path); err == nil {
			return loadCharFeaturesIntoConfig(cfg, path)
		}
	}
	return fmt.Errorf("char_vocab_size=%d requires %s next to config or weights", cfg.CharVocabSize, charFeaturesFilename)
}

func charFeaturesPathForTrainPattern(trainPattern string) (string, bool, error) {
	matches, err := filepath.Glob(trainPattern)
	if err != nil {
		return "", false, err
	}
	sort.Strings(matches)
	candidates := make([]string, 0, 2)
	if len(matches) > 0 {
		candidates = append(candidates, filepath.Join(filepath.Dir(matches[0]), charFeaturesFilename))
	}
	if trainPattern != "" {
		candidates = append(candidates, filepath.Join(filepath.Dir(trainPattern), charFeaturesFilename))
	}
	for _, path := range uniqueStrings(candidates) {
		if _, err := os.Stat(path); err == nil {
			return path, true, nil
		}
	}
	return "", false, nil
}

func loadCharFeaturesIntoConfig(cfg *ArchConfig, path string) error {
	ids, err := loadCharFeatures(path, cfg.VocabSize, cfg.CharVocabSize, cfg.CharMaxPerToken)
	if err != nil {
		return err
	}
	cfg.CharFeatureIDs = ids
	cfg.CharFeatureSource = path
	return nil
}

func loadCharFeatures(path string, vocabSize, charVocabSize, charMaxPerToken int) ([]int32, error) {
	if vocabSize <= 0 {
		return nil, fmt.Errorf("invalid vocab_size=%d", vocabSize)
	}
	if charVocabSize < 257 {
		return nil, fmt.Errorf("invalid char_vocab_size=%d", charVocabSize)
	}
	if charMaxPerToken <= 0 {
		return nil, fmt.Errorf("invalid char_max_per_token=%d", charMaxPerToken)
	}
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read char features %q: %w", path, err)
	}
	headerBytes := charFeatureHeaderInts * 4
	payloadBytes := vocabSize * charMaxPerToken * 2
	wantBytes := headerBytes + payloadBytes
	if len(blob) != wantBytes {
		return nil, fmt.Errorf("char features %q size=%d, want %d for vocab_size=%d char_max_per_token=%d", path, len(blob), wantBytes, vocabSize, charMaxPerToken)
	}
	header := func(i int) int32 {
		return int32(binary.LittleEndian.Uint32(blob[i*4 : i*4+4]))
	}
	if got := header(0); got != charFeatureMagic {
		return nil, fmt.Errorf("char features %q magic=%d, want %d", path, got, charFeatureMagic)
	}
	if got := header(1); got != charFeatureVersion {
		return nil, fmt.Errorf("char features %q version=%d, want %d", path, got, charFeatureVersion)
	}
	if got := int(header(2)); got != vocabSize {
		return nil, fmt.Errorf("char features %q vocab_size=%d, want %d", path, got, vocabSize)
	}
	if got := int(header(3)); got != charVocabSize {
		return nil, fmt.Errorf("char features %q char_vocab_size=%d, want %d", path, got, charVocabSize)
	}
	if got := int(header(4)); got != charMaxPerToken {
		return nil, fmt.Errorf("char features %q char_max_per_token=%d, want %d", path, got, charMaxPerToken)
	}
	payload := blob[headerBytes:]
	out := make([]int32, vocabSize*charMaxPerToken)
	for i := range out {
		id := int(binary.LittleEndian.Uint16(payload[i*2 : i*2+2]))
		if id >= charVocabSize {
			return nil, fmt.Errorf("char features %q id[%d]=%d outside char_vocab_size=%d", path, i, id, charVocabSize)
		}
		out[i] = int32(id)
	}
	return out, nil
}

func uniqueStrings(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}
