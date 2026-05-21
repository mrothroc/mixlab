package train

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"unicode/utf8"
)

type byteLevelTokenizerJSON struct {
	AddedTokens []struct {
		ID      int  `json:"id"`
		Special bool `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Vocab map[string]int `json:"vocab"`
	} `json:"model"`
}

func configureFirstByteMaskForTraining(cfg *ArchConfig, trainPattern string) (string, error) {
	if cfg == nil || !cfg.Training.FirstByteMask {
		return "", nil
	}
	if len(cfg.Training.FirstByteMaskValid) == cfg.VocabSize {
		return "configured vocab mask", nil
	}
	valid, source, err := loadFirstByteMaskValid(trainPattern, cfg.VocabSize)
	if err != nil {
		return "", err
	}
	cfg.Training.FirstByteMaskValid = valid
	return source, nil
}

func loadFirstByteMaskValid(trainPattern string, vocab int) ([]int32, string, error) {
	if vocab <= 0 {
		return nil, "", fmt.Errorf("invalid vocab_size=%d", vocab)
	}
	tokenizerPath, ok, err := tokenizerPathForTrainPattern(trainPattern)
	if err != nil {
		return nil, "", err
	}
	if ok {
		valid, err := firstByteMaskValidFromTokenizer(tokenizerPath, vocab)
		if err != nil {
			return nil, "", err
		}
		return valid, tokenizerPath, nil
	}
	if vocab > 256 {
		return nil, "", fmt.Errorf("training.first_byte_mask requires tokenizer.json next to training shards when vocab_size=%d exceeds byte ids", vocab)
	}
	return identityFirstByteMaskValid(vocab), "byte-id fallback", nil
}

func tokenizerPathForTrainPattern(trainPattern string) (string, bool, error) {
	matches, err := filepath.Glob(trainPattern)
	if err != nil {
		return "", false, err
	}
	sort.Strings(matches)
	if len(matches) > 0 {
		path := filepath.Join(filepath.Dir(matches[0]), "tokenizer.json")
		if _, err := os.Stat(path); err == nil {
			return path, true, nil
		}
	}
	path := filepath.Join(filepath.Dir(trainPattern), "tokenizer.json")
	if _, err := os.Stat(path); err == nil {
		return path, true, nil
	}
	return "", false, nil
}

func firstByteMaskValidFromTokenizer(path string, vocab int) ([]int32, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer %q: %w", path, err)
	}
	dec := json.NewDecoder(bytes.NewReader(blob))
	var tok byteLevelTokenizerJSON
	if err := dec.Decode(&tok); err != nil {
		return nil, fmt.Errorf("parse tokenizer %q: %w", path, err)
	}
	if len(tok.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer %q has no model.vocab", path)
	}

	special := make(map[int]bool, len(tok.AddedTokens))
	for _, added := range tok.AddedTokens {
		if added.Special && added.ID >= 0 && added.ID < vocab {
			special[added.ID] = true
		}
	}

	reverse := byteLevelReverseMap()
	valid := make([]int32, vocab)
	seen := 0
	for token, id := range tok.Model.Vocab {
		if id < 0 || id >= vocab || special[id] || token == "" {
			continue
		}
		r, _ := utf8.DecodeRuneInString(token)
		b, ok := reverse[r]
		if !ok {
			return nil, fmt.Errorf("tokenizer %q token %q starts with non-ByteLevel rune %U", path, token, r)
		}
		if isValidUTF8FirstByte(b) {
			valid[id] = 1
		}
		seen++
	}
	if seen == 0 {
		return nil, fmt.Errorf("tokenizer %q has no non-special vocab entries below vocab_size=%d", path, vocab)
	}
	return valid, nil
}

func identityFirstByteMaskValid(vocab int) []int32 {
	valid := make([]int32, vocab)
	for i := 0; i < vocab; i++ {
		if i < 256 && isValidUTF8FirstByte(byte(i)) {
			valid[i] = 1
		}
	}
	return valid
}

func isValidUTF8FirstByte(b byte) bool {
	return b <= 0x7f || (b >= 0xc2 && b <= 0xf4)
}

func byteLevelReverseMap() map[rune]byte {
	bs := make([]byte, 0, 256)
	for b := byte('!'); b <= byte('~'); b++ {
		bs = append(bs, b)
	}
	for b := 0xa1; b <= 0xac; b++ {
		bs = append(bs, byte(b))
	}
	for b := 0xae; b <= 0xff; b++ {
		bs = append(bs, byte(b))
	}

	seen := make(map[byte]bool, len(bs))
	for _, b := range bs {
		seen[b] = true
	}
	cs := make([]rune, 0, 256)
	for _, b := range bs {
		cs = append(cs, rune(b))
	}
	n := 0
	for b := 0; b < 256; b++ {
		if seen[byte(b)] {
			continue
		}
		bs = append(bs, byte(b))
		cs = append(cs, rune(256+n))
		n++
	}

	reverse := make(map[rune]byte, len(bs))
	for i, b := range bs {
		reverse[cs[i]] = b
	}
	return reverse
}
