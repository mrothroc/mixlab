package train

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"
)

func configureMLMWordBoundariesForTraining(cfg *ArchConfig, trainPattern string) (string, error) {
	if cfg == nil || !cfg.Training.UsesWholeWordMasking() {
		return "", nil
	}
	if len(cfg.Training.MLMWordStart) == cfg.VocabSize && len(cfg.Training.MLMMaskEligible) == cfg.VocabSize {
		return "configured vocabulary lookup", nil
	}
	path, ok, err := tokenizerPathForTrainPattern(trainPattern)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("training.mlm_mask_unit=%q requires tokenizer.json next to training shards", cfg.Training.EffectiveMLMMaskUnitForStep(0))
	}
	wordStart, eligible, scheme, err := mlmWordBoundaryLUTFromTokenizer(path, cfg.VocabSize, cfg.Training.MLMMaskTokenID)
	if err != nil {
		return "", err
	}
	cfg.Training.MLMWordStart = wordStart
	cfg.Training.MLMMaskEligible = eligible
	return fmt.Sprintf("tokenizer=%s scheme=%s eligible=%d word_starts=%d", path, scheme, countUint8Set(eligible), countUint8Set(wordStart)), nil
}

func mlmWordBoundaryLUTFromTokenizer(path string, vocabSize, maskTokenID int) ([]uint8, []uint8, string, error) {
	if vocabSize <= 0 {
		return nil, nil, "", fmt.Errorf("invalid vocab_size=%d", vocabSize)
	}
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, "", fmt.Errorf("read tokenizer %q: %w", path, err)
	}
	dec := json.NewDecoder(bytes.NewReader(blob))
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
		return nil, nil, "", fmt.Errorf("parse tokenizer %q: %w", path, err)
	}
	model, ok := doc["model"].(map[string]any)
	if !ok {
		return nil, nil, "", fmt.Errorf("tokenizer %q has no model object", path)
	}
	modelType := strings.ToLower(strings.TrimSpace(stringValue(model["type"])))
	tokens, err := tokenizerVocabularyByID(model, vocabSize)
	if err != nil {
		return nil, nil, "", fmt.Errorf("tokenizer %q: %w", path, err)
	}
	special := make([]bool, vocabSize)
	if added, ok := doc["added_tokens"].([]any); ok {
		for i, raw := range added {
			entry, ok := raw.(map[string]any)
			if !ok {
				return nil, nil, "", fmt.Errorf("tokenizer %q added_tokens[%d] is not an object", path, i)
			}
			id, ok := integerValue(entry["id"])
			if !ok || id < 0 || id >= vocabSize {
				return nil, nil, "", fmt.Errorf("tokenizer %q added_tokens[%d] has invalid id", path, i)
			}
			content := stringValue(entry["content"])
			if tokens[id] == "" {
				tokens[id] = content
			} else if content != "" && tokens[id] != content {
				return nil, nil, "", fmt.Errorf("tokenizer %q has conflicting token strings for id %d", path, id)
			}
			if value, ok := entry["special"].(bool); ok && value {
				special[id] = true
			}
		}
	}
	for id, token := range tokens {
		if token == "" {
			return nil, nil, "", fmt.Errorf("tokenizer %q is missing token id %d for vocab_size=%d", path, id, vocabSize)
		}
	}

	preTokenizer := doc["pre_tokenizer"]
	normalizer := doc["normalizer"]
	wordStart := make([]uint8, vocabSize)
	eligible := make([]uint8, vocabSize)
	scheme := ""

	if node := findTokenizerNode(preTokenizer, "Metaspace"); node != nil {
		if modelType != "bpe" && modelType != "unigram" {
			return nil, nil, "", fmt.Errorf("tokenizer %q uses Metaspace with unsupported model.type=%q", path, modelType)
		}
		replacement := stringValue(node["replacement"])
		if replacement == "" {
			replacement = "▁"
		}
		prepend := strings.ToLower(strings.TrimSpace(stringValue(node["prepend_scheme"])))
		legacyPrefix, _ := node["add_prefix_space"].(bool)
		if prepend != "always" && !legacyPrefix {
			return nil, nil, "", fmt.Errorf("tokenizer %q Metaspace must use prepend_scheme=\"always\" for whole-word masking", path)
		}
		for id, token := range tokens {
			if strings.HasPrefix(token, replacement) {
				wordStart[id] = 1
			}
		}
		scheme = "sentencepiece"
	} else if node := findTokenizerNode(preTokenizer, "ByteLevel"); node != nil {
		if modelType != "bpe" {
			return nil, nil, "", fmt.Errorf("tokenizer %q uses ByteLevel with unsupported model.type=%q", path, modelType)
		}
		prefix, _ := node["add_prefix_space"].(bool)
		if !prefix && !hasLeadingSpacePrepend(normalizer) {
			return nil, nil, "", fmt.Errorf("tokenizer %q ByteLevel pre-tokenizer must set add_prefix_space=true for whole-word masking", path)
		}
		reverse := byteLevelReverseMap()
		for id, token := range tokens {
			r, _ := utf8.DecodeRuneInString(token)
			if b, ok := reverse[r]; ok && b == ' ' {
				wordStart[id] = 1
			}
		}
		scheme = "bytelevel"
	} else if modelType == "wordpiece" {
		prefix := stringValue(model["continuing_subword_prefix"])
		if prefix == "" {
			prefix = "##"
		}
		for id, token := range tokens {
			if !strings.HasPrefix(token, prefix) {
				wordStart[id] = 1
			}
		}
		scheme = "wordpiece"
	} else {
		return nil, nil, "", fmt.Errorf("tokenizer %q has no supported whole-word boundary convention", path)
	}

	for id, token := range tokens {
		if token != "" && !special[id] && id != maskTokenID {
			eligible[id] = 1
		}
		if eligible[id] == 0 {
			wordStart[id] = 0
		}
	}
	if countUint8Set(eligible) == 0 {
		return nil, nil, "", fmt.Errorf("tokenizer %q has no eligible non-special tokens", path)
	}
	if countUint8Set(wordStart) == 0 {
		return nil, nil, "", fmt.Errorf("tokenizer %q has no eligible word-start tokens", path)
	}
	return wordStart, eligible, scheme, nil
}

func tokenizerVocabularyByID(model map[string]any, vocabSize int) ([]string, error) {
	tokens := make([]string, vocabSize)
	switch vocab := model["vocab"].(type) {
	case map[string]any:
		for token, rawID := range vocab {
			id, ok := integerValue(rawID)
			if !ok || id < 0 || id >= vocabSize {
				return nil, fmt.Errorf("token %q has invalid id", token)
			}
			if tokens[id] != "" {
				return nil, fmt.Errorf("duplicate token id %d", id)
			}
			tokens[id] = token
		}
	case []any:
		if len(vocab) > vocabSize {
			return nil, fmt.Errorf("model vocab has %d entries, exceeds vocab_size=%d", len(vocab), vocabSize)
		}
		for id, raw := range vocab {
			pair, ok := raw.([]any)
			if !ok || len(pair) == 0 {
				return nil, fmt.Errorf("model vocab[%d] is not a [token, score] entry", id)
			}
			tokens[id] = stringValue(pair[0])
		}
	default:
		return nil, fmt.Errorf("model.vocab must be a token-id object or Unigram list")
	}
	return tokens, nil
}

func findTokenizerNode(value any, nodeType string) map[string]any {
	switch node := value.(type) {
	case map[string]any:
		if strings.EqualFold(stringValue(node["type"]), nodeType) {
			return node
		}
		for _, child := range node {
			if found := findTokenizerNode(child, nodeType); found != nil {
				return found
			}
		}
	case []any:
		for _, child := range node {
			if found := findTokenizerNode(child, nodeType); found != nil {
				return found
			}
		}
	}
	return nil
}

func hasLeadingSpacePrepend(value any) bool {
	node := findTokenizerNode(value, "Prepend")
	if node == nil {
		return false
	}
	prefix := stringValue(node["prepend"])
	if prefix == "" {
		prefix = stringValue(node["content"])
	}
	return strings.HasPrefix(prefix, " ")
}

func integerValue(value any) (int, bool) {
	switch v := value.(type) {
	case json.Number:
		i, err := v.Int64()
		return int(i), err == nil
	case float64:
		i := int(v)
		return i, float64(i) == v
	case int:
		return v, true
	default:
		return 0, false
	}
}

func stringValue(value any) string {
	s, _ := value.(string)
	return s
}

func countUint8Set(values []uint8) int {
	count := 0
	for _, value := range values {
		if value != 0 {
			count++
		}
	}
	return count
}
