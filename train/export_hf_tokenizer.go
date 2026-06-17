package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type hfSpecialToken struct {
	Token string
	ID    int
	Set   bool
}

type hfTokenizerSpecials struct {
	Pad  hfSpecialToken
	EOS  hfSpecialToken
	BOS  hfSpecialToken
	UNK  hfSpecialToken
	Mask hfSpecialToken
}

func specialTokenIDPtr(tok hfSpecialToken) *int {
	if !tok.Set {
		return nil
	}
	id := tok.ID
	return &id
}

func writeHFTokenizerArtifacts(outputDir string, src hfTokenizerSource, specials hfTokenizerSpecials) error {
	if err := copyFile(src.TokenizerJSON, filepath.Join(outputDir, "tokenizer.json")); err != nil {
		return err
	}
	for _, name := range []string{"tokenizer_config.json", "special_tokens_map.json"} {
		sourcePath := filepath.Join(src.Dir, name)
		targetPath := filepath.Join(outputDir, name)
		doc := map[string]any{}
		if _, err := os.Stat(sourcePath); err == nil {
			if err := readJSONMapFile(sourcePath, &doc); err != nil {
				return err
			}
		}
		switch name {
		case "tokenizer_config.json":
			if cls, _ := doc["tokenizer_class"].(string); strings.TrimSpace(cls) == "" {
				doc["tokenizer_class"] = "PreTrainedTokenizerFast"
			}
			mergeTokenizerConfigSpecials(doc, specials)
		case "special_tokens_map.json":
			mergeSpecialTokensMap(doc, specials)
		}
		if err := writeJSONFile(targetPath, doc); err != nil {
			return err
		}
	}
	return nil
}

func deriveHFTokenizerSpecials(src hfTokenizerSource, cfg *ArchConfig) (hfTokenizerSpecials, error) {
	tokenToID, added, err := inspectTokenizerJSONForSpecials(src.TokenizerJSON)
	if err != nil {
		return hfTokenizerSpecials{}, err
	}
	var out hfTokenizerSpecials
	if err := mergeSpecialsFromSidecar(filepath.Join(src.Dir, "special_tokens_map.json"), tokenToID, &out); err != nil {
		return hfTokenizerSpecials{}, err
	}
	if err := mergeSpecialsFromSidecar(filepath.Join(src.Dir, "tokenizer_config.json"), tokenToID, &out); err != nil {
		return hfTokenizerSpecials{}, err
	}
	for _, tok := range added {
		if !tok.Special {
			continue
		}
		applyInferredSpecialToken(tok.Content, tok.ID, &out)
	}
	// Masked-capable exports advertise an AutoModelForMaskedLM head; the MNTP/MLM
	// eval path needs tokenizer.mask_token to be set. The authoritative mask id is
	// training.mlm_mask_token_id (required and range-validated for masked objectives),
	// so resolve its token string from the vocab when the tokenizer didn't already
	// declare a mask token.
	if !out.Mask.Set && hfExportSupportsMaskedLM(cfg) {
		if token, ok := tokenForID(tokenToID, cfg.Training.MLMMaskTokenID); ok {
			setSpecialToken(&out.Mask, token, cfg.Training.MLMMaskTokenID)
		}
	}
	return out, nil
}

func tokenForID(tokenToID map[string]int, id int) (string, bool) {
	for token, tokenID := range tokenToID {
		if tokenID == id {
			return token, true
		}
	}
	return "", false
}

type hfAddedTokenJSON struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

func inspectTokenizerJSONForSpecials(path string) (map[string]int, []hfAddedTokenJSON, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read tokenizer json %q: %w", path, err)
	}
	var raw struct {
		AddedTokens []hfAddedTokenJSON `json:"added_tokens"`
		Model       struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil, fmt.Errorf("parse tokenizer json %q: %w", path, err)
	}
	tokenToID := map[string]int{}
	for token, id := range raw.Model.Vocab {
		tokenToID[token] = id
	}
	for _, tok := range raw.AddedTokens {
		if tok.Content != "" {
			tokenToID[tok.Content] = tok.ID
		}
	}
	return tokenToID, raw.AddedTokens, nil
}

func mergeSpecialsFromSidecar(path string, tokenToID map[string]int, out *hfTokenizerSpecials) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("stat tokenizer sidecar %q: %w", path, err)
	}
	doc := map[string]any{}
	if err := readJSONMapFile(path, &doc); err != nil {
		return err
	}
	for _, spec := range []struct {
		key string
		dst *hfSpecialToken
	}{
		{key: "pad_token", dst: &out.Pad},
		{key: "eos_token", dst: &out.EOS},
		{key: "bos_token", dst: &out.BOS},
		{key: "unk_token", dst: &out.UNK},
		{key: "mask_token", dst: &out.Mask},
	} {
		token, ok := specialTokenString(doc[spec.key])
		if !ok {
			continue
		}
		id, ok := tokenToID[token]
		if !ok {
			continue
		}
		setSpecialToken(spec.dst, token, id)
	}
	return nil
}

func specialTokenString(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, strings.TrimSpace(t) != ""
	case map[string]any:
		if content, ok := t["content"].(string); ok && strings.TrimSpace(content) != "" {
			return content, true
		}
	}
	return "", false
}

func applyInferredSpecialToken(token string, id int, out *hfTokenizerSpecials) {
	lower := strings.ToLower(token)
	switch {
	case !out.Pad.Set && strings.Contains(lower, "pad"):
		setSpecialToken(&out.Pad, token, id)
	case !out.EOS.Set && (strings.Contains(lower, "eos") || strings.Contains(lower, "endoftext") || token == "</s>"):
		setSpecialToken(&out.EOS, token, id)
	case !out.BOS.Set && (strings.Contains(lower, "bos") || token == "<s>"):
		setSpecialToken(&out.BOS, token, id)
	case !out.UNK.Set && strings.Contains(lower, "unk"):
		setSpecialToken(&out.UNK, token, id)
	case !out.Mask.Set && strings.Contains(lower, "mask"):
		setSpecialToken(&out.Mask, token, id)
	}
}

func setSpecialToken(dst *hfSpecialToken, token string, id int) {
	if dst == nil || dst.Set {
		return
	}
	dst.Token = token
	dst.ID = id
	dst.Set = true
}

func mergeTokenizerConfigSpecials(doc map[string]any, specials hfTokenizerSpecials) {
	for _, item := range []struct {
		tokenKey string
		idKey    string
		token    hfSpecialToken
	}{
		{tokenKey: "pad_token", idKey: "pad_token_id", token: specials.Pad},
		{tokenKey: "eos_token", idKey: "eos_token_id", token: specials.EOS},
		{tokenKey: "bos_token", idKey: "bos_token_id", token: specials.BOS},
		{tokenKey: "unk_token", idKey: "unk_token_id", token: specials.UNK},
		{tokenKey: "mask_token", idKey: "mask_token_id", token: specials.Mask},
	} {
		if !item.token.Set {
			continue
		}
		doc[item.tokenKey] = item.token.Token
		doc[item.idKey] = item.token.ID
	}
}

func mergeSpecialTokensMap(doc map[string]any, specials hfTokenizerSpecials) {
	for _, item := range []struct {
		key   string
		token hfSpecialToken
	}{
		{key: "pad_token", token: specials.Pad},
		{key: "eos_token", token: specials.EOS},
		{key: "bos_token", token: specials.BOS},
		{key: "unk_token", token: specials.UNK},
		{key: "mask_token", token: specials.Mask},
	} {
		if item.token.Set {
			doc[item.key] = item.token.Token
		}
	}
}

func readJSONMapFile(path string, out *map[string]any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read json %q: %w", path, err)
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("parse json %q: %w", path, err)
	}
	if *out == nil {
		*out = map[string]any{}
	}
	return nil
}
