package train

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

type hfSpecialToken struct {
	Token  string
	ID     int
	Set    bool
	Source string
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

func deriveHFTokenizerSpecials(src hfTokenizerSource, cfg *ArchConfig, opts ExportHFOptions) (hfTokenizerSpecials, error) {
	tokenToID, added, err := inspectTokenizerJSONForSpecials(src.TokenizerJSON)
	if err != nil {
		return hfTokenizerSpecials{}, err
	}
	if err := validateHFExportSpecialTokenOptions(opts, cfg.VocabSize); err != nil {
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
			setSpecialToken(&out.Mask, token, cfg.Training.MLMMaskTokenID, "training.mlm_mask_token_id")
		}
	}
	setSpecialTokenFlag(&out.BOS, opts.BOSTokenID, tokenToID, "-bos-token-id")
	setSpecialTokenFlag(&out.EOS, opts.EOSTokenID, tokenToID, "-eos-token-id")
	setSpecialTokenFlag(&out.Pad, opts.PADTokenID, tokenToID, "-pad-token-id")
	if err := mergeSpecialsFromDatasetManifest(src.Dir, cfg, tokenToID, &out); err != nil {
		return hfTokenizerSpecials{}, err
	}
	if framing := cfg.Training.ExampleFraming; framing != nil {
		setSpecialTokenID(&out.BOS, framing.BosID, tokenToID, "training.example_framing.bos_id")
		setSpecialTokenID(&out.EOS, framing.EosID, tokenToID, "training.example_framing.eos_id")
	}
	if err := validateHFTokenizerSpecials(out, cfg.VocabSize); err != nil {
		return hfTokenizerSpecials{}, err
	}
	return out, nil
}

func validateHFExportSpecialTokenOptions(opts ExportHFOptions, vocabSize int) error {
	for _, item := range []struct {
		name string
		id   *int
	}{
		{name: "-bos-token-id", id: opts.BOSTokenID},
		{name: "-eos-token-id", id: opts.EOSTokenID},
		{name: "-pad-token-id", id: opts.PADTokenID},
	} {
		if item.id != nil && (*item.id < 0 || *item.id >= vocabSize) {
			return fmt.Errorf("%s=%d is outside model vocabulary [0,%d)", item.name, *item.id, vocabSize)
		}
	}
	return nil
}

func mergeSpecialsFromDatasetManifest(dir string, cfg *ArchConfig, tokenToID map[string]int, out *hfTokenizerSpecials) error {
	path := filepath.Join(dir, data.DatasetManifestFilename)
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("stat dataset manifest %q: %w", path, err)
	}
	manifest, err := data.LoadDatasetManifest(path)
	if err != nil {
		return err
	}
	if err := manifest.ValidateModelVocab(cfg.VocabSize); err != nil {
		return fmt.Errorf("dataset manifest %q is incompatible with HF export: %w", path, err)
	}
	for _, item := range []struct {
		name string
		dst  *hfSpecialToken
	}{
		{name: "pad", dst: &out.Pad},
		{name: "eos", dst: &out.EOS},
		{name: "bos", dst: &out.BOS},
	} {
		if id, ok := manifest.SpecialTokenIDs[item.name]; ok {
			setSpecialTokenID(item.dst, id, tokenToID, data.DatasetManifestFilename+".special_token_ids."+item.name)
		}
	}
	return nil
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
		key   string
		idKey string
		dst   *hfSpecialToken
	}{
		{key: "pad_token", idKey: "pad_token_id", dst: &out.Pad},
		{key: "eos_token", idKey: "eos_token_id", dst: &out.EOS},
		{key: "bos_token", idKey: "bos_token_id", dst: &out.BOS},
		{key: "unk_token", idKey: "unk_token_id", dst: &out.UNK},
		{key: "mask_token", idKey: "mask_token_id", dst: &out.Mask},
	} {
		token, ok := specialTokenString(doc[spec.key])
		tokenID, tokenFound := tokenToID[token]
		declaredID, idFound := jsonInteger(doc[spec.idKey])
		if ok && tokenFound && idFound && tokenID != declaredID {
			return fmt.Errorf("tokenizer sidecar %q has %s=%q at id %d but %s=%d", path, spec.key, token, tokenID, spec.idKey, declaredID)
		}
		if ok && tokenFound {
			setSpecialToken(spec.dst, token, tokenID, filepath.Base(path)+"."+spec.key)
			continue
		}
		if idFound {
			setSpecialTokenID(spec.dst, declaredID, tokenToID, filepath.Base(path)+"."+spec.idKey)
		}
	}
	return nil
}

func jsonInteger(v any) (int, bool) {
	number, ok := v.(float64)
	if !ok || math.IsNaN(number) || math.IsInf(number, 0) || math.Trunc(number) != number {
		return 0, false
	}
	return int(number), true
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
		setSpecialToken(&out.Pad, token, id, "tokenizer.json added token")
	case !out.EOS.Set && (strings.Contains(lower, "eos") || strings.Contains(lower, "endoftext") || token == "</s>"):
		setSpecialToken(&out.EOS, token, id, "tokenizer.json added token")
	case !out.BOS.Set && (strings.Contains(lower, "bos") || token == "<s>"):
		setSpecialToken(&out.BOS, token, id, "tokenizer.json added token")
	case !out.UNK.Set && strings.Contains(lower, "unk"):
		setSpecialToken(&out.UNK, token, id, "tokenizer.json added token")
	case !out.Mask.Set && strings.Contains(lower, "mask"):
		setSpecialToken(&out.Mask, token, id, "tokenizer.json added token")
	}
}

func setSpecialToken(dst *hfSpecialToken, token string, id int, source string) {
	if dst == nil || dst.Set {
		return
	}
	dst.Token = token
	dst.ID = id
	dst.Set = true
	dst.Source = source
}

func setSpecialTokenID(dst *hfSpecialToken, id int, tokenToID map[string]int, source string) {
	if dst == nil || dst.Set {
		return
	}
	token, _ := tokenForID(tokenToID, id)
	*dst = hfSpecialToken{Token: token, ID: id, Set: true, Source: source}
}

func setSpecialTokenFlag(dst *hfSpecialToken, id *int, tokenToID map[string]int, source string) {
	if id == nil {
		return
	}
	setSpecialTokenID(dst, *id, tokenToID, source)
}

func validateHFTokenizerSpecials(specials hfTokenizerSpecials, vocabSize int) error {
	for _, item := range []struct {
		name  string
		token hfSpecialToken
	}{
		{name: "pad", token: specials.Pad},
		{name: "eos", token: specials.EOS},
		{name: "bos", token: specials.BOS},
		{name: "unk", token: specials.UNK},
		{name: "mask", token: specials.Mask},
	} {
		if !item.token.Set {
			continue
		}
		if item.token.ID < 0 || item.token.ID >= vocabSize {
			return fmt.Errorf("%s token id %d from %s is outside model vocabulary [0,%d)", item.name, item.token.ID, item.token.Source, vocabSize)
		}
	}
	return nil
}

func warnMissingHFGenerationSpecials(specials hfTokenizerSpecials) {
	missing := make([]string, 0, 3)
	for _, item := range []struct {
		name string
		set  bool
	}{
		{name: "bos", set: specials.BOS.Set},
		{name: "eos", set: specials.EOS.Set},
		{name: "pad", set: specials.Pad.Set},
	} {
		if !item.set {
			missing = append(missing, item.name)
		}
	}
	if len(missing) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "warning: export-hf could not resolve %s token ids; pass explicit -bos-token-id, -eos-token-id, or -pad-token-id overrides as needed\n", strings.Join(missing, ", "))
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
		if item.token.Token != "" {
			doc[item.tokenKey] = item.token.Token
		}
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
		if item.token.Set && item.token.Token != "" {
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
