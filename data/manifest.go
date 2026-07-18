package data

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	DatasetManifestFilename             = "mixlab.dataset.json"
	DatasetManifestFormat               = "mixlab.dataset"
	DatasetManifestVersion              = 1
	DatasetRepresentationDiscreteTokens = "discrete_tokens"
	DatasetTokenDTypeUint16             = "uint16"
	DatasetShardFormatTokenStreamV1     = "mixlab_token_shard_v1"
)

// DatasetManifest describes the representation shared by a set of Mixlab
// shards. It is optional for legacy datasets and required for new modality
// adapters introduced after the discrete-token foundation release.
type DatasetManifest struct {
	Format          string                   `json:"format"`
	Version         int                      `json:"version"`
	Representation  string                   `json:"representation"`
	Modality        string                   `json:"modality"`
	VocabSize       int                      `json:"vocab_size"`
	TokenDType      string                   `json:"token_dtype"`
	ShardFormat     string                   `json:"shard_format"`
	SpecialTokenIDs map[string]int           `json:"special_token_ids,omitempty"`
	Artifacts       DatasetManifestArtifacts `json:"artifacts,omitempty"`
	Splits          map[string]DatasetSplit  `json:"splits"`
}

// DatasetManifestArtifacts names optional files relative to the manifest.
type DatasetManifestArtifacts struct {
	Tokenizer  string `json:"tokenizer,omitempty"`
	Vocabulary string `json:"vocabulary,omitempty"`
}

// DatasetSplit records the shard pattern and exact token/shard counts emitted
// by preparation. Patterns are relative to the manifest directory.
type DatasetSplit struct {
	Pattern string `json:"pattern"`
	Tokens  int64  `json:"tokens"`
	Shards  int    `json:"shards"`
}

// LoadDatasetManifest parses and validates a manifest from disk.
func LoadDatasetManifest(path string) (*DatasetManifest, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var manifest DatasetManifest
	dec := json.NewDecoder(strings.NewReader(string(blob)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return nil, fmt.Errorf("parse dataset manifest %q: %w", path, err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return nil, fmt.Errorf("parse dataset manifest %q: %w", path, err)
	}
	manifest.normalize()
	if err := manifest.Validate(); err != nil {
		return nil, fmt.Errorf("dataset manifest %q: %w", path, err)
	}
	return &manifest, nil
}

// FindDatasetManifest locates a manifest beside the first sorted shard matched
// by pattern. A missing manifest is valid for backward compatibility.
func FindDatasetManifest(pattern string) (*DatasetManifest, string, bool, error) {
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, "", false, err
	}
	sort.Strings(matches)
	dir := filepath.Dir(pattern)
	if len(matches) > 0 {
		dir = filepath.Dir(matches[0])
	}
	path := filepath.Join(dir, DatasetManifestFilename)
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, "", false, nil
		}
		return nil, "", false, err
	}
	manifest, err := LoadDatasetManifest(path)
	if err != nil {
		return nil, "", false, err
	}
	return manifest, path, true, nil
}

// Validate checks the representation contract independent of any model.
func (m *DatasetManifest) Validate() error {
	if m == nil {
		return fmt.Errorf("manifest is nil")
	}
	if m.Format != DatasetManifestFormat {
		return fmt.Errorf("format=%q, want %q", m.Format, DatasetManifestFormat)
	}
	if m.Version != DatasetManifestVersion {
		return fmt.Errorf("version=%d is unsupported; this build supports version %d", m.Version, DatasetManifestVersion)
	}
	if m.Representation != DatasetRepresentationDiscreteTokens {
		return fmt.Errorf("representation=%q is unsupported; this release supports %q", m.Representation, DatasetRepresentationDiscreteTokens)
	}
	if !validDatasetIdentifier(m.Modality) {
		return fmt.Errorf("modality=%q must be a lowercase identifier", m.Modality)
	}
	if m.VocabSize <= 0 || m.VocabSize > 1<<16 {
		return fmt.Errorf("vocab_size=%d must be in [1,65536] for uint16 token shards", m.VocabSize)
	}
	if m.TokenDType != DatasetTokenDTypeUint16 {
		return fmt.Errorf("token_dtype=%q is unsupported; want %q", m.TokenDType, DatasetTokenDTypeUint16)
	}
	if m.ShardFormat != DatasetShardFormatTokenStreamV1 {
		return fmt.Errorf("shard_format=%q is unsupported; want %q", m.ShardFormat, DatasetShardFormatTokenStreamV1)
	}
	seenSpecialIDs := make(map[int]string, len(m.SpecialTokenIDs))
	for name, id := range m.SpecialTokenIDs {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("special_token_ids contains an empty token name")
		}
		if id < 0 || id >= m.VocabSize {
			return fmt.Errorf("special_token_ids[%q]=%d is outside [0,%d)", name, id, m.VocabSize)
		}
		if previous, ok := seenSpecialIDs[id]; ok {
			return fmt.Errorf("special tokens %q and %q both use id %d", previous, name, id)
		}
		seenSpecialIDs[id] = name
	}
	if err := validateManifestArtifactPath("artifacts.tokenizer", m.Artifacts.Tokenizer); err != nil {
		return err
	}
	if err := validateManifestArtifactPath("artifacts.vocabulary", m.Artifacts.Vocabulary); err != nil {
		return err
	}
	if len(m.Splits) == 0 {
		return fmt.Errorf("splits must contain at least one dataset split")
	}
	for name, split := range m.Splits {
		if !validDatasetIdentifier(name) {
			return fmt.Errorf("split name %q must be a lowercase identifier", name)
		}
		if err := validateManifestArtifactPath("splits."+name+".pattern", split.Pattern); err != nil {
			return err
		}
		if split.Tokens < 0 {
			return fmt.Errorf("splits.%s.tokens=%d must be >= 0", name, split.Tokens)
		}
		if split.Shards < 0 {
			return fmt.Errorf("splits.%s.shards=%d must be >= 0", name, split.Shards)
		}
		if split.Tokens > 0 && split.Shards == 0 {
			return fmt.Errorf("splits.%s has %d tokens but zero shards", name, split.Tokens)
		}
	}
	return nil
}

// ValidateModelVocab rejects using a token dataset with an incompatible model
// embedding/output vocabulary.
func (m *DatasetManifest) ValidateModelVocab(vocabSize int) error {
	if err := m.Validate(); err != nil {
		return err
	}
	if vocabSize != m.VocabSize {
		return fmt.Errorf("dataset vocab_size=%d does not match model vocab_size=%d", m.VocabSize, vocabSize)
	}
	return nil
}

func (m *DatasetManifest) normalize() {
	m.Format = strings.ToLower(strings.TrimSpace(m.Format))
	m.Representation = strings.ToLower(strings.TrimSpace(m.Representation))
	m.Modality = strings.ToLower(strings.TrimSpace(m.Modality))
	m.TokenDType = strings.ToLower(strings.TrimSpace(m.TokenDType))
	m.ShardFormat = strings.ToLower(strings.TrimSpace(m.ShardFormat))
}

func validDatasetIdentifier(value string) bool {
	if value == "" || value[0] < 'a' || value[0] > 'z' {
		return false
	}
	for i := 1; i < len(value); i++ {
		c := value[i]
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			continue
		}
		return false
	}
	return true
}

func validateManifestArtifactPath(field, value string) error {
	if strings.TrimSpace(value) == "" {
		if strings.HasPrefix(field, "splits.") {
			return fmt.Errorf("%s must not be empty", field)
		}
		return nil
	}
	if filepath.IsAbs(value) {
		return fmt.Errorf("%s=%q must be relative to the manifest", field, value)
	}
	cleaned := filepath.Clean(value)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return fmt.Errorf("%s=%q escapes the manifest directory", field, value)
	}
	return nil
}
