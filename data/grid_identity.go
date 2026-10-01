package data

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// GridDatasetIdentity binds replay to the manifest and all train/val bytes.
// Read once per run, not once per step/checkpoint; shards must remain immutable.
func GridDatasetIdentity(path string) (string, error) {
	m, err := LoadDatasetManifest(path)
	if err != nil {
		return "", err
	}
	if m.Representation != "grid" {
		return "", fmt.Errorf("grid identity requires a grid manifest")
	}
	paths := []string{path}
	for _, split := range []string{"train", "val"} {
		if s, ok := m.Splits[split]; ok {
			files, err := filepath.Glob(filepath.Join(filepath.Dir(path), s.Pattern))
			if err != nil {
				return "", err
			}
			if len(files) != s.Shards {
				return "", fmt.Errorf("grid %s shard count mismatch", split)
			}
			paths = append(paths, files...)
		}
	}
	sort.Strings(paths)
	hash := sha256.New()
	enc := json.NewEncoder(hash)
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		h := sha256.New()
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			return "", copyErr
		}
		if closeErr != nil {
			return "", closeErr
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", err
		}
		if err = enc.Encode([]string{abs, hex.EncodeToString(h.Sum(nil))}); err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
