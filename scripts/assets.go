// Package prepareassets embeds the Python assets required by mixlab prepare.
package prepareassets

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
)

const PrepareScriptName = "prepare.py"

var prepareAssetNames = []string{
	PrepareScriptName,
	"prepare_records.py",
}

//go:embed prepare.py prepare_records.py
var prepareAssets embed.FS

// Materialize writes the embedded prepare bundle into dir and returns the
// prepare.py path. The caller owns dir and its cleanup.
func Materialize(dir string) (string, error) {
	for _, name := range prepareAssetNames {
		content, err := prepareAssets.ReadFile(name)
		if err != nil {
			return "", fmt.Errorf("read embedded %s: %w", name, err)
		}
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, content, 0o600); err != nil {
			return "", fmt.Errorf("write embedded %s: %w", name, err)
		}
	}
	return filepath.Join(dir, PrepareScriptName), nil
}
