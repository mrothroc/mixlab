package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// LoadArchConfig reads and validates a JSON architecture config from path.
func LoadArchConfig(path string) (*ArchConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if filepath.IsAbs(path) {
			return nil, fmt.Errorf("read config %q: %w", path, err)
		}
		b, err = os.ReadFile(filepath.Join("..", path))
		if err != nil {
			return nil, fmt.Errorf("read config %q: %w", path, err)
		}
	}
	cfg, err := ParseArchConfig(b, path)
	if err != nil {
		return nil, err
	}
	cfg.SourcePath = path
	return cfg, nil
}

// stripJSONComments removes // line comments from JSONC input.
// This allows config files to contain inline documentation.
func stripJSONComments(data []byte) []byte {
	var out []byte
	inString := false
	escaped := false
	for i := 0; i < len(data); i++ {
		c := data[i]
		if escaped {
			out = append(out, c)
			escaped = false
			continue
		}
		if inString {
			out = append(out, c)
			switch c {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
			out = append(out, c)
			continue
		case '/':
			if i+1 >= len(data) || data[i+1] != '/' {
				break
			}
			// Skip to end of line.
			for i < len(data) && data[i] != '\n' {
				i++
			}
			if i < len(data) {
				out = append(out, '\n')
			}
			continue
		}
		out = append(out, c)
	}
	return out
}

// ParseArchConfig parses and validates a JSON architecture config.
// The source parameter is used in error messages (typically the file path).
// Supports JSONC: // line comments are stripped before parsing.
// Unknown JSON fields are rejected to prevent silent misconfiguration.
func ParseArchConfig(data []byte, source string) (*ArchConfig, error) {
	data = stripJSONComments(data)
	var cfg ArchConfig
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w (check field names against docs/config-reference.md)", source, err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", source, err)
	}
	_, cfg.attnDropoutSet = fields["attn_dropout"]
	_, cfg.hiddenDropoutSet = fields["hidden_dropout"]
	warnDeprecatedMamba3Blocks(cfg.Blocks)
	return validateConfig(&cfg, source)
}
