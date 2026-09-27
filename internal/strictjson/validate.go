// Package strictjson rejects ambiguous JSON without interpreting application
// schemas. Callers own byte budgets, object schemas and canonical encodings.
package strictjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"unicode/utf8"
)

func Validate(b []byte, maxDepth int) error {
	if maxDepth < 1 || !utf8.Valid(b) || !json.Valid(b) {
		return fmt.Errorf("invalid JSON")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := value(d, 0, maxDepth); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON value")
	}
	return nil
}
func value(d *json.Decoder, depth, maxDepth int) error {
	if depth > maxDepth {
		return fmt.Errorf("JSON nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			k, err := d.Token()
			if err != nil {
				return err
			}
			s, ok := k.(string)
			if !ok || seen[s] {
				return fmt.Errorf("duplicate JSON key")
			}
			seen[s] = true
			if err := value(d, depth+1, maxDepth); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := value(d, depth+1, maxDepth); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}
