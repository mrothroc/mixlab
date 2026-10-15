// Package wire provides encoding mechanics only, never application authority.
package wire

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/mrothroc/mixlab/internal/strictjson"
)

const MaxRecordBytes = 1 << 20
const MaxLifetimeSeconds int64 = 7 * 24 * 3600

type Validated interface{ Validate() error }

// Decode accepts the owner's canonical encoding only, as does workerjob. This
// also rejects missing zero-valued fields, null scalars and case-folded aliases.
func Decode[T Validated](b []byte, dst T) error {
	if len(b) == 0 || len(b) > MaxRecordBytes {
		return fmt.Errorf("DiLoCo record exceeds byte budget")
	}
	if err := strictjson.Validate(b, 32); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	c, err := json.Marshal(dst)
	if err != nil {
		return err
	}
	if !bytes.Equal(b, c) {
		return fmt.Errorf("noncanonical DiLoCo record")
	}
	return dst.Validate()
}

func Digest(v any) (string, error) {
	b, e := json.Marshal(v)
	if e != nil {
		return "", e
	}
	if len(b) > MaxRecordBytes {
		return "", fmt.Errorf("record too large")
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}
func Hash(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 32 && hex.EncodeToString(b) == s
}
func Text(s string) bool {
	return len(s) > 0 && len(s) <= 256 && strings.IndexFunc(s, func(r rune) bool { return r < 33 || r > 126 }) < 0
}
func Finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func Window(start, end int64) bool {
	return start > 0 && end > start && end-start <= MaxLifetimeSeconds
}
