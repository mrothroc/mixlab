// Package identity renders public cluster fingerprints and channel-confirmation
// digests. A phrase is presentation data, never proof of enrollment authority.
package identity

import (
	_ "embed"
	"encoding/hex"
	"fmt"
	"strings"
)

const WordlistVersion = "mixlab_wordlist_v1"

//go:embed mixlab_wordlist_v1.txt
var wordlistText string
var words = strings.Split(strings.TrimSuffix(wordlistText, "\n"), "\n")

type Presentation struct {
	Fingerprint     string `json:"fingerprint"`
	WordlistVersion string `json:"wordlist_version"`
	Phrase          string `json:"phrase"`
}

// Cluster uses the first 88 fingerprint bits, most-significant bit first, as
// eight 11-bit word indexes. It always returns the full fingerprint alongside.
func Cluster(fingerprint string) (Presentation, error) {
	b, err := hex.DecodeString(fingerprint)
	if err != nil || len(b) != 32 || hex.EncodeToString(b) != fingerprint {
		return Presentation{}, fmt.Errorf("expected canonical SHA-256 fingerprint")
	}
	return Presentation{Fingerprint: fingerprint, WordlistVersion: WordlistVersion, Phrase: phrase(b, 8)}, nil
}

// Confirmation renders the first 55 bits of a completed, domain-separated
// channel/request digest. Enrollment owns that digest and the approval policy.
func Confirmation(digest [32]byte) string { return phrase(digest[:], 5) }

func phrase(digest []byte, n int) string {
	out := make([]string, n)
	for i := range out {
		index := 0
		for bit := i * 11; bit < (i+1)*11; bit++ {
			index = index<<1 | int((digest[bit/8]>>uint(7-bit%8))&1)
		}
		out[i] = words[index]
	}
	return strings.Join(out, " ")
}
