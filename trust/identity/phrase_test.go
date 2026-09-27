package identity

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
	"testing"
)

func TestWordlistVersionAndIntegrity(t *testing.T) {
	if WordlistVersion != "mixlab_wordlist_v1" || len(words) != 2048 {
		t.Fatal("wordlist contract changed")
	}
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(wordlistText))); got != "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda" {
		t.Fatal("wordlist changed", got)
	}
	for i, w := range words {
		if w == "" || strings.TrimSpace(w) != w || (i > 0 && words[i-1] >= w) {
			t.Fatal("invalid word ordering", i)
		}
	}
}

func TestPhraseBitOrder(t *testing.T) {
	for _, value := range []byte{0, 255} {
		fp := hex.EncodeToString(bytesOf(value, 32))
		p, err := Cluster(fp)
		index := 0
		if value == 255 {
			index = 2047
		}
		want := strings.TrimSpace(strings.Repeat(words[index]+" ", 8))
		if err != nil || p.Phrase != want || p.Fingerprint != fp || p.WordlistVersion != WordlistVersion {
			t.Fatal(p, err)
		}
	}
	var digest [32]byte
	for i := range digest {
		digest[i] = byte(i)
	}
	p, err := Cluster(hex.EncodeToString(digest[:]))
	if err != nil {
		t.Fatal(err)
	}
	// Independent integer oracle: discard the low 168 bits, then extract
	// eight base-2048 digits from most to least significant.
	x := new(big.Int).SetBytes(digest[:11])
	want := make([]string, 8)
	mask := big.NewInt(2047)
	for i := 7; i >= 0; i-- {
		want[i] = words[new(big.Int).And(x, mask).Int64()]
		x.Rsh(x, 11)
	}
	if p.Phrase != strings.Join(want, " ") || Confirmation(digest) != strings.Join(want[:5], " ") {
		t.Fatal("bit order", p.Phrase)
	}
	for i := 11; i < 32; i++ {
		digest[i] ^= 255
	}
	p2, err := Cluster(hex.EncodeToString(digest[:]))
	if err != nil || p2.Phrase != p.Phrase || p2.Fingerprint == p.Fingerprint {
		t.Fatal("truncated phrase/full fingerprint contract", err)
	}
}

func bytesOf(value byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = value
	}
	return b
}

func TestClusterRequiresCanonicalFingerprint(t *testing.T) {
	for _, fp := range []string{"", strings.Repeat("A", 64), strings.Repeat("g", 64), strings.Repeat("0", 62), strings.Repeat("0", 66)} {
		if _, err := Cluster(fp); err == nil {
			t.Fatal("invalid fingerprint accepted")
		}
	}
}
