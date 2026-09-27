package credentialcrypto

import (
	"bytes"
	"crypto/ecdh"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

func unhex(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// RFC 9180 vectors, revision 5f503c5, as distributed with CIRCL v1.6.4.
// Base mode, DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, ChaCha20Poly1305.
func TestRFC9180Vector(t *testing.T) {
	info := unhex(t, "4f6465206f6e2061204772656369616e2055726e")
	random := unhex(t, "909a9b35d3dc4713a5e72a4da274b55d3d3821a37e5d099e74a647db583a904b")
	sk := unhex(t, "8057991eef8f1f1af18f4a9491d16a1ce333f695d4db8e38da75975c4478e0fb")
	pk := unhex(t, "4310ee97d88cc1f088a5576c77ab0cf5c3ac797f3d95139c6c84b5429c59662a")
	wantEnc := unhex(t, "1afa08d3dec047a643885163f1180476fa7ddb54c6a8029ea33f95796bf2ac4a")
	aad := unhex(t, "436f756e742d30")
	pt := unhex(t, "4265617574792069732074727574682c20747275746820626561757479")
	wantCT := unhex(t, "1c5250d8034ec2b784ba2cfd69dbdb8af406cfe3ff938e131f0def8c8b60b4db21993c62ce81883d2dd1b51a28")
	enc, ct, err := seal(pk, info, aad, pt, bytes.NewReader(random))
	if err != nil || !bytes.Equal(enc, wantEnc) || !bytes.Equal(ct, wantCT) {
		t.Fatalf("RFC seal: enc=%x ct=%x err=%v", enc, ct, err)
	}
	got, err := Open(sk, info, aad, wantEnc, wantCT)
	if err != nil || !bytes.Equal(got, pt) {
		t.Fatalf("RFC open: %x %v", got, err)
	}
}

func TestHPKEBindingsAndBounds(t *testing.T) {
	sk, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pk, info, aad, pt := sk.PublicKey().Bytes(), []byte("protocol v1"), []byte("node/job/attempt"), []byte("opaque credential")
	enc, ct, err := Seal(pk, info, aad, pt)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Open(sk.Bytes(), info, aad, enc, ct)
	if err != nil || !bytes.Equal(got, pt) {
		t.Fatal("round trip", err)
	}
	enc2, _, err := Seal(pk, info, aad, pt)
	if err != nil || bytes.Equal(enc, enc2) {
		t.Fatal("ephemeral key reused", err)
	}
	for _, name := range []string{"info", "aad", "ciphertext", "enc", "private"} {
		t.Run(name, func(t *testing.T) {
			i, a, e, c, k := bytes.Clone(info), bytes.Clone(aad), bytes.Clone(enc), bytes.Clone(ct), sk.Bytes()
			switch name {
			case "info":
				i[0] ^= 1
			case "aad":
				a[0] ^= 1
			case "ciphertext":
				c[0] ^= 1
			case "enc":
				e[0] ^= 1
			case "private":
				k[4] ^= 1
			}
			if out, err := Open(k, i, a, e, c); err == nil || len(out) != 0 {
				t.Fatal("tamper accepted")
			}
		})
	}
	for _, n := range []int{0, MaxPlaintext + 1} {
		if _, _, err := Seal(pk, info, aad, make([]byte, n)); err == nil {
			t.Fatal("plaintext bounds")
		}
	}
	for _, infoTooLong := range []bool{false, true} {
		i, a := info, bytes.Repeat([]byte{1}, MaxBinding+1)
		if infoTooLong {
			i, a = a, aad
		}
		if _, _, err := Seal(pk, i, a, pt); err == nil {
			t.Fatal("seal binding bounds")
		}
		if _, err := Open(sk.Bytes(), i, a, enc, ct); err == nil {
			t.Fatal("open binding bounds")
		}
	}
	for _, n := range []int{0, 16, MaxPlaintext + 17} {
		if _, err := Open(sk.Bytes(), info, aad, enc, make([]byte, n)); err == nil {
			t.Fatal("ciphertext bounds")
		}
	}
	if _, _, err := seal(pk, info, aad, pt, bytes.NewReader(nil)); err == nil {
		t.Fatal("missing entropy accepted")
	}
	max := bytes.Repeat([]byte{42}, MaxPlaintext)
	e, c, err := Seal(pk, info, aad, max)
	if err != nil {
		t.Fatal(err)
	}
	got, err = Open(sk.Bytes(), info, aad, e, c)
	if err != nil || !bytes.Equal(got, max) {
		t.Fatal("maximum payload", err)
	}
}

func TestRejectInvalidX25519(t *testing.T) {
	for _, s := range []string{
		"", "00", "0000000000000000000000000000000000000000000000000000000000000000",
		"0100000000000000000000000000000000000000000000000000000000000000",
		"ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
		"edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
		"eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
		"0900000000000000000000000000000000000000000000000000000000000080",
	} {
		if err := ValidatePublic(unhex(t, s)); err == nil {
			t.Fatalf("accepted invalid public key %s", s)
		}
	}
	base := make([]byte, 32)
	base[0] = 9
	if err := ValidatePublic(base); err != nil {
		t.Fatal(err)
	}
}
