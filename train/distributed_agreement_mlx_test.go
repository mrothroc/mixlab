//go:build mlx && cgo && (darwin || linux)

package train

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"testing"
)

func TestDistributedWeightHashChunkingPreservesCanonicalDigest(t *testing.T) {
	weights := [][]float32{
		{0, 1, -2, float32(math.Inf(1))},
		make([]float32, 4097),
	}
	for i := range weights[1] {
		weights[1][i] = float32(i)/17 - 3
	}
	got := hashWeightData64(weights)

	digest := sha256.New()
	var encoded [4]byte
	for _, weight := range weights {
		writeUint64(digest, uint64(len(weight)))
		for _, value := range weight {
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			_, _ = digest.Write(encoded[:])
		}
	}
	want := binary.LittleEndian.Uint64(digest.Sum(nil)[:8])
	if got != want {
		t.Fatalf("chunked hash=%x want canonical scalar hash=%x", got, want)
	}
}
