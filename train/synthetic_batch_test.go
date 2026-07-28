//go:build mlx && cgo && (darwin || linux)

package train

import "math/rand"

// generateSyntheticBatch builds a next-token training pair from random tokens:
// it draws batchTokens+1 tokens in [0, vocabSize) and returns x = tokens[:-1]
// and y = tokens[1:]. It is shared by the MLX training/eval tests on both the
// Metal (darwin) and CUDA (linux) backends, so it lives in a platform-neutral
// file rather than the darwin-only integration suite.
func generateSyntheticBatch(rng *rand.Rand, batchTokens, vocabSize int) (x, y []int) {
	raw := make([]int, batchTokens+1)
	for i := range raw {
		raw[i] = rng.Intn(vocabSize)
	}
	x = make([]int, batchTokens)
	y = make([]int, batchTokens)
	copy(x, raw[:batchTokens])
	copy(y, raw[1:batchTokens+1])
	return x, y
}
