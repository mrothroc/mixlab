package train

import "fmt"

func ComputeTrigramIDs(tok32 []int32, batchSize, seqLen, trigramVocabSize int) ([]int32, error) {
	if trigramVocabSize <= 0 {
		return nil, nil
	}
	if trigramVocabSize <= 1 {
		return nil, fmt.Errorf("invalid trigram_vocab_size=%d; set trigram_vocab_size to 0 to disable or >= 2 to enable hashing", trigramVocabSize)
	}
	if batchSize < 0 || seqLen < 0 {
		return nil, fmt.Errorf("invalid trigram shape batch=%d seq=%d", batchSize, seqLen)
	}
	need := batchSize * seqLen
	if len(tok32) < need {
		return nil, fmt.Errorf("token buffer too small: have=%d need=%d; pass one token id per requested trigram slot", len(tok32), need)
	}
	trigramIDs := make([]int32, need)
	if need == 0 {
		return trigramIDs, nil
	}
	modulus := uint64(trigramVocabSize - 1)
	vocab := uint64(trigramVocabSize)
	for b := 0; b < batchSize; b++ {
		base := b * seqLen
		limit := base + seqLen
		for i := base; i < limit && i < base+2; i++ {
			trigramIDs[i] = 0
		}
		for i := base + 2; i < limit; i++ {
			h := ((uint64(tok32[i-2])*vocab+uint64(tok32[i-1]))*vocab + uint64(tok32[i])) % modulus
			trigramIDs[i] = int32(h + 1) // reserve 0 as the sequence-boundary sentinel
		}
	}
	return trigramIDs, nil
}
