package train

import "fmt"

func ComputeBigramIDs(tok32 []int32, need, bigramVocabSize int) ([]int32, error) {
	if bigramVocabSize <= 0 {
		return nil, nil
	}
	if bigramVocabSize <= 1 {
		return nil, fmt.Errorf("invalid bigram_vocab_size=%d; set bigram_vocab_size to 0 to disable or >= 2 to enable hashing", bigramVocabSize)
	}
	if len(tok32) < need {
		return nil, fmt.Errorf("token buffer too small: have=%d need=%d; pass one token id per requested bigram slot", len(tok32), need)
	}
	bigramIDs := make([]int32, need)
	if need == 0 {
		return bigramIDs, nil
	}
	modulus := bigramVocabSize - 1
	bigramIDs[0] = int32(modulus)
	for i := 1; i < need; i++ {
		bigramIDs[i] = int32((36313*int(tok32[i]) ^ 27191*int(tok32[i-1])) % modulus)
	}
	return bigramIDs, nil
}
