package arch

// Weight counts and IR emitters for the optional model-level feature residual
// channels (char bag, bigram embedding, trigram embedding) that mix into the
// token stream before the first block.

func bigramWeightCount(modelDim, bigramVocabSize, bigramDim int) int {
	if bigramVocabSize <= 0 {
		return 0
	}
	if bigramDim <= 0 {
		bigramDim = modelDim
	}
	count := 2 // embed table + learned scale
	if bigramDim != modelDim {
		count++
	}
	return count
}

func trigramWeightCount(modelDim, trigramVocabSize, trigramDim int) int {
	if trigramVocabSize <= 0 {
		return 0
	}
	if trigramDim <= 0 {
		trigramDim = modelDim
	}
	count := 2 // embed table + learned scale
	if trigramDim != modelDim {
		count++
	}
	return count
}

func charWeightCount(modelDim, charVocabSize, charDim int) int {
	if charVocabSize <= 0 {
		return 0
	}
	if charDim <= 0 {
		charDim = modelDim
	}
	count := 2 // char table + learned scale
	if charDim != modelDim {
		count++
	}
	return count
}

func emitCharIR(prog *Program, baseState string, B, T, D, wi, charVocabSize, charDim, charMaxPerToken int) int {
	if charVocabSize <= 0 {
		return wi
	}
	if charDim <= 0 {
		charDim = D
	}
	if charMaxPerToken <= 0 {
		charMaxPerToken = 16
	}
	prog.DeclareInput("char_ids", TensorInt32, []int{B, T, charMaxPerToken})
	prog.CharFeatureBag(weightName(wi), "char_ids", "char_bag", B, T, charMaxPerToken, charDim)
	wi++
	charState := "char_bag"
	if charDim != D {
		prog.MatMul(charState, weightName(wi), "char_proj")
		wi++
		charState = "char_proj"
	}
	prog.Mul(charState, weightName(wi), "char_scaled")
	wi++
	prog.Add(baseState, "char_scaled", "x")
	return wi
}

func emitBigramIR(prog *Program, baseState string, B, T, D, wi, bigramVocabSize, bigramDim int) int {
	if bigramVocabSize <= 0 {
		return wi
	}
	if bigramDim <= 0 {
		bigramDim = D
	}
	prog.DeclareInput("bigram_ids", TensorInt32, []int{B, T})
	prog.Embed(weightName(wi), "bigram_ids", "bigram_embed")
	wi++
	prog.Reshape("bigram_embed", []int{B * T, bigramDim}, "bigram_flat")
	bigramState := "bigram_flat"
	if bigramDim != D {
		prog.MatMul(bigramState, weightName(wi), "bigram_proj")
		wi++
		bigramState = "bigram_proj"
	}
	prog.Mul(bigramState, weightName(wi), "bigram_scaled")
	wi++
	prog.Add(baseState, "bigram_scaled", "x")
	return wi
}

func emitTrigramIR(prog *Program, baseState string, B, T, D, wi, trigramVocabSize, trigramDim int) int {
	if trigramVocabSize <= 0 {
		return wi
	}
	if trigramDim <= 0 {
		trigramDim = D
	}
	prog.DeclareInput("trigram_ids", TensorInt32, []int{B, T})
	prog.Embed(weightName(wi), "trigram_ids", "trigram_embed")
	wi++
	prog.Reshape("trigram_embed", []int{B * T, trigramDim}, "trigram_flat")
	trigramState := "trigram_flat"
	if trigramDim != D {
		prog.MatMul(trigramState, weightName(wi), "trigram_proj")
		wi++
		trigramState = "trigram_proj"
	}
	prog.Mul(trigramState, weightName(wi), "trigram_scaled")
	wi++
	prog.Add(baseState, "trigram_scaled", "x")
	return wi
}
