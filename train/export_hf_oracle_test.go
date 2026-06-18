package train

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"
)

func runNativeCPUForward(t *testing.T, cfg *ArchConfig, weights [][]float32, tokens [][]int) [][][]float64 {
	t.Helper()
	w := make([][]float64, len(weights))
	for i := range weights {
		w[i] = toFloat64(weights[i])
	}
	x := embedCPU(w[0], cfg.VocabSize, cfg.ModelDim, tokens)
	wi := 3
	if cfg.CharVocabSize > 0 {
		var next int
		x, next = addCharFeaturesCPU(t, cfg, x, tokens, w, wi)
		wi = next
	}
	if cfg.BigramVocabSize > 0 {
		var next int
		x, next = addBigramFeaturesCPU(t, cfg, x, tokens, w, wi)
		wi = next
	}
	if cfg.TrigramVocabSize > 0 {
		var next int
		x, next = addTrigramFeaturesCPU(t, cfg, x, tokens, w, wi)
		wi = next
	}
	var sharedRelativeEmbeddings []float64
	if hfConfigUsesSharedRelativeAttention(cfg) {
		sharedRelativeEmbeddings = w[wi]
		wi++
		if hfConfigUsesSharedRelativeEmbeddingNorm(cfg) {
			rows := len(sharedRelativeEmbeddings) / cfg.ModelDim
			sharedRelativeEmbeddings = layerNorm2DCPU(sharedRelativeEmbeddings, w[wi], w[wi+1], rows, cfg.ModelDim, float64(cfg.EffectiveNormSpec().Eps))
			wi += 2
		}
	}
	for blockIdx, block := range cfg.Blocks {
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			n := plainHFExportWeightCount(block, cfg.EffectiveNormPlacement())
			x = plainCPUForward(t, cfg, block, x, w[wi:wi+n], sharedRelativeEmbeddings)
			wi += n
		case "moe":
			n := moeHFExportWeightCount(block)
			x = moeCPUForward(t, cfg, block, x, w[wi:wi+n])
			wi += n
		case "swiglu":
			x = gatedGLUCPUForward(t, cfg, x, w[wi:wi+4], "sigmoid")
			wi += 4
		case "geglu":
			x = gatedGLUCPUForward(t, cfg, x, w[wi:wi+4], "gelu")
			wi += 4
		case "mlp":
			x = mlpCPUForward(t, cfg, block, x, w[wi:wi+3])
			wi += 3
		default:
			t.Fatalf("unsupported native parity block %q at %d", block.Type, blockIdx)
		}
	}
	x = rmsNormCPU(x, w[2])
	return matmul3DCPU(x, w[1], cfg.ModelDim, cfg.VocabSize)
}

func runHFCPUForward(t *testing.T, cfg *ArchConfig, weights map[string][]float64, tokens [][]int) [][][]float64 {
	t.Helper()
	x := embedCPU(weights["embed_tokens.weight"], cfg.VocabSize, cfg.ModelDim, tokens)
	if cfg.CharVocabSize > 0 {
		x, _ = addCharFeaturesCPU(t, cfg, x, tokens, hfFeatureWeights(weights, "char"))
	}
	if cfg.BigramVocabSize > 0 {
		x, _ = addBigramFeaturesCPU(t, cfg, x, tokens, hfFeatureWeights(weights, "bigram"))
	}
	if cfg.TrigramVocabSize > 0 {
		x, _ = addTrigramFeaturesCPU(t, cfg, x, tokens, hfFeatureWeights(weights, "trigram"))
	}
	sharedRelativeEmbeddings := []float64(nil)
	if hfConfigUsesSharedRelativeAttention(cfg) {
		sharedRelativeEmbeddings = weights["relative_embeddings"]
		if hfConfigUsesSharedRelativeEmbeddingNorm(cfg) {
			rows := len(sharedRelativeEmbeddings) / cfg.ModelDim
			sharedRelativeEmbeddings = layerNorm2DCPU(sharedRelativeEmbeddings, weights["relative_layer_norm.weight"], weights["relative_layer_norm.bias"], rows, cfg.ModelDim, float64(cfg.EffectiveNormSpec().Eps))
		}
	}
	normPlacement := cfg.EffectiveNormPlacement()
	for i, block := range cfg.Blocks {
		prefix := fmt.Sprintf("blocks.%d.", i)
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			blockWeights := [][]float64{
				weights[prefix+"norm.weight"],
				weights[prefix+"wq.weight"],
			}
			if block.AttnBias {
				blockWeights = append(blockWeights, weights[prefix+"wq.bias"])
			}
			blockWeights = append(blockWeights, weights[prefix+"wk.weight"])
			if block.AttnBias {
				blockWeights = append(blockWeights, weights[prefix+"wk.bias"])
			}
			blockWeights = append(blockWeights, weights[prefix+"wv.weight"])
			if block.AttnBias {
				blockWeights = append(blockWeights, weights[prefix+"wv.bias"])
			}
			if block.QKNorm {
				blockWeights = append(blockWeights,
					weights[prefix+"q_norm.weight"],
					weights[prefix+"k_norm.weight"],
				)
			}
			if relativeAttentionEnabledForHF(block) && !hfRelativeAttentionUsesSharedQKReuse(block) {
				blockWeights = append(blockWeights,
					weights[prefix+"relative_embeddings"],
					weights[prefix+"w_pos_key.weight"],
					weights[prefix+"w_pos_query.weight"],
				)
			}
			if block.QKGain > 0 {
				blockWeights = append(blockWeights, weights[prefix+"qk_gain"])
			}
			if block.SparseAttnGate {
				blockWeights = append(blockWeights, weights[prefix+"attn_gate_w"])
			}
			attnPostNorm := hfEffectivePlainAttnPostNorm(block, normPlacement)
			if attnPostNorm == "before_outproj" {
				blockWeights = append(blockWeights, weights[prefix+"post_attn_norm.weight"])
			}
			blockWeights = append(blockWeights, weights[prefix+"wo.weight"])
			if block.AttnBias {
				blockWeights = append(blockWeights, weights[prefix+"wo.bias"])
			}
			if attnPostNorm == "after_outproj" {
				blockWeights = append(blockWeights, weights[prefix+"post_attn_norm.weight"])
			}
			if hfPlainFFNActivationUsesGate(block) {
				blockWeights = append(blockWeights, weights[prefix+"ff_gate.weight"])
			}
			blockWeights = append(blockWeights,
				weights[prefix+"ff1.weight"],
				weights[prefix+"ff2.weight"],
			)
			x = plainCPUForward(t, cfg, block, x, blockWeights, sharedRelativeEmbeddings)
		case "swiglu":
			x = gatedGLUCPUForward(t, cfg, x, [][]float64{
				weights[prefix+"norm.weight"],
				weights[prefix+"w_gate.weight"],
				weights[prefix+"w_up.weight"],
				weights[prefix+"w_down.weight"],
			}, "sigmoid")
		case "geglu":
			x = gatedGLUCPUForward(t, cfg, x, [][]float64{
				weights[prefix+"norm.weight"],
				weights[prefix+"w_gate.weight"],
				weights[prefix+"w_up.weight"],
				weights[prefix+"w_down.weight"],
			}, "gelu")
		case "mlp":
			x = mlpCPUForward(t, cfg, block, x, [][]float64{
				weights[prefix+"norm.weight"],
				weights[prefix+"w_up.weight"],
				weights[prefix+"w_down.weight"],
			})
		case "moe":
			blockWeights := [][]float64{
				weights[prefix+"norm.weight"],
				weights[prefix+"router_w"],
			}
			expert := BlockSpec{Type: "swiglu"}
			if block.ExpertBlock != nil {
				expert = *block.ExpertBlock
			}
			expertType := strings.ToLower(strings.TrimSpace(expert.Type))
			if expertType == "" {
				expertType = "swiglu"
			}
			for e := 0; e < block.NumExperts; e++ {
				expertPrefix := fmt.Sprintf("%sexperts.%d.", prefix, e)
				switch expertType {
				case "swiglu", "geglu":
					blockWeights = append(blockWeights,
						weights[expertPrefix+"w_gate.weight"],
						weights[expertPrefix+"w_up.weight"],
						weights[expertPrefix+"w_down.weight"],
					)
				case "mlp":
					blockWeights = append(blockWeights,
						weights[expertPrefix+"w_up.weight"],
						weights[expertPrefix+"w_down.weight"],
					)
				default:
					t.Fatalf("unsupported MoE expert %q", expertType)
				}
			}
			x = moeCPUForward(t, cfg, block, x, blockWeights)
		default:
			t.Fatalf("unsupported HF parity block %q at %d", block.Type, i)
		}
	}
	x = rmsNormCPU(x, weights["final_norm.weight"])
	return matmul3DCPU(x, weights["lm_head_weight"], cfg.ModelDim, cfg.VocabSize)
}

func hfFeatureWeights(weights map[string][]float64, feature string) [][]float64 {
	out := [][]float64{weights[feature+"_table.weight"]}
	if proj := weights[feature+"_proj.weight"]; proj != nil {
		out = append(out, proj)
	}
	out = append(out, weights[feature+"_scale"])
	return out
}

func plainHFExportWeightCount(block BlockSpec, normPlacement string) int {
	n := 7
	if block.AttnBias {
		n += 4
	}
	if block.QKNorm {
		n += 2
	}
	if relativeAttentionEnabledForHF(block) && !hfRelativeAttentionUsesSharedQKReuse(block) {
		n += 3
	}
	if block.QKGain > 0 {
		n++
	}
	if block.SparseAttnGate {
		n++
	}
	if hfEffectivePlainAttnPostNorm(block, normPlacement) != "none" {
		n++
	}
	if hfPlainFFNActivationUsesGate(block) {
		n++
	}
	return n
}

func hfPlainSparseAttnGateWidth(dim int) int {
	if dim <= 0 {
		return 1
	}
	if dim < 12 {
		return dim
	}
	return 12
}

func moeHFExportWeightCount(block BlockSpec) int {
	expert := BlockSpec{Type: "swiglu"}
	if block.ExpertBlock != nil {
		expert = *block.ExpertBlock
	}
	switch strings.ToLower(strings.TrimSpace(expert.Type)) {
	case "", "swiglu", "geglu":
		return 2 + block.NumExperts*3
	case "mlp":
		return 2 + block.NumExperts*2
	default:
		return 0
	}
}

func embedCPU(table []float64, vocab, dim int, tokens [][]int) [][][]float64 {
	out := make3D(len(tokens), len(tokens[0]), dim)
	for b := range tokens {
		for t := range tokens[b] {
			id := tokens[b][t]
			copy(out[b][t], table[id*dim:(id+1)*dim])
		}
	}
	_ = vocab
	return out
}

func plainCPUForward(t *testing.T, cfg *ArchConfig, block BlockSpec, x [][][]float64, w [][]float64, sharedRelativeEmbeddings []float64) [][][]float64 {
	t.Helper()
	batch := len(x)
	seqLen := len(x[0])
	dim := cfg.ModelDim
	heads := block.Heads
	headDim := dim / heads
	kvHeads := block.KVHeads
	if kvHeads <= 0 {
		kvHeads = heads
	}
	if heads%kvHeads != 0 {
		t.Fatalf("invalid heads=%d kv_heads=%d", heads, kvHeads)
	}
	groupSize := heads / kvHeads
	xNorm := rmsNormCPU(x, w[0])
	wi := 1
	qWeight := w[wi]
	wi++
	var qBias []float64
	if block.AttnBias {
		qBias = w[wi]
		wi++
	}
	kWeight := w[wi]
	wi++
	var kBias []float64
	if block.AttnBias {
		kBias = w[wi]
		wi++
	}
	vWeight := w[wi]
	wi++
	var vBias []float64
	if block.AttnBias {
		vBias = w[wi]
		wi++
	}
	valueDim := kvHeads * headDim
	vProjDim := valueDim
	if block.AttnValueGate {
		vProjDim += dim
	}
	q := matmul3DBiasCPU(xNorm, qWeight, qBias, dim, dim)
	k := matmul3DBiasCPU(xNorm, kWeight, kBias, dim, valueDim)
	vProj := matmul3DBiasCPU(xNorm, vWeight, vBias, dim, vProjDim)
	v := slice3DLastDimCPU(vProj, 0, valueDim)
	var valueGate [][][]float64
	if block.AttnValueGate {
		valueGate = slice3DLastDimCPU(vProj, valueDim, valueDim+dim)
		gelu3DInPlace(valueGate)
	}
	qh := splitHeadsCPU(q, heads, headDim)
	khKV := splitHeadsCPU(k, kvHeads, headDim)
	vhKV := splitHeadsCPU(v, kvHeads, headDim)
	kh := repeatKVHeadsCPU(khKV, heads, groupSize)
	vh := repeatKVHeadsCPU(vhKV, heads, groupSize)
	woIndex := wi
	if block.QKNorm {
		rmsNormHeadsCPU(qh, w[woIndex])
		woIndex++
		rmsNormHeadsCPU(kh, w[woIndex])
		woIndex++
	}
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			if !relativeAttentionEnabledForHF(block) {
				flatQ := flattenHead(qh[b][h])
				flatK := flattenHead(kh[b][h])
				var rotQ, rotK []float64
				if normalizeHFRopeConvention(block.RopeConvention) == "half_rotation" {
					rotQ = halfRotationRoPEForTest(flatQ, seqLen, headDim, block.RopeDims)
					rotK = halfRotationRoPEForTest(flatK, seqLen, headDim, block.RopeDims)
				} else {
					rotQ = hfAdjacentRoPEForTest(flatQ, 0, 1, seqLen, headDim, block.RopeDims)
					rotK = hfAdjacentRoPEForTest(flatK, 0, 1, seqLen, headDim, block.RopeDims)
				}
				unflattenHead(rotQ, qh[b][h])
				unflattenHead(rotK, kh[b][h])
			}
		}
	}
	qkGain := []float64(nil)
	var relKey, relQuery [][][]float64
	relWindow := effectiveHFRelativeAttentionWindow(block)
	if relativeAttentionEnabledForHF(block) {
		if hfRelativeAttentionUsesSharedQKReuse(block) {
			relKey, relQuery = sharedRelativeAttentionProjectionCPU(t, block, sharedRelativeEmbeddings, kWeight, kBias, qWeight, qBias, dim, heads, kvHeads, headDim, relWindow)
		} else {
			relKey, relQuery = relativeAttentionProjectionCPU(t, block, w[woIndex], w[woIndex+1], w[woIndex+2], dim, heads, headDim, relWindow)
			woIndex += 3
		}
	}
	if block.QKGain > 0 {
		qkGain = w[woIndex]
		woIndex++
	}
	var attnGateW []float64
	if block.SparseAttnGate {
		attnGateW = w[woIndex]
		woIndex++
	}
	mask := strings.ToLower(strings.TrimSpace(block.AttentionMask))
	if mask == "" {
		mask = "causal"
	}
	ctx := make4D(batch, heads, seqLen, headDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for i := 0; i < seqLen; i++ {
				scores := make([]float64, seqLen)
				for j := 0; j < seqLen; j++ {
					if mask == "causal" && (j > i || (block.WindowSize > 0 && block.WindowSize < seqLen && j < i-(block.WindowSize-1))) {
						scores[j] = math.Inf(-1)
						continue
					}
					scale := math.Sqrt(float64(headDim))
					if relativeAttentionEnabledForHF(block) {
						scale = math.Sqrt(float64(headDim * 3))
					}
					scores[j] = dot(qh[b][h][i], kh[b][h][j])
					if relativeAttentionEnabledForHF(block) {
						relIdx := gptBertRelativeBucketIndex(i-j, relWindow, seqLen)
						scores[j] += dot(qh[b][h][i], relKey[h][relIdx]) + dot(kh[b][h][j], relQuery[h][relIdx])
					}
					scores[j] /= scale
					if qkGain != nil {
						scores[j] *= qkGain[h]
					}
				}
				probs := softmaxCPU(scores)
				for j := 0; j < seqLen; j++ {
					for d := 0; d < headDim; d++ {
						ctx[b][h][i][d] += probs[j] * vh[b][h][j][d]
					}
				}
			}
		}
	}
	if block.XSA {
		for b := 0; b < batch; b++ {
			for h := 0; h < heads; h++ {
				for i := 0; i < seqLen; i++ {
					dotYV := dot(ctx[b][h][i], vh[b][h][i])
					dotVV := dot(vh[b][h][i], vh[b][h][i])
					scale := dotYV / (dotVV + 1e-8)
					for d := 0; d < headDim; d++ {
						ctx[b][h][i][d] -= scale * vh[b][h][i][d]
					}
				}
			}
		}
	}
	if attnGateW != nil {
		gateDim := hfPlainSparseAttnGateWidth(dim)
		for b := 0; b < batch; b++ {
			for h := 0; h < heads; h++ {
				for i := 0; i < seqLen; i++ {
					logit := 0.0
					for d := 0; d < gateDim; d++ {
						logit += x[b][i][d] * attnGateW[h*gateDim+d]
					}
					gate := 1.0 / (1.0 + math.Exp(-logit))
					for d := 0; d < headDim; d++ {
						ctx[b][h][i][d] *= gate
					}
				}
			}
		}
	}
	merged := mergeHeadsCPU(ctx, dim)
	if valueGate != nil {
		mul3DInPlace(merged, valueGate)
	}
	attnPostNorm := hfEffectivePlainAttnPostNorm(block, cfg.EffectiveNormPlacement())
	if attnPostNorm == "before_outproj" {
		merged = rmsNormCPU(merged, w[woIndex])
		woIndex++
	}
	projWeight := w[woIndex]
	woIndex++
	var projBias []float64
	if block.AttnBias {
		projBias = w[woIndex]
		woIndex++
	}
	proj := matmul3DBiasCPU(merged, projWeight, projBias, dim, dim)
	if attnPostNorm == "after_outproj" {
		proj = rmsNormCPU(proj, w[woIndex])
		woIndex++
	}
	x = add3D(x, proj)
	ffWeightIndex := woIndex
	var ffHidden [][][]float64
	switch hfPlainFFNActivation(block) {
	case "silu":
		ffHidden = matmul3DCPU(x, w[ffWeightIndex], dim, ffnDimForConfig(cfg))
		ffWeightIndex++
		for b := range ffHidden {
			for i := range ffHidden[b] {
				for d := range ffHidden[b][i] {
					ffHidden[b][i][d] = silu(ffHidden[b][i][d])
				}
			}
		}
	case "geglu", "swiglu":
		gate := matmul3DCPU(x, w[ffWeightIndex], dim, ffnDimForConfig(cfg))
		ffWeightIndex++
		up := matmul3DCPU(x, w[ffWeightIndex], dim, ffnDimForConfig(cfg))
		ffWeightIndex++
		ffHidden = gate
		for b := range ffHidden {
			for i := range ffHidden[b] {
				for d := range ffHidden[b][i] {
					if hfPlainFFNActivation(block) == "geglu" {
						ffHidden[b][i][d] = gelu(ffHidden[b][i][d])
					} else {
						ffHidden[b][i][d] = silu(ffHidden[b][i][d])
					}
					ffHidden[b][i][d] *= up[b][i][d]
				}
			}
		}
	default:
		t.Fatalf("unsupported plain ffn_activation %q", block.FFNActivation)
	}
	ff2 := matmul3DCPU(ffHidden, w[ffWeightIndex], ffnDimForConfig(cfg), dim)
	return add3D(x, ff2)
}

func relativeAttentionProjectionCPU(t *testing.T, block BlockSpec, embeddings, wPosKey, wPosQuery []float64, dim, heads, headDim, window int) ([][][]float64, [][][]float64) {
	t.Helper()
	rows := 2*window - 1
	keyFlat := matmul2DCPU(embeddings, wPosKey, rows, dim, dim)
	queryFlat := matmul2DCPU(embeddings, wPosQuery, rows, dim, dim)
	key := make3D(heads, rows, headDim)
	query := make3D(heads, rows, headDim)
	for r := 0; r < rows; r++ {
		for h := 0; h < heads; h++ {
			copy(key[h][r], keyFlat[r*dim+h*headDim:r*dim+(h+1)*headDim])
			copy(query[h][r], queryFlat[r*dim+h*headDim:r*dim+(h+1)*headDim])
		}
	}
	_ = block
	return key, query
}

func sharedRelativeAttentionProjectionCPU(t *testing.T, block BlockSpec, embeddings, wk, wkBias, wq, wqBias []float64, dim, heads, kvHeads, headDim, window int) ([][][]float64, [][][]float64) {
	t.Helper()
	if embeddings == nil {
		t.Fatal("missing shared relative embeddings")
	}
	if kvHeads <= 0 {
		kvHeads = heads
	}
	if heads%kvHeads != 0 {
		t.Fatalf("invalid shared relative heads=%d kv_heads=%d", heads, kvHeads)
	}
	rows := 2*window - 1
	keyFlat := matmul2DBiasCPU(embeddings, wk, wkBias, rows, dim, kvHeads*headDim)
	queryFlat := matmul2DBiasCPU(embeddings, wq, wqBias, rows, dim, dim)
	key := make3D(heads, rows, headDim)
	query := make3D(heads, rows, headDim)
	groupSize := heads / kvHeads
	for r := 0; r < rows; r++ {
		for h := 0; h < heads; h++ {
			srcH := h / groupSize
			copy(key[h][r], keyFlat[r*kvHeads*headDim+srcH*headDim:r*kvHeads*headDim+(srcH+1)*headDim])
			copy(query[h][r], queryFlat[r*dim+h*headDim:r*dim+(h+1)*headDim])
		}
	}
	_ = block
	return key, query
}

func gptBertRelativeBucketIndex(rel, bucketSize, maxPosition int) int {
	if bucketSize <= 1 {
		return 0
	}
	mid := bucketSize / 2
	absPos := 0
	if rel < mid && rel > -mid {
		absPos = mid - 1
	} else {
		absPos = absInt(rel)
		if max := maxPosition - 1; absPos > max {
			absPos = max
		}
	}
	bucketPos := rel
	if absPos > mid {
		logPos := bucketSize - 1
		if mid > 0 && maxPosition-1 > mid {
			denom := math.Log(float64(maxPosition-1) / float64(mid))
			if denom > 0 && !math.IsInf(denom, 0) && !math.IsNaN(denom) {
				scaled := math.Log(float64(absPos)/float64(mid)) / denom * float64(mid-1)
				logPos = int(math.Ceil(scaled)) + mid
			}
		}
		if rel < 0 {
			bucketPos = -logPos
		} else {
			bucketPos = logPos
		}
	}
	maxBucket := bucketSize - 1
	if bucketPos < -maxBucket {
		bucketPos = -maxBucket
	}
	if bucketPos > maxBucket {
		bucketPos = maxBucket
	}
	return bucketPos + maxBucket
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func moeCPUForward(t *testing.T, cfg *ArchConfig, block BlockSpec, x [][][]float64, w [][]float64) [][][]float64 {
	t.Helper()
	batch := len(x)
	seqLen := len(x[0])
	rows := batch * seqLen
	dim := cfg.ModelDim
	experts := block.NumExperts
	topK := effectiveHFMoETopK(block)
	xNorm := rmsNormCPU(x, w[0])
	flat := flatten3DCPU(xNorm)
	router := matmul2DCPU(flat, w[1], rows, dim, experts)
	probs := softmaxRowsFloat64(router, rows, experts)
	expert := BlockSpec{Type: "swiglu"}
	if block.ExpertBlock != nil {
		expert = *block.ExpertBlock
	}
	expertType := strings.ToLower(strings.TrimSpace(expert.Type))
	if expertType == "" {
		expertType = "swiglu"
	}
	perExpert := 3
	if expertType == "mlp" {
		perExpert = 2
	}
	deltaFlat := make([]float64, rows*dim)
	ffn := ffnDimForConfig(cfg)
	for r := 0; r < rows; r++ {
		order := make([]int, experts)
		for e := 0; e < experts; e++ {
			order[e] = e
		}
		sort.SliceStable(order, func(i, j int) bool {
			return probs[r*experts+order[i]] > probs[r*experts+order[j]]
		})
		denom := 0.0
		for k := 0; k < topK; k++ {
			denom += probs[r*experts+order[k]]
		}
		if denom == 0 {
			denom = 1
		}
		row := flat[r*dim : (r+1)*dim]
		for k := 0; k < topK; k++ {
			e := order[k]
			weight := probs[r*experts+e] / denom
			base := 2 + e*perExpert
			var expertOut []float64
			switch expertType {
			case "swiglu", "geglu":
				gate := matmulRowFloat64(row, w[base], dim, ffn)
				up := matmulRowFloat64(row, w[base+1], dim, ffn)
				for i := range gate {
					if expertType == "swiglu" {
						gate[i] = sigmoid(gate[i])
					} else {
						gate[i] = gelu(gate[i])
					}
					gate[i] *= up[i]
				}
				expertOut = matmulRowFloat64(gate, w[base+2], ffn, dim)
			case "mlp":
				up := matmulRowFloat64(row, w[base], dim, ffn)
				act := strings.ToLower(strings.TrimSpace(expert.Activation))
				if act == "" {
					act = "silu"
				}
				slope := expert.LeakySlope
				if slope == 0 {
					slope = 0.5
				}
				for i := range up {
					switch act {
					case "silu":
						up[i] = silu(up[i])
					case "gelu":
						up[i] = gelu(up[i])
					case "relu":
						if up[i] < 0 {
							up[i] = 0
						}
					case "leaky_relu_sq":
						if up[i] < 0 {
							up[i] *= slope
						}
						up[i] *= up[i]
					default:
						t.Fatalf("unsupported MoE MLP activation %q", act)
					}
				}
				expertOut = matmulRowFloat64(up, w[base+1], ffn, dim)
			default:
				t.Fatalf("unsupported MoE expert type %q", expertType)
			}
			for d := 0; d < dim; d++ {
				deltaFlat[r*dim+d] += weight * expertOut[d]
			}
		}
	}
	return add3D(x, unflatten3DCPU(deltaFlat, batch, seqLen, dim))
}

func gatedGLUCPUForward(t *testing.T, cfg *ArchConfig, x [][][]float64, w [][]float64, gateActivation string) [][][]float64 {
	t.Helper()
	dim := cfg.ModelDim
	ffn := ffnDimForConfig(cfg)
	xNorm := rmsNormCPU(x, w[0])
	gate := matmul3DCPU(xNorm, w[1], dim, ffn)
	up := matmul3DCPU(xNorm, w[2], dim, ffn)
	for b := range gate {
		for i := range gate[b] {
			for d := range gate[b][i] {
				switch gateActivation {
				case "sigmoid":
					gate[b][i][d] = sigmoid(gate[b][i][d]) * up[b][i][d]
				case "gelu":
					gate[b][i][d] = gelu(gate[b][i][d]) * up[b][i][d]
				default:
					t.Fatalf("unsupported gate activation %q", gateActivation)
				}
			}
		}
	}
	down := matmul3DCPU(gate, w[3], ffn, dim)
	return add3D(x, down)
}

func mlpCPUForward(t *testing.T, cfg *ArchConfig, block BlockSpec, x [][][]float64, w [][]float64) [][][]float64 {
	t.Helper()
	dim := cfg.ModelDim
	ffn := ffnDimForConfig(cfg)
	xNorm := rmsNormCPU(x, w[0])
	up := matmul3DCPU(xNorm, w[1], dim, ffn)
	act := strings.ToLower(strings.TrimSpace(block.Activation))
	if act == "" {
		act = "silu"
	}
	slope := block.LeakySlope
	if slope == 0 {
		slope = 0.5
	}
	for b := range up {
		for i := range up[b] {
			for d := range up[b][i] {
				switch act {
				case "silu":
					up[b][i][d] = silu(up[b][i][d])
				case "gelu":
					up[b][i][d] = gelu(up[b][i][d])
				case "relu":
					if up[b][i][d] < 0 {
						up[b][i][d] = 0
					}
				case "leaky_relu_sq":
					v := up[b][i][d]
					if v < 0 {
						v *= slope
					}
					up[b][i][d] = v * v
				default:
					t.Fatalf("unsupported mlp activation %q", act)
				}
			}
		}
	}
	down := matmul3DCPU(up, w[2], ffn, dim)
	return add3D(x, down)
}

func addCharFeaturesCPU(t *testing.T, cfg *ArchConfig, x [][][]float64, tokens [][]int, w [][]float64, wi ...int) ([][][]float64, int) {
	t.Helper()
	start := 0
	if len(wi) > 0 {
		start = wi[0]
	}
	if len(cfg.CharFeatureIDs) != cfg.VocabSize*cfg.EffectiveCharMaxPerToken() {
		t.Fatalf("char feature ids len=%d want %d", len(cfg.CharFeatureIDs), cfg.VocabSize*cfg.EffectiveCharMaxPerToken())
	}
	charDim := cfg.EffectiveCharDim()
	table := w[start]
	next := start + 1
	var proj []float64
	if charDim != cfg.ModelDim {
		proj = w[next]
		next++
	}
	scale := w[next][0]
	next++
	state := make3D(len(tokens), len(tokens[0]), charDim)
	for b := range tokens {
		for pos, tok := range tokens[b] {
			for slot := 0; slot < cfg.EffectiveCharMaxPerToken(); slot++ {
				id := int(cfg.CharFeatureIDs[tok*cfg.EffectiveCharMaxPerToken()+slot])
				if id == 0 {
					continue
				}
				for d := 0; d < charDim; d++ {
					state[b][pos][d] += table[id*charDim+d]
				}
			}
		}
	}
	if proj != nil {
		state = matmul3DCPU(state, proj, charDim, cfg.ModelDim)
	}
	return add3D(x, scale3D(state, scale)), next
}

func addBigramFeaturesCPU(t *testing.T, cfg *ArchConfig, x [][][]float64, tokens [][]int, w [][]float64, wi ...int) ([][][]float64, int) {
	t.Helper()
	start := 0
	if len(wi) > 0 {
		start = wi[0]
	}
	flat := flattenTokensInt32(tokens)
	ids, err := ComputeBigramIDs(flat, len(flat), cfg.BigramVocabSize)
	if err != nil {
		t.Fatalf("ComputeBigramIDs: %v", err)
	}
	return addDenseFeatureCPU(cfg, x, ids, w, start, cfg.BigramVocabSize, cfg.EffectiveBigramDim())
}

func addTrigramFeaturesCPU(t *testing.T, cfg *ArchConfig, x [][][]float64, tokens [][]int, w [][]float64, wi ...int) ([][][]float64, int) {
	t.Helper()
	start := 0
	if len(wi) > 0 {
		start = wi[0]
	}
	flat := flattenTokensInt32(tokens)
	ids, err := ComputeTrigramIDs(flat, len(tokens), len(tokens[0]), cfg.TrigramVocabSize)
	if err != nil {
		t.Fatalf("ComputeTrigramIDs: %v", err)
	}
	return addDenseFeatureCPU(cfg, x, ids, w, start, cfg.TrigramVocabSize, cfg.EffectiveTrigramDim())
}

func addDenseFeatureCPU(cfg *ArchConfig, x [][][]float64, ids []int32, w [][]float64, start, vocab, featureDim int) ([][][]float64, int) {
	table := w[start]
	next := start + 1
	var proj []float64
	if featureDim != cfg.ModelDim {
		proj = w[next]
		next++
	}
	scale := w[next][0]
	next++
	state := make3D(len(x), len(x[0]), featureDim)
	for b := range x {
		for t := range x[b] {
			id := int(ids[b*len(x[b])+t])
			copy(state[b][t], table[id*featureDim:(id+1)*featureDim])
		}
	}
	_ = vocab
	if proj != nil {
		state = matmul3DCPU(state, proj, featureDim, cfg.ModelDim)
	}
	return add3D(x, scale3D(state, scale)), next
}
