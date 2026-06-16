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
	for blockIdx, block := range cfg.Blocks {
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			n := plainHFExportWeightCount(block)
			x = plainCPUForward(t, cfg, block, x, w[wi:wi+n])
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
	for i, block := range cfg.Blocks {
		prefix := fmt.Sprintf("blocks.%d.", i)
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			blockWeights := [][]float64{
				weights[prefix+"norm.weight"],
				weights[prefix+"wq.weight"],
				weights[prefix+"wk.weight"],
				weights[prefix+"wv.weight"],
			}
			if block.QKNorm {
				blockWeights = append(blockWeights,
					weights[prefix+"q_norm.weight"],
					weights[prefix+"k_norm.weight"],
				)
			}
			if relativeAttentionEnabledForHF(block) {
				blockWeights = append(blockWeights,
					weights[prefix+"relative_embeddings"],
					weights[prefix+"w_pos_key.weight"],
					weights[prefix+"w_pos_query.weight"],
				)
			}
			if block.QKGain > 0 {
				blockWeights = append(blockWeights, weights[prefix+"qk_gain"])
			}
			blockWeights = append(blockWeights,
				weights[prefix+"wo.weight"],
				weights[prefix+"ff1.weight"],
				weights[prefix+"ff2.weight"],
			)
			x = plainCPUForward(t, cfg, block, x, blockWeights)
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

func plainHFExportWeightCount(block BlockSpec) int {
	n := 7
	if block.QKNorm {
		n += 2
	}
	if relativeAttentionEnabledForHF(block) {
		n += 3
	}
	if block.QKGain > 0 {
		n++
	}
	return n
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

func plainCPUForward(t *testing.T, cfg *ArchConfig, block BlockSpec, x [][][]float64, w [][]float64) [][][]float64 {
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
	q := matmul3DCPU(xNorm, w[1], dim, dim)
	k := matmul3DCPU(xNorm, w[2], dim, kvHeads*headDim)
	v := matmul3DCPU(xNorm, w[3], dim, kvHeads*headDim)
	qh := splitHeadsCPU(q, heads, headDim)
	khKV := splitHeadsCPU(k, kvHeads, headDim)
	vhKV := splitHeadsCPU(v, kvHeads, headDim)
	kh := repeatKVHeadsCPU(khKV, heads, groupSize)
	vh := repeatKVHeadsCPU(vhKV, heads, groupSize)
	woIndex := 4
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
		relKey, relQuery = relativeAttentionProjectionCPU(t, block, w[woIndex], w[woIndex+1], w[woIndex+2], dim, heads, headDim, relWindow)
		woIndex += 3
	}
	if block.QKGain > 0 {
		qkGain = w[woIndex]
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
						c2p := clipRelativeIndex(i-j+relWindow, relWindow)
						p2c := clipRelativeIndex(j-i+relWindow, relWindow)
						scores[j] += dot(qh[b][h][i], relKey[h][c2p]) + dot(kh[b][h][j], relQuery[h][p2c])
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
	merged := mergeHeadsCPU(ctx, dim)
	proj := matmul3DCPU(merged, w[woIndex], dim, dim)
	x = add3D(x, proj)
	ff1 := matmul3DCPU(x, w[woIndex+1], dim, ffnDimForConfig(cfg))
	for b := range ff1 {
		for i := range ff1[b] {
			for d := range ff1[b][i] {
				ff1[b][i][d] = silu(ff1[b][i][d])
			}
		}
	}
	ff2 := matmul3DCPU(ff1, w[woIndex+2], ffnDimForConfig(cfg), dim)
	return add3D(x, ff2)
}

func relativeAttentionProjectionCPU(t *testing.T, block BlockSpec, embeddings, wPosKey, wPosQuery []float64, dim, heads, headDim, window int) ([][][]float64, [][][]float64) {
	t.Helper()
	rows := 2 * window
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

func clipRelativeIndex(v, window int) int {
	maxV := 2*window - 1
	if v < 0 {
		return 0
	}
	if v > maxV {
		return maxV
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

func rmsNormCPU(x [][][]float64, scale []float64) [][][]float64 {
	out := make3D(len(x), len(x[0]), len(x[0][0]))
	for b := range x {
		for t := range x[b] {
			sum := 0.0
			for _, v := range x[b][t] {
				sum += v * v
			}
			inv := 1.0 / math.Sqrt(sum/float64(len(x[b][t]))+1e-5)
			for d, v := range x[b][t] {
				out[b][t][d] = v * inv * scale[d]
			}
		}
	}
	return out
}

func rmsNormHeadsCPU(x [][][][]float64, scale []float64) {
	for b := range x {
		for h := range x[b] {
			for t := range x[b][h] {
				sum := 0.0
				for _, v := range x[b][h][t] {
					sum += v * v
				}
				inv := 1.0 / math.Sqrt(sum/float64(len(x[b][h][t]))+1e-5)
				for d, v := range x[b][h][t] {
					x[b][h][t][d] = v * inv * scale[d]
				}
			}
		}
	}
}

func matmul3DCPU(x [][][]float64, w []float64, inDim, outDim int) [][][]float64 {
	out := make3D(len(x), len(x[0]), outDim)
	for b := range x {
		for t := range x[b] {
			for o := 0; o < outDim; o++ {
				sum := 0.0
				for i := 0; i < inDim; i++ {
					sum += x[b][t][i] * w[i*outDim+o]
				}
				out[b][t][o] = sum
			}
		}
	}
	return out
}

func matmul2DCPU(x []float64, w []float64, rows, inDim, outDim int) []float64 {
	out := make([]float64, rows*outDim)
	for r := 0; r < rows; r++ {
		for o := 0; o < outDim; o++ {
			sum := 0.0
			for i := 0; i < inDim; i++ {
				sum += x[r*inDim+i] * w[i*outDim+o]
			}
			out[r*outDim+o] = sum
		}
	}
	return out
}

func matmulRowFloat64(x []float64, w []float64, inDim, outDim int) []float64 {
	out := make([]float64, outDim)
	for o := 0; o < outDim; o++ {
		sum := 0.0
		for i := 0; i < inDim; i++ {
			sum += x[i] * w[i*outDim+o]
		}
		out[o] = sum
	}
	return out
}

func softmaxRowsFloat64(logits []float64, rows, cols int) []float64 {
	out := make([]float64, len(logits))
	for r := 0; r < rows; r++ {
		row := logits[r*cols : (r+1)*cols]
		maxV := row[0]
		for _, v := range row[1:] {
			if v > maxV {
				maxV = v
			}
		}
		sum := 0.0
		for c, v := range row {
			out[r*cols+c] = math.Exp(v - maxV)
			sum += out[r*cols+c]
		}
		for c := 0; c < cols; c++ {
			out[r*cols+c] /= sum
		}
	}
	return out
}

func flatten3DCPU(x [][][]float64) []float64 {
	out := make([]float64, 0, len(x)*len(x[0])*len(x[0][0]))
	for b := range x {
		for t := range x[b] {
			out = append(out, x[b][t]...)
		}
	}
	return out
}

func unflatten3DCPU(flat []float64, batch, seqLen, dim int) [][][]float64 {
	out := make3D(batch, seqLen, dim)
	for b := 0; b < batch; b++ {
		for t := 0; t < seqLen; t++ {
			start := (b*seqLen + t) * dim
			copy(out[b][t], flat[start:start+dim])
		}
	}
	return out
}

func splitHeadsCPU(x [][][]float64, heads, headDim int) [][][][]float64 {
	out := make4D(len(x), heads, len(x[0]), headDim)
	for b := range x {
		for t := range x[b] {
			for h := 0; h < heads; h++ {
				copy(out[b][h][t], x[b][t][h*headDim:(h+1)*headDim])
			}
		}
	}
	return out
}

func repeatKVHeadsCPU(x [][][][]float64, heads, groupSize int) [][][][]float64 {
	if groupSize <= 1 {
		return x
	}
	out := make4D(len(x), heads, len(x[0][0]), len(x[0][0][0]))
	for b := range x {
		for h := 0; h < heads; h++ {
			src := h / groupSize
			for t := range x[b][src] {
				copy(out[b][h][t], x[b][src][t])
			}
		}
	}
	return out
}

func mergeHeadsCPU(x [][][][]float64, dim int) [][][]float64 {
	batch := len(x)
	heads := len(x[0])
	seqLen := len(x[0][0])
	headDim := len(x[0][0][0])
	out := make3D(batch, seqLen, dim)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for t := 0; t < seqLen; t++ {
				copy(out[b][t][h*headDim:(h+1)*headDim], x[b][h][t])
			}
		}
	}
	return out
}

func scale3D(a [][][]float64, scale float64) [][][]float64 {
	out := make3D(len(a), len(a[0]), len(a[0][0]))
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				out[i][j][k] = a[i][j][k] * scale
			}
		}
	}
	return out
}

func add3D(a, b [][][]float64) [][][]float64 {
	out := make3D(len(a), len(a[0]), len(a[0][0]))
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				out[i][j][k] = a[i][j][k] + b[i][j][k]
			}
		}
	}
	return out
}

func cpuCrossEntropy(logits [][][]float64, targets [][]int) float64 {
	total := 0.0
	count := 0
	for b := range logits {
		for t := range logits[b] {
			row := logits[b][t]
			maxV := row[0]
			for _, v := range row[1:] {
				if v > maxV {
					maxV = v
				}
			}
			sum := 0.0
			for _, v := range row {
				sum += math.Exp(v - maxV)
			}
			total += -(row[targets[b][t]] - maxV - math.Log(sum))
			count++
		}
	}
	return total / float64(count)
}

func halfRotationRoPEForTest(x []float64, seqLen, headDim, ropeDims int) []float64 {
	out := append([]float64(nil), x...)
	if ropeDims <= 0 || ropeDims >= headDim {
		ropeDims = headDim
	}
	half := ropeDims / 2
	for t := 0; t < seqLen; t++ {
		for d := 0; d < half; d++ {
			i := t*headDim + d
			j := t*headDim + d + half
			freq := math.Exp(float64(d) * (-math.Log(10000.0) * 2.0 / float64(ropeDims)))
			angle := float64(t) * freq
			c := math.Cos(angle)
			s := math.Sin(angle)
			a := x[i]
			b := x[j]
			out[i] = a*c - b*s
			out[j] = b*c + a*s
		}
	}
	return out
}

func hfAdjacentRoPEForTest(x []float64, positions, heads, seqLen, headDim, ropeDims int) []float64 {
	out := append([]float64(nil), x...)
	if ropeDims <= 0 || ropeDims >= headDim {
		ropeDims = headDim
	}
	pairs := ropeDims / 2
	for h := 0; h < heads; h++ {
		for t := 0; t < seqLen; t++ {
			for p := 0; p < pairs; p++ {
				base := ((h*seqLen + t) * headDim) + p*2
				even := x[base]
				odd := x[base+1]
				freq := math.Exp(float64(p) * (-math.Log(10000.0) * 2.0 / float64(ropeDims)))
				angle := float64(positions+t) * freq
				c := math.Cos(angle)
				s := math.Sin(angle)
				out[base] = even*c - odd*s
				out[base+1] = even*s + odd*c
			}
		}
	}
	return out
}

func softmaxCPU(x []float64) []float64 {
	maxV := math.Inf(-1)
	for _, v := range x {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		if math.IsInf(v, -1) {
			out[i] = 0
			continue
		}
		out[i] = math.Exp(v - maxV)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func maxAbsDiff3D(a, b [][][]float64) float64 {
	maxDiff := 0.0
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				if diff := math.Abs(a[i][j][k] - b[i][j][k]); diff > maxDiff {
					maxDiff = diff
				}
			}
		}
	}
	return maxDiff
}

func make3D(a, b, c int) [][][]float64 {
	out := make([][][]float64, a)
	for i := range out {
		out[i] = make([][]float64, b)
		for j := range out[i] {
			out[i][j] = make([]float64, c)
		}
	}
	return out
}

func flattenTokensInt32(tokens [][]int) []int32 {
	out := make([]int32, 0, len(tokens)*len(tokens[0]))
	for b := range tokens {
		for _, tok := range tokens[b] {
			out = append(out, int32(tok))
		}
	}
	return out
}

func make4D(a, b, c, d int) [][][][]float64 {
	out := make([][][][]float64, a)
	for i := range out {
		out[i] = make([][][]float64, b)
		for j := range out[i] {
			out[i][j] = make([][]float64, c)
			for k := range out[i][j] {
				out[i][j][k] = make([]float64, d)
			}
		}
	}
	return out
}

func flattenHead(x [][]float64) []float64 {
	out := make([]float64, 0, len(x)*len(x[0]))
	for i := range x {
		out = append(out, x[i]...)
	}
	return out
}

func unflattenHead(flat []float64, out [][]float64) {
	width := len(out[0])
	for i := range out {
		copy(out[i], flat[i*width:(i+1)*width])
	}
}

func toFloat64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func ffnDimForConfig(cfg *ArchConfig) int {
	ffn := int(math.Round(float64(cfg.ModelDim) * cfg.EffectiveMLPMult()))
	if ffn < cfg.ModelDim {
		return cfg.ModelDim
	}
	return ffn
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func silu(x float64) float64 {
	return x / (1.0 + math.Exp(-x))
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
}
