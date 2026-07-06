package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestExportHFMultiheadExportsScorerOnly(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_multihead",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31,
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
				{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
				 "diffusion": {"block_size": 2, "timestep_conditioning": "adaln", "timestep_conditioning_dim": 5}}
			]
		}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfCfg hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfCfg)
	if hfCfg.MLMHead != "bert" {
		t.Fatalf("mlm_head=%q, want bert", hfCfg.MLMHead)
	}
	if _, ok := hfCfg.AutoMap["AutoModelForMaskedLM"]; !ok {
		t.Fatalf("AutoModelForMaskedLM missing from auto_map: %+v", hfCfg.AutoMap)
	}
	var mapping []hfWeightMapping
	readJSONFileForTest(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, want := range []string{"final_norm.weight", "mlm_head_dense.weight", "mlm_head_dense.bias", "mlm_head_output_bias", "lm_head_weight"} {
		if !containsHFWeight(mapping, want) {
			t.Fatalf("weight_map missing %q: %+v", want, mapping)
		}
	}
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "denoiser") || strings.Contains(entry.Mixlab, "adaln") || strings.Contains(entry.HF, "denoiser") {
			t.Fatalf("native-only denoiser/AdaLN weight exported: %+v", entry)
		}
	}
}

func TestExportHFMultiheadAllowsMinimalPairMLMSpanPLL(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_multihead_pll_ranking",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31,
			"export_head": "scorer",
			"minimal_pair": {
				"path": "pairs.jsonl",
				"energy_aggregation": "differing_span",
				"score_source": "mlm_span_pll",
				"score_head": "scorer"
			},
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
				{"name": "aux", "objective": "causal", "loss_weight": 0.3}
			]
		}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfCfg hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfCfg)
	if _, ok := hfCfg.AutoMap["AutoModelForMaskedLM"]; !ok {
		t.Fatalf("AutoModelForMaskedLM missing from auto_map: %+v", hfCfg.AutoMap)
	}
	var mapping []hfWeightMapping
	readJSONFileForTest(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "minimal_pair") || strings.Contains(entry.HF, "minimal_pair") {
			t.Fatalf("minimal-pair training-only mapping exported: %+v", entry)
		}
	}
}

func TestExportHFMultiheadExportsHeadLevelDWA(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, `{
		"name": "hf_multihead_dwa",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "geglu"}
		],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31,
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7, "layer_aggregation": "dwa"},
				{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
				 "diffusion": {"block_size": 2}}
			]
		}
	}`, func(weights [][]float32, shapes []WeightShape) error {
		idx := weightShapeIndex(shapes, "head_scorer_dwa_alpha")
		if idx < 0 {
			return os.ErrNotExist
		}
		copy(weights[idx], []float32{0.125, 0.25, 0.375, 0.5})
		return nil
	})
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfCfg hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfCfg)
	if hfCfg.LayerAggregation != "dwa" {
		t.Fatalf("layer_aggregation=%q, want dwa", hfCfg.LayerAggregation)
	}
	if hfCfg.LayerAggregationScope != "head" {
		t.Fatalf("layer_aggregation_scope=%q, want head", hfCfg.LayerAggregationScope)
	}
	var mapping []hfWeightMapping
	readJSONFileForTest(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, want := range []string{"dwa_alphas.0", "dwa_alphas.1", "dwa_alphas.2"} {
		if !containsHFWeight(mapping, want) {
			t.Fatalf("weight_map missing %q: %+v", want, mapping)
		}
	}
	hfWeights, err := loadHFWeightsForParity(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("loadHFWeightsForParity: %v", err)
	}
	want := [][]float64{
		{0.125, 0.25},
		{0.125, 0.25, 0.375},
		{0.125, 0.25, 0.375, 0.5},
	}
	names := []string{"dwa_alphas.0", "dwa_alphas.1", "dwa_alphas.2"}
	for i, wantAlpha := range want {
		name := names[i]
		got := hfWeights[name]
		if len(got) != len(wantAlpha) {
			t.Fatalf("%s length=%d want %d", name, len(got), len(wantAlpha))
		}
		for j := range wantAlpha {
			if math.Abs(got[j]-wantAlpha[j]) > 1e-7 {
				t.Fatalf("%s[%d]=%g want %g", name, j, got[j], wantAlpha[j])
			}
		}
	}
}

func TestExportHFMultiheadMaskedScorerParityCPUOracle(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, `{
		"name": "hf_multihead_masked_scorer_parity",
		"model_dim": 8,
		"vocab_size": 13,
		"seq_len": 5,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 3, "relative_attention_parameterization": "shared_qk_reuse"},
			{"type": "swiglu"}
		],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 10,
			"mlm_mask_token_id": 1,
			"export_head": "scorer",
			"minimal_pair": {
				"path": "pairs.jsonl",
				"energy_aggregation": "differing_span",
				"score_source": "mlm_span_pll",
				"score_head": "scorer"
			},
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7, "layer_aggregation": "dwa", "output_head": "bert_mlm", "tie_embeddings": true, "final_norm": true},
				{"name": "aux", "objective": "causal", "loss_weight": 0.3, "final_norm": true}
			]
		}
	}`, func(weights [][]float32, shapes []WeightShape) error {
		if err := scaleHFExportWeightsToTrainedMagnitude(weights, shapes); err != nil {
			return err
		}
		idx := weightShapeIndex(shapes, "head_scorer_dwa_alpha")
		if idx < 0 {
			return os.ErrNotExist
		}
		copy(weights[idx], []float32{0.125, 0.25, 0.375, 0.5})
		return nil
	})
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	nativeShapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes(native): %v", err)
	}
	nativeWeights, err := loadSafetensorsWeights(weightsPath, nativeShapes)
	if err != nil {
		t.Fatalf("load native weights: %v", err)
	}

	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfDoc hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfDoc)
	if hfDoc.MLMHead != "bert" {
		t.Fatalf("mlm_head=%q, want bert", hfDoc.MLMHead)
	}
	if hfDoc.LayerAggregation != "dwa" || hfDoc.LayerAggregationScope != "head" {
		t.Fatalf("layer aggregation=%q scope=%q, want dwa/head", hfDoc.LayerAggregation, hfDoc.LayerAggregationScope)
	}
	if len(hfDoc.MaskedBlocks) != len(cfg.Blocks) {
		t.Fatalf("masked_blocks=%d, want %d", len(hfDoc.MaskedBlocks), len(cfg.Blocks))
	}
	if got := hfDoc.MaskedBlocks[0]["attention_mask"]; got != "bidirectional" {
		t.Fatalf("masked attention_mask=%v, want bidirectional", got)
	}
	hfWeights, err := loadHFWeightsForParity(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load HF weights: %v", err)
	}
	hfCfg := hfExportInferenceConfig(cfg)
	hfCfg.Blocks = decodeHFBlockSpecs(t, hfDoc.MaskedBlocks)

	tokens := [][]int{
		{2, cfg.Training.MLMMaskTokenID, 4, 5, 6},
		{3, 4, cfg.Training.MLMMaskTokenID, 7, 8},
	}
	nativeLogits := runNativeMultiheadScorerCPUMaskedForward(t, cfg, nativeWeights, tokens)
	hfLogits := runHFCPUMaskedForwardWithDWAScope(t, hfCfg, hfWeights, tokens, hfDoc.LayerAggregationScope)
	if diff := maxAbsDiff3D(nativeLogits, hfLogits); diff >= 1e-3 {
		t.Fatalf("native multihead masked scorer vs HF MaskedLM max logit diff=%g, want < 1e-3", diff)
	}

	plain := []int{2, 3, 4, 5, 6}
	nativeScore, err := scoreEBMFullSeqPLLSequence(cfg, &multiheadCPUPLLParityEvaluator{
		t:       t,
		cfg:     cfg,
		weights: nativeWeights,
	}, plain, 2, map[int]bool{cfg.Training.MLMMaskTokenID: true})
	if err != nil {
		t.Fatalf("scoreEBMFullSeqPLLSequence: %v", err)
	}
	hfScore := hfMaskedPLLScoreCPU(t, hfCfg, hfWeights, hfDoc.LayerAggregationScope, plain, cfg.Training.MLMMaskTokenID)
	if math.Abs(nativeScore-hfScore) >= 1e-4 {
		t.Fatalf("native score-ebm full_seq score=%g HF masked PLL score=%g delta=%g, want < 1e-4", nativeScore, hfScore, math.Abs(nativeScore-hfScore))
	}
}

type multiheadCPUPLLParityEvaluator struct {
	t       *testing.T
	cfg     *ArchConfig
	weights [][]float32
	logits  []float32
}

func (e *multiheadCPUPLLParityEvaluator) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	return e.EvaluateObjectiveGPUWithOutputs(batch, batchSize, seqLen, nil)
}

func (e *multiheadCPUPLLParityEvaluator) EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, _ []string) (float32, error) {
	if e == nil || e.t == nil || e.cfg == nil {
		return 0, os.ErrInvalid
	}
	headName, err := mlmSpanPLLScoreHeadName(e.cfg)
	if err != nil {
		return 0, err
	}
	headIdx := -1
	for i, head := range e.cfg.Training.Heads {
		if head.Name == headName {
			headIdx = i
			break
		}
	}
	if headIdx < 0 {
		return 0, os.ErrNotExist
	}
	rowsPerRawBatch := len(e.cfg.Training.Heads)
	if minimalPairUsesMLMSpanPLL(e.cfg) {
		rowsPerRawBatch++
	}
	if batchSize%rowsPerRawBatch != 0 {
		return 0, os.ErrInvalid
	}
	rawBatch := batchSize / rowsPerRawBatch
	rowOffset := headIdx * rawBatch * seqLen
	if len(batch.x) < rowOffset+rawBatch*seqLen {
		return 0, os.ErrInvalid
	}
	tokens := make([][]int, rawBatch)
	for row := 0; row < rawBatch; row++ {
		start := rowOffset + row*seqLen
		tokens[row] = append([]int(nil), batch.x[start:start+seqLen]...)
	}
	logits := runNativeMultiheadScorerCPUMaskedForward(e.t, e.cfg, e.weights, tokens)
	e.logits = flatten3DFloat32(logits)
	return 0, nil
}

func (e *multiheadCPUPLLParityEvaluator) ReadOutput(name string, shape []int) ([]float32, error) {
	if name != "head_scorer_logits" {
		return nil, os.ErrNotExist
	}
	want := 1
	for _, dim := range shape {
		want *= dim
	}
	if len(e.logits) != want {
		return nil, os.ErrInvalid
	}
	return append([]float32(nil), e.logits...), nil
}

func flatten3DFloat32(x [][][]float64) []float32 {
	if len(x) == 0 || len(x[0]) == 0 || len(x[0][0]) == 0 {
		return nil
	}
	out := make([]float32, 0, len(x)*len(x[0])*len(x[0][0]))
	for b := range x {
		for t := range x[b] {
			for _, v := range x[b][t] {
				out = append(out, float32(v))
			}
		}
	}
	return out
}

func hfMaskedPLLScoreCPU(t *testing.T, cfg *ArchConfig, weights map[string][]float64, scope string, tokens []int, maskTokenID int) float64 {
	t.Helper()
	var total float64
	skip := map[int]bool{maskTokenID: true}
	for _, pos := range scoreEBMFullSeqPLLPositions(tokens, skip) {
		masked := append([]int(nil), tokens...)
		masked[pos] = maskTokenID
		logits := runHFCPUMaskedForwardWithDWAScope(t, cfg, weights, [][]int{masked}, scope)
		lp, err := targetLogProbFromFloat64Logits(logits[0][pos], tokens[pos])
		if err != nil {
			t.Fatalf("target logprob pos=%d: %v", pos, err)
		}
		total += lp
	}
	return total
}

func targetLogProbFromFloat64Logits(logits []float64, target int) (float64, error) {
	if target < 0 || target >= len(logits) {
		return 0, os.ErrInvalid
	}
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(v - maxLogit)
	}
	return logits[target] - maxLogit - math.Log(sum), nil
}

func runNativeMultiheadScorerCPUMaskedForward(t *testing.T, cfg *ArchConfig, weights [][]float32, tokens [][]int) [][][]float64 {
	t.Helper()
	head := cfg.Training.MultiheadExportHead()
	if head == nil {
		t.Fatal("multihead config has no export head")
	}
	if head.Objective != arch.ObjectiveMLM && head.Objective != arch.ObjectiveMNTP {
		t.Fatalf("export head objective=%q, want MLM/MNTP", head.Objective)
	}
	w := make([][]float64, len(weights))
	for i := range weights {
		w[i] = toFloat64(weights[i])
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weightByName := func(name string) []float64 {
		idx := weightShapeIndex(shapes, name)
		if idx < 0 {
			t.Fatalf("missing weight %q", name)
		}
		return w[idx]
	}

	x := embedCPU(w[0], cfg.VocabSize, cfg.ModelDim, tokens)
	wi := 1
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

	var dwa *dwaCPUState
	if head.LayerAggregation == arch.LayerAggregationDWA {
		count, err := archDWAHeadPointCount(cfg.Blocks)
		if err != nil {
			t.Fatalf("DWA point count: %v", err)
		}
		alpha := weightByName("head_" + head.Name + "_dwa_alpha")
		alphas := make([][]float64, count)
		for i := 0; i < count; i++ {
			alphas[i] = alpha[:i+2]
		}
		dwa = newDWACPUState(alphas, x)
		dwa.captureOnly = true
	}

	for blockIdx, block := range cfg.Blocks {
		blockForMaskedRows := block
		if strings.ToLower(strings.TrimSpace(blockForMaskedRows.Type)) == "plain" {
			blockForMaskedRows.AttentionMask = arch.AttentionMaskBidirectional
		}
		switch strings.ToLower(strings.TrimSpace(blockForMaskedRows.Type)) {
		case "plain":
			n := plainHFExportWeightCount(blockForMaskedRows, cfg.EffectiveNormPlacement())
			x = plainCPUForward(t, cfg, blockForMaskedRows, x, w[wi:wi+n], sharedRelativeEmbeddings, dwa)
			wi += n
		case "swiglu":
			x = gatedGLUCPUForward(t, cfg, x, w[wi:wi+4], "sigmoid")
			if dwa != nil {
				x = dwa.apply(t, x)
			}
			wi += 4
		case "geglu":
			x = gatedGLUCPUForward(t, cfg, x, w[wi:wi+4], "gelu")
			if dwa != nil {
				x = dwa.apply(t, x)
			}
			wi += 4
		case "mlp":
			x = mlpCPUForward(t, cfg, blockForMaskedRows, x, w[wi:wi+3])
			if dwa != nil {
				x = dwa.apply(t, x)
			}
			wi += 3
		case "moe":
			n := moeHFExportWeightCount(blockForMaskedRows)
			x = moeCPUForward(t, cfg, blockForMaskedRows, x, w[wi:wi+n])
			if dwa != nil {
				x = dwa.apply(t, x)
			}
			wi += n
		default:
			t.Fatalf("unsupported block %q at %d", blockForMaskedRows.Type, blockIdx)
		}
	}
	if head.LayerAggregation == arch.LayerAggregationDWA {
		x = dwa.finishHead(t)
	}
	if head.FinalNorm {
		x = rmsNormCPU(x, weightByName("head_"+head.Name+"_final_norm_scale"))
	}
	switch head.OutputHead {
	case arch.MultiheadOutputBERTMLM:
		return bertMLMHeadCPU(cfg, x,
			weightByName("head_"+head.Name+"_mlm_dense"),
			weightByName("head_"+head.Name+"_mlm_dense_bias"),
			w[0],
			weightByName("head_"+head.Name+"_mlm_output_bias"),
		)
	case arch.MultiheadOutputLinear:
		if head.TieEmbeddings {
			return matmul3DCPU(x, transposeEmbeddingToHead64(w[0], cfg.VocabSize, cfg.ModelDim), cfg.ModelDim, cfg.VocabSize)
		}
		return matmul3DCPU(x, weightByName("head_"+head.Name+"_proj"), cfg.ModelDim, cfg.VocabSize)
	default:
		t.Fatalf("unsupported scorer output_head=%q", head.OutputHead)
		return nil
	}
}

func archDWAHeadPointCount(blocks []BlockSpec) (int, error) {
	count := 0
	for i, block := range blocks {
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			count += 2
		case "swiglu", "geglu", "mlp", "moe":
			count++
		default:
			return 0, os.ErrInvalid
		}
		_ = i
	}
	return count, nil
}

func transposeEmbeddingToHead64(embed []float64, vocab, dim int) []float64 {
	head := make([]float64, dim*vocab)
	for v := 0; v < vocab; v++ {
		for d := 0; d < dim; d++ {
			head[d*vocab+v] = embed[v*dim+d]
		}
	}
	return head
}

func readJSONFileForTest(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}
