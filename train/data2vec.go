package train

import (
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

type data2VecTeacher struct {
	cfg              *ArchConfig
	prog             *gpu.Program
	handles          []int64
	shapes           []WeightShape
	emaWeights       [][]float32
	outputNames      []string
	outputSizes      []int
	modelDim         int
	batchTokens      int
	vocabSize        int
	charMaxPerToken  int
	bigramVocabSize  int
	trigramVocabSize int
	lossMaskInput    bool
	segmentIDsInput  bool
	tokBuf           []int32
	tgtBuf           []int32
	lossMaskBuf      []float32
	segmentIDBuf     []int32
	charBuf          []int32
	bigramBuf        []int32
	trigramBuf       []int32
	charFeatures     []int32
	targetBuf        []float32
	maskBuf          []float32
}

func newData2VecTeacher(cfg *ArchConfig, initialWeights [][]float32, concreteObjective string) (*data2VecTeacher, error) {
	if cfg == nil || !cfg.Training.Data2VecActive() {
		return nil, nil
	}
	teacherProg, err := arch.BuildData2VecTeacherIRProgramFromConfig(cfg, concreteObjective)
	if err != nil {
		return nil, fmt.Errorf("build data2vec teacher IR: %w", err)
	}
	gpuProg, err := gpu.LowerIRProgram(teacherProg)
	if err != nil {
		return nil, fmt.Errorf("lower data2vec teacher IR: %w", err)
	}
	allShapes, err := computeWeightShapes(cfg)
	if err != nil {
		gpuProg.Destroy()
		return nil, fmt.Errorf("compute data2vec weight shapes: %w", err)
	}
	if len(initialWeights) < teacherProg.NumWeights {
		gpuProg.Destroy()
		return nil, fmt.Errorf("initial weight count=%d smaller than teacher program weight count=%d", len(initialWeights), teacherProg.NumWeights)
	}
	teacherShapes := allShapes[:teacherProg.NumWeights]
	ema := cloneWeights(initialWeights[:teacherProg.NumWeights])
	handles, err := uploadWeightHandles(teacherShapes, ema)
	if err != nil {
		gpuProg.Destroy()
		return nil, err
	}
	outputNames, outputSizes, err := data2VecHiddenOutputs(teacherProg)
	if err != nil {
		gpu.FreeHandles(handles)
		gpuProg.Destroy()
		return nil, err
	}
	out := &data2VecTeacher{
		cfg:              cfg,
		prog:             gpuProg,
		handles:          handles,
		shapes:           teacherShapes,
		emaWeights:       ema,
		outputNames:      outputNames,
		outputSizes:      outputSizes,
		modelDim:         cfg.ModelDim,
		batchTokens:      cfg.Training.BatchTokens,
		vocabSize:        cfg.VocabSize,
		charMaxPerToken:  cfg.CharMaxPerToken,
		bigramVocabSize:  cfg.BigramVocabSize,
		trigramVocabSize: cfg.TrigramVocabSize,
		lossMaskInput:    data2VecProgramDeclaresInput(teacherProg, "loss_mask"),
		segmentIDsInput:  data2VecProgramDeclaresInput(teacherProg, "segment_ids"),
		tokBuf:           make([]int32, cfg.Training.BatchTokens),
		tgtBuf:           make([]int32, cfg.Training.BatchTokens),
		lossMaskBuf:      make([]float32, cfg.Training.BatchTokens),
		segmentIDBuf:     make([]int32, cfg.Training.BatchTokens),
		charBuf:          make([]int32, cfg.Training.BatchTokens*cfg.CharMaxPerToken),
		bigramBuf:        make([]int32, cfg.Training.BatchTokens),
		trigramBuf:       make([]int32, cfg.Training.BatchTokens),
		charFeatures:     append([]int32(nil), cfg.CharFeatureIDs...),
		targetBuf:        make([]float32, cfg.Training.BatchTokens*cfg.ModelDim),
		maskBuf:          make([]float32, cfg.Training.BatchTokens),
	}
	fmt.Printf("  [%s] data2vec: top_k_layers=%d tau=%.6f->%.6f ramp_steps=%d target_norm=%s mask_source=%s\n",
		cfg.Name, cfg.Training.Data2Vec.TopKLayers, cfg.Training.Data2Vec.EMATauStart,
		cfg.Training.Data2Vec.EMATauEnd, cfg.Training.Data2Vec.EMATauRampSteps,
		cfg.Training.Data2Vec.TargetNorm, cfg.Training.Data2Vec.MaskSource)
	return out, nil
}

func data2VecTeacherObjective(cfg *ArchConfig, concreteObjective string) string {
	if concreteObjective == arch.ObjectiveHybridExample && cfg != nil {
		return cfg.Training.EffectiveHybridSecondaryObjective()
	}
	if arch.IsMaskedTrainingObjectiveForData2Vec(concreteObjective) {
		return concreteObjective
	}
	if cfg != nil && cfg.Training.EffectiveObjective() == arch.ObjectiveHybrid {
		return cfg.Training.EffectiveHybridSecondaryObjective()
	}
	return concreteObjective
}

func data2VecHiddenOutputs(prog *arch.Program) ([]string, []int, error) {
	var names []string
	var sizes []int
	for _, out := range prog.Outputs {
		if len(out.Name) < len("data2vec_layer_") || out.Name[:len("data2vec_layer_")] != "data2vec_layer_" {
			continue
		}
		size := 1
		for _, dim := range out.Shape {
			if dim <= 0 {
				return nil, nil, fmt.Errorf("invalid data2vec output %q shape %v", out.Name, out.Shape)
			}
			size *= dim
		}
		names = append(names, out.Name)
		sizes = append(sizes, size)
	}
	if len(names) == 0 {
		return nil, nil, fmt.Errorf("data2vec teacher program declared no hidden outputs")
	}
	return names, sizes, nil
}

func data2VecProgramDeclaresInput(prog *arch.Program, name string) bool {
	if prog == nil {
		return false
	}
	for _, in := range prog.Inputs {
		if in.Name == name {
			return true
		}
	}
	return false
}

func (t *data2VecTeacher) Close() {
	if t == nil {
		return
	}
	if t.prog != nil {
		t.prog.Destroy()
		t.prog = nil
	}
	if len(t.handles) > 0 {
		gpu.FreeHandles(t.handles)
		t.handles = nil
	}
}

func attachData2VecTargets(t *data2VecTeacher, batch objectiveBatch, objective string, batchSize, seqLen int) (objectiveBatch, error) {
	if t == nil {
		return batch, nil
	}
	need := batchSize * seqLen
	if !arch.IsMaskedTrainingObjectiveForData2Vec(objective) {
		batch.data2vecTargets = t.zeroTargets(need)
		batch.data2vecMask = t.zeroMask(need)
		return batch, nil
	}
	if len(batch.lossMask) < need {
		return objectiveBatch{}, fmt.Errorf("data2vec objective batch missing loss mask")
	}
	x := batch.unmaskedX
	if len(x) < need {
		x = batch.x
	}
	data2vecMask := batch.lossMask
	if len(batch.maskedLossMask) >= need {
		data2vecMask = batch.maskedLossMask
	}
	targets, mask, err := t.targets(x, batch.y, data2vecMask, batchSize, seqLen)
	if err != nil {
		return objectiveBatch{}, err
	}
	batch.data2vecTargets = targets
	batch.data2vecMask = mask
	return batch, nil
}

func (t *data2VecTeacher) targets(xTok, yTok []int, mask []float32, batchSize, seqLen int) ([]float32, []float32, error) {
	need := batchSize * seqLen
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return nil, nil, err
	}
	outs, err := gpu.EvalProgramOutputs(t.prog, t.handles, inputs, t.outputNames, t.outputSizes)
	if err != nil {
		return nil, nil, err
	}
	targets := t.targetBuffer(need)
	if err := fillData2VecTargetsFromOutputs(
		targets,
		t.outputNames,
		outs,
		need,
		t.modelDim,
		t.cfg.Training.Data2Vec.TargetNorm,
		float32(t.cfg.Training.Data2Vec.TargetNormEps),
	); err != nil {
		return nil, nil, err
	}
	outMask := t.maskBuffer(need)
	copy(outMask, mask[:need])
	return targets, outMask, nil
}

func fillData2VecTargetsFromOutputs(targets []float32, outputNames []string, outs map[string][]float32, tokens, modelDim int, targetNorm string, eps float32) error {
	need := tokens * modelDim
	if tokens < 0 || modelDim <= 0 {
		return fmt.Errorf("invalid data2vec target shape: tokens=%d model_dim=%d", tokens, modelDim)
	}
	if len(targets) < need {
		return fmt.Errorf("data2vec target buffer size=%d want at least %d", len(targets), need)
	}
	if len(outputNames) == 0 {
		return fmt.Errorf("data2vec teacher program declared no hidden outputs")
	}
	targets = targets[:need]
	clear(targets)
	for _, name := range outputNames {
		out, ok := outs[name]
		if !ok {
			return fmt.Errorf("data2vec output %q missing", name)
		}
		if len(out) != need {
			return fmt.Errorf("data2vec output %q size=%d want %d", name, len(out), need)
		}
		for i, v := range out {
			targets[i] += v
		}
	}
	scale := float32(1.0 / float64(len(outputNames)))
	for i := range targets {
		targets[i] *= scale
	}
	if targetNorm != arch.Data2VecTargetNormNone {
		normalizeRowsInPlace(targets, tokens, modelDim, eps)
	}
	return nil
}

func (t *data2VecTeacher) zeroTargets(tokens int) []float32 {
	targets := t.targetBuffer(tokens)
	clear(targets)
	return targets
}

func (t *data2VecTeacher) zeroMask(tokens int) []float32 {
	mask := t.maskBuffer(tokens)
	clear(mask)
	return mask
}

func (t *data2VecTeacher) targetBuffer(tokens int) []float32 {
	need := tokens * t.modelDim
	if len(t.targetBuf) < need {
		t.targetBuf = make([]float32, need)
	}
	return t.targetBuf[:need]
}

func (t *data2VecTeacher) maskBuffer(tokens int) []float32 {
	if len(t.maskBuf) < tokens {
		t.maskBuf = make([]float32, tokens)
	}
	return t.maskBuf[:tokens]
}

func (t *data2VecTeacher) updateFromStudentWeights(weights [][]float32, step int) error {
	if t == nil {
		return nil
	}
	if len(weights) < len(t.emaWeights) {
		return fmt.Errorf("student weight count=%d smaller than data2vec EMA count=%d", len(weights), len(t.emaWeights))
	}
	updateData2VecEMAWeights(t.emaWeights, weights[:len(t.emaWeights)], t.cfg.Training.Data2Vec, step)
	gpu.FreeHandles(t.handles)
	handles, err := uploadWeightHandles(t.shapes, t.emaWeights)
	if err != nil {
		t.handles = nil
		return err
	}
	t.handles = handles
	return nil
}

func (t *data2VecTeacher) restoreEMAWeights(weights [][]float32) error {
	if t == nil {
		if len(weights) != 0 {
			return fmt.Errorf("checkpoint contains data2vec EMA state but training.data2vec is disabled")
		}
		return nil
	}
	if len(weights) != len(t.emaWeights) {
		return fmt.Errorf("data2vec EMA weight count mismatch: checkpoint=%d trainer=%d", len(weights), len(t.emaWeights))
	}
	restored := make([][]float32, len(weights))
	for i := range weights {
		want := shapeProduct(t.shapes[i].Shape)
		if len(weights[i]) != want {
			return fmt.Errorf("data2vec EMA weight %d size=%d want=%d", i, len(weights[i]), want)
		}
		restored[i] = append([]float32(nil), weights[i]...)
	}
	handles, err := uploadWeightHandles(t.shapes, restored)
	if err != nil {
		return err
	}
	gpu.FreeHandles(t.handles)
	t.handles = handles
	t.emaWeights = restored
	return nil
}

func updateData2VecEMAWeights(ema, current [][]float32, spec *arch.Data2VecSpec, step int) {
	tau := float32(data2VecTauForStep(spec, step))
	updateEMAWeights(ema, current, tau)
}

func data2VecTauForStep(spec *arch.Data2VecSpec, step int) float64 {
	if spec == nil {
		return 0.999
	}
	if spec.EMATauRampSteps <= 0 {
		return spec.EMATau
	}
	if step <= 0 {
		return spec.EMATauStart
	}
	frac := float64(step) / float64(spec.EMATauRampSteps)
	if frac > 1 {
		frac = 1
	}
	return spec.EMATauStart + (spec.EMATauEnd-spec.EMATauStart)*frac
}

func normalizeRowsInPlace(x []float32, rows, cols int, eps float32) {
	if cols <= 0 {
		return
	}
	for r := 0; r < rows; r++ {
		start := r * cols
		end := start + cols
		var mean float64
		for _, v := range x[start:end] {
			mean += float64(v)
		}
		mean /= float64(cols)
		var variance float64
		for _, v := range x[start:end] {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(cols)
		invStd := float32(1.0 / math.Sqrt(variance+float64(eps)))
		for i := start; i < end; i++ {
			x[i] = (x[i] - float32(mean)) * invStd
		}
	}
}

func (t *data2VecTeacher) makeInputs(xTok, yTok []int, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	need := batchSize * seqLen
	if len(xTok) < need || len(yTok) < need {
		return nil, fmt.Errorf("input size mismatch: tokens=%d targets=%d need=%d", len(xTok), len(yTok), need)
	}
	if len(t.tokBuf) < need {
		t.tokBuf = make([]int32, need)
		t.tgtBuf = make([]int32, need)
		t.charBuf = make([]int32, need*t.charMaxPerToken)
		t.bigramBuf = make([]int32, need)
		t.trigramBuf = make([]int32, need)
	}
	if t.segmentIDsInput && len(t.segmentIDBuf) < need {
		t.segmentIDBuf = make([]int32, need)
	}
	for i := 0; i < need; i++ {
		t.tokBuf[i] = int32(xTok[i])
		t.tgtBuf[i] = int32(yTok[i])
	}
	inputs := []gpu.TensorInput{
		{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.tokBuf[:need]},
		{Name: "targets", DType: gpu.TensorInt32, Shape: []int{need}, Data: t.tgtBuf[:need]},
	}
	if t.lossMaskInput {
		if len(t.lossMaskBuf) < need {
			t.lossMaskBuf = make([]float32, need)
		}
		for i := 0; i < need; i++ {
			t.lossMaskBuf[i] = 1
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.lossMaskBuf[:need],
		})
	}
	if t.segmentIDsInput {
		ids := deriveSegmentIDs(xTok[:need], need, seqLen, t.cfg.Training.AttentionSegmentBoundaryTokenID)
		copy(t.segmentIDBuf[:need], ids)
		inputs = append(inputs, gpu.TensorInput{
			Name: "segment_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.segmentIDBuf[:need],
		})
	}
	if t.charMaxPerToken > 0 {
		want := t.vocabSize * t.charMaxPerToken
		if len(t.charFeatures) != want {
			return nil, fmt.Errorf("char feature lookup size=%d does not match vocab_size*char_max_per_token=%d", len(t.charFeatures), want)
		}
		if len(t.charBuf) < need*t.charMaxPerToken {
			t.charBuf = make([]int32, need*t.charMaxPerToken)
		}
		for i := 0; i < need; i++ {
			tok := int(t.tokBuf[i])
			if tok < 0 || tok >= t.vocabSize {
				return nil, fmt.Errorf("token id %d at position %d outside vocab_size=%d", tok, i, t.vocabSize)
			}
			copy(t.charBuf[i*t.charMaxPerToken:(i+1)*t.charMaxPerToken], t.charFeatures[tok*t.charMaxPerToken:(tok+1)*t.charMaxPerToken])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "char_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen, t.charMaxPerToken}, Data: t.charBuf[:need*t.charMaxPerToken],
		})
	}
	if t.bigramVocabSize > 0 {
		bigramIDs, err := ComputeBigramIDs(t.tokBuf[:need], need, t.bigramVocabSize)
		if err != nil {
			return nil, err
		}
		copy(t.bigramBuf[:need], bigramIDs)
		inputs = append(inputs, gpu.TensorInput{
			Name: "bigram_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.bigramBuf[:need],
		})
	}
	if t.trigramVocabSize > 0 {
		trigramIDs, err := ComputeTrigramIDs(t.tokBuf[:need], batchSize, seqLen, t.trigramVocabSize)
		if err != nil {
			return nil, err
		}
		copy(t.trigramBuf[:need], trigramIDs)
		inputs = append(inputs, gpu.TensorInput{
			Name: "trigram_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.trigramBuf[:need],
		})
	}
	return inputs, nil
}
