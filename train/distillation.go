package train

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

type distillationEnsemble struct {
	spec        *DistillationSpec
	teachers    []*distillationTeacher
	strategy    string
	vocabSize   int
	batchTokens int
	objective   string
	probBuf     []float32
}

type distillationTeacher struct {
	name             string
	cfg              *ArchConfig
	prog             *gpu.Program
	handles          []int64
	vocabSize        int
	lossMaskInput    bool
	causalMaskInput  bool
	charMaxPerToken  int
	bigramVocabSize  int
	trigramVocabSize int
	tokBuf           []int32
	tgtBuf           []int32
	lossMaskBuf      []float32
	causalMaskBuf    []int32
	charBuf          []int32
	bigramBuf        []int32
	trigramBuf       []int32
	charFeatures     []int32
}

func newDistillationEnsemble(student *ArchConfig) (*distillationEnsemble, error) {
	if student == nil || !student.Training.DistillationKLEffectiveActive() {
		return nil, nil
	}
	spec := student.Training.Distillation
	teacherObjective := distillationTeacherObjective(student)
	out := &distillationEnsemble{
		spec:        spec,
		strategy:    spec.EffectiveEnsembleStrategy(),
		vocabSize:   student.VocabSize,
		batchTokens: student.Training.BatchTokens,
		objective:   teacherObjective,
		teachers:    make([]*distillationTeacher, 0, len(spec.TeacherConfigs)),
	}
	for i := range spec.TeacherConfigs {
		teacher, err := newDistillationTeacher(spec.TeacherConfigs[i], spec.TeacherCheckpoints[i], student, teacherObjective)
		if err != nil {
			out.Close()
			return nil, fmt.Errorf("distillation teacher %d: %w", i, err)
		}
		out.teachers = append(out.teachers, teacher)
	}
	return out, nil
}

func distillationTeacherObjective(student *ArchConfig) string {
	if student == nil {
		return arch.ObjectiveCausal
	}
	switch student.Training.EffectiveObjective() {
	case arch.ObjectiveHybrid:
		if student.Training.EffectiveHybridMixGranularity() == arch.HybridMixGranularityExample {
			return arch.ObjectiveHybridExample
		}
		return student.Training.EffectiveHybridSecondaryObjective()
	case arch.ObjectiveMLM, arch.ObjectiveMNTP:
		return student.Training.EffectiveObjective()
	default:
		return arch.ObjectiveCausal
	}
}

func newDistillationTeacher(configPath, checkpointPath string, student *ArchConfig, teacherObjective string) (*distillationTeacher, error) {
	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("load teacher config %q: %w", configPath, err)
	}
	if cfg.VocabSize != student.VocabSize {
		return nil, fmt.Errorf("teacher vocab_size=%d must match student vocab_size=%d", cfg.VocabSize, student.VocabSize)
	}
	if cfg.SeqLen != student.SeqLen {
		return nil, fmt.Errorf("teacher seq_len=%d must match student seq_len=%d", cfg.SeqLen, student.SeqLen)
	}
	if teacherObjective == arch.ObjectiveMLM || teacherObjective == arch.ObjectiveMNTP || teacherObjective == arch.ObjectiveHybridExample {
		if cfg.Training.MLMMaskTokenID != student.Training.MLMMaskTokenID {
			return nil, fmt.Errorf("teacher mlm_mask_token_id=%d must match student mlm_mask_token_id=%d", cfg.Training.MLMMaskTokenID, student.Training.MLMMaskTokenID)
		}
		if err := validateDistillationTokenizerMatch(student.SourcePath, "", configPath, checkpointPath); err != nil {
			return nil, err
		}
	}
	cfg.Training.BatchTokens = student.Training.BatchTokens
	if err := configureCharFeaturesForConfigPath(cfg, configPath, checkpointPath); err != nil {
		return nil, err
	}
	prog, err := arch.BuildDistillationTeacherIRProgramFromConfig(cfg, teacherObjective)
	if err != nil {
		return nil, fmt.Errorf("build teacher distillation IR: %w", err)
	}
	lossMaskInput := distillationProgramDeclaresInput(prog, "loss_mask")
	causalMaskInput := distillationProgramDeclaresInput(prog, "attention_causal_mask")
	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		return nil, fmt.Errorf("lower teacher distillation IR: %w", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		gpuProg.Destroy()
		return nil, fmt.Errorf("compute teacher weight shapes: %w", err)
	}
	weights, err := loadSafetensorsWeights(checkpointPath, shapes)
	if err != nil {
		gpuProg.Destroy()
		return nil, fmt.Errorf("load teacher checkpoint %q: %w", checkpointPath, err)
	}
	handles, err := uploadWeightHandles(shapes, weights)
	if err != nil {
		gpuProg.Destroy()
		return nil, err
	}
	return &distillationTeacher{
		name:             cfg.Name,
		cfg:              cfg,
		prog:             gpuProg,
		handles:          handles,
		vocabSize:        cfg.VocabSize,
		lossMaskInput:    lossMaskInput,
		causalMaskInput:  causalMaskInput,
		charMaxPerToken:  cfg.CharMaxPerToken,
		bigramVocabSize:  cfg.BigramVocabSize,
		trigramVocabSize: cfg.TrigramVocabSize,
		tokBuf:           make([]int32, student.Training.BatchTokens),
		tgtBuf:           make([]int32, student.Training.BatchTokens),
		lossMaskBuf:      make([]float32, student.Training.BatchTokens),
		causalMaskBuf:    make([]int32, student.Training.BatchTokens/student.SeqLen),
		charBuf:          make([]int32, student.Training.BatchTokens*cfg.CharMaxPerToken),
		bigramBuf:        make([]int32, student.Training.BatchTokens),
		trigramBuf:       make([]int32, student.Training.BatchTokens),
		charFeatures:     append([]int32(nil), cfg.CharFeatureIDs...),
	}, nil
}

func uploadWeightHandles(shapes []WeightShape, weights [][]float32) ([]int64, error) {
	if len(shapes) != len(weights) {
		return nil, fmt.Errorf("shape/weight count mismatch: shapes=%d weights=%d", len(shapes), len(weights))
	}
	handles := make([]int64, len(weights))
	for i, data := range weights {
		h, err := gpu.FromDataShape(data, shapes[i].Shape)
		if err != nil {
			gpu.FreeHandles(handles[:i])
			return nil, fmt.Errorf("upload teacher weight %d (%s): %w", i, shapes[i].Name, err)
		}
		handles[i] = h
	}
	return handles, nil
}

func distillationProgramDeclaresInput(prog *arch.Program, name string) bool {
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

func validateDistillationTokenizerMatch(studentConfigPath, studentCheckpointPath, teacherConfigPath, teacherCheckpointPath string) error {
	studentTok, okStudent := discoverTokenizerJSON(studentConfigPath, studentCheckpointPath)
	teacherTok, okTeacher := discoverTokenizerJSON(teacherConfigPath, teacherCheckpointPath)
	if !okStudent || !okTeacher {
		return nil
	}
	studentHash, err := fileSHA256(studentTok)
	if err != nil {
		return fmt.Errorf("hash student tokenizer %q: %w", studentTok, err)
	}
	teacherHash, err := fileSHA256(teacherTok)
	if err != nil {
		return fmt.Errorf("hash teacher tokenizer %q: %w", teacherTok, err)
	}
	if studentHash != teacherHash {
		return fmt.Errorf("teacher tokenizer %q does not match student tokenizer %q", teacherTok, studentTok)
	}
	return nil
}

func discoverTokenizerJSON(paths ...string) (string, bool) {
	seen := map[string]bool{}
	for _, p := range paths {
		if p == "" {
			continue
		}
		dir := p
		if filepath.Base(p) != "tokenizer.json" {
			dir = filepath.Dir(p)
		}
		candidate := filepath.Join(dir, "tokenizer.json")
		if seen[candidate] {
			continue
		}
		seen[candidate] = true
		if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
			return candidate, true
		}
	}
	return "", false
}

func fileSHA256(path string) ([32]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return [32]byte{}, err
	}
	return sha256.Sum256(data), nil
}

func (e *distillationEnsemble) Close() {
	if e == nil {
		return
	}
	for _, teacher := range e.teachers {
		teacher.Close()
	}
	e.teachers = nil
}

func (t *distillationTeacher) Close() {
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

func (e *distillationEnsemble) TeacherProbs(batch objectiveBatch, batchSize, seqLen int) ([]float32, error) {
	if e == nil {
		return nil, nil
	}
	need := batchSize * seqLen
	out := e.teacherProbBuffer(need)
	for i, teacher := range e.teachers {
		logits, err := teacher.Logits(batch, batchSize, seqLen)
		if err != nil {
			return nil, fmt.Errorf("teacher %d logits: %w", i, err)
		}
		if len(logits) != len(out) {
			return nil, fmt.Errorf("teacher %d logits size=%d want %d", i, len(logits), len(out))
		}
		accumulateTeacherLogits(out, logits, e.strategy, e.vocabSize, e.spec.EffectiveTemperature())
	}
	if len(e.teachers) == 0 {
		return nil, fmt.Errorf("distillation has no teachers")
	}
	finalizeTeacherProbs(out, len(e.teachers), e.vocabSize, e.strategy, e.spec.EffectiveTemperature())
	return out, nil
}

func accumulateTeacherLogits(dst, logits []float32, strategy string, vocabSize int, temperature float64) {
	switch strategy {
	case arch.DistillationMeanLogProbs:
		addLogSoftmaxRowsWithTemperature(dst, logits, vocabSize, temperature)
	default:
		addRows(dst, logits)
	}
}

func finalizeTeacherProbs(values []float32, nTeachers, vocabSize int, strategy string, temperature float64) {
	if nTeachers <= 0 {
		return
	}
	scale := float32(1.0 / float64(nTeachers))
	for i := range values {
		values[i] *= scale
	}
	if strategy != arch.DistillationMeanLogProbs && temperature != 1 {
		invT := float32(1.0 / temperature)
		for i := range values {
			values[i] *= invT
		}
	}
	softmaxRowsInPlace(values, vocabSize)
}

func (e *distillationEnsemble) teacherProbBuffer(tokens int) []float32 {
	if e == nil || tokens <= 0 || e.vocabSize <= 0 {
		return nil
	}
	need := tokens * e.vocabSize
	if len(e.probBuf) < need {
		e.probBuf = make([]float32, need)
	}
	out := e.probBuf[:need]
	clear(out)
	return out
}

func attachDistillationTeacherProbs(e *distillationEnsemble, batch objectiveBatch, batchSize, seqLen int) (objectiveBatch, error) {
	if e == nil {
		return batch, nil
	}
	if batch.lossMask == nil && e.objective != arch.ObjectiveCausal {
		return batch, nil
	}
	if batch.lossMask != nil {
		kdMask := batch.lossMask
		if batch.maskedLossMask != nil {
			kdMask = batch.maskedLossMask
		}
		if !hasPositiveMask(kdMask, batchSize*seqLen) {
			return batch, nil
		}
	}
	probs, err := e.TeacherProbs(batch, batchSize, seqLen)
	if err != nil {
		return objectiveBatch{}, err
	}
	batch.teacherProbs = probs
	return batch, nil
}

func hasPositiveMask(mask []float32, need int) bool {
	for i := 0; i < need && i < len(mask); i++ {
		if mask[i] > 0 {
			return true
		}
	}
	return false
}

func (t *distillationTeacher) Logits(batch objectiveBatch, batchSize, seqLen int) ([]float32, error) {
	inputs, err := t.makeInputs(batch, batchSize, seqLen)
	if err != nil {
		return nil, err
	}
	return gpu.EvalProgramOutput(t.prog, t.handles, inputs, "logits")
}

func (t *distillationTeacher) makeInputs(batch objectiveBatch, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	need := batchSize * seqLen
	xTok := batch.x
	yTok := batch.y
	if len(xTok) < need || len(yTok) < need {
		return nil, fmt.Errorf("input size mismatch: tokens=%d targets=%d need=%d", len(xTok), len(yTok), need)
	}
	if len(t.tokBuf) < need {
		t.tokBuf = make([]int32, need)
		t.tgtBuf = make([]int32, need)
		t.lossMaskBuf = make([]float32, need)
		t.charBuf = make([]int32, need*t.charMaxPerToken)
		t.bigramBuf = make([]int32, need)
		t.trigramBuf = make([]int32, need)
	}
	if len(t.causalMaskBuf) < batchSize {
		t.causalMaskBuf = make([]int32, batchSize)
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
		if len(batch.lossMask) >= need {
			copy(t.lossMaskBuf[:need], batch.lossMask[:need])
		} else {
			clear(t.lossMaskBuf[:need])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.lossMaskBuf[:need],
		})
	}
	if t.causalMaskInput {
		if len(batch.attentionCausal) >= batchSize {
			copy(t.causalMaskBuf[:batchSize], batch.attentionCausal[:batchSize])
		} else {
			clear(t.causalMaskBuf[:batchSize])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "attention_causal_mask", DType: gpu.TensorInt32, Shape: []int{batchSize}, Data: t.causalMaskBuf[:batchSize],
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

func addRows(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func addLogSoftmaxRowsWithTemperature(dst, logits []float32, vocabSize int, temperature float64) {
	if vocabSize <= 0 {
		return
	}
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		temperature = 1.0
	}
	invT := float32(1.0 / temperature)
	for row := 0; row*vocabSize < len(logits); row++ {
		start := row * vocabSize
		end := start + vocabSize
		maxVal := logits[start] * invT
		for _, v := range logits[start+1 : end] {
			scaled := v * invT
			if scaled > maxVal {
				maxVal = scaled
			}
		}
		var sum float64
		for _, v := range logits[start:end] {
			sum += math.Exp(float64(v*invT - maxVal))
		}
		logNorm := maxVal + float32(math.Log(sum))
		for i := start; i < end; i++ {
			dst[i] += logits[i]*invT - logNorm
		}
	}
}

func softmaxRowsInPlace(values []float32, vocabSize int) {
	if vocabSize <= 0 {
		return
	}
	for row := 0; row*vocabSize < len(values); row++ {
		start := row * vocabSize
		end := start + vocabSize
		maxVal := values[start]
		for _, v := range values[start+1 : end] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float64
		for i := start; i < end; i++ {
			ev := math.Exp(float64(values[i] - maxVal))
			values[i] = float32(ev)
			sum += ev
		}
		if sum == 0 {
			fillUniformTeacherProbs(values[start:end], vocabSize)
			continue
		}
		inv := float32(1.0 / sum)
		for i := start; i < end; i++ {
			values[i] *= inv
		}
	}
}

func fillUniformTeacherProbs(values []float32, vocabSize int) {
	if vocabSize <= 0 {
		return
	}
	p := float32(1.0 / float64(vocabSize))
	for i := range values {
		values[i] = p
	}
}
