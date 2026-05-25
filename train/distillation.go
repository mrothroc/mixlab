package train

import (
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

type distillationEnsemble struct {
	spec        *DistillationSpec
	teachers    []*distillationTeacher
	strategy    string
	vocabSize   int
	batchTokens int
	probBuf     []float32
}

type distillationTeacher struct {
	name             string
	cfg              *ArchConfig
	prog             *gpu.Program
	handles          []int64
	vocabSize        int
	bigramVocabSize  int
	trigramVocabSize int
	tokBuf           []int32
	tgtBuf           []int32
	bigramBuf        []int32
	trigramBuf       []int32
}

func newDistillationEnsemble(student *ArchConfig) (*distillationEnsemble, error) {
	if student == nil || student.Training.Distillation == nil {
		return nil, nil
	}
	spec := student.Training.Distillation
	out := &distillationEnsemble{
		spec:        spec,
		strategy:    spec.EffectiveEnsembleStrategy(),
		vocabSize:   student.VocabSize,
		batchTokens: student.Training.BatchTokens,
		teachers:    make([]*distillationTeacher, 0, len(spec.TeacherConfigs)),
	}
	for i := range spec.TeacherConfigs {
		teacher, err := newDistillationTeacher(spec.TeacherConfigs[i], spec.TeacherCheckpoints[i], student)
		if err != nil {
			out.Close()
			return nil, fmt.Errorf("distillation teacher %d: %w", i, err)
		}
		out.teachers = append(out.teachers, teacher)
	}
	return out, nil
}

func newDistillationTeacher(configPath, checkpointPath string, student *ArchConfig) (*distillationTeacher, error) {
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
	cfg.Training.BatchTokens = student.Training.BatchTokens
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("build teacher eval IR: %w", err)
	}
	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		return nil, fmt.Errorf("lower teacher eval IR: %w", err)
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
		bigramVocabSize:  cfg.BigramVocabSize,
		trigramVocabSize: cfg.TrigramVocabSize,
		tokBuf:           make([]int32, student.Training.BatchTokens),
		tgtBuf:           make([]int32, student.Training.BatchTokens),
		bigramBuf:        make([]int32, student.Training.BatchTokens),
		trigramBuf:       make([]int32, student.Training.BatchTokens),
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
		logits, err := teacher.Logits(batch.x, batch.y, batchSize, seqLen)
		if err != nil {
			return nil, fmt.Errorf("teacher %d logits: %w", i, err)
		}
		if len(logits) != len(out) {
			return nil, fmt.Errorf("teacher %d logits size=%d want %d", i, len(logits), len(out))
		}
		switch e.strategy {
		case arch.DistillationMeanLogProbs:
			addLogSoftmaxRows(out, logits, e.vocabSize)
		default:
			addRows(out, logits)
		}
	}
	if len(e.teachers) == 0 {
		return nil, fmt.Errorf("distillation has no teachers")
	}
	scale := float32(1.0 / float64(len(e.teachers)))
	for i := range out {
		out[i] *= scale
	}
	switch e.strategy {
	case arch.DistillationMeanLogProbs:
		softmaxRowsInPlace(out, e.vocabSize)
	default:
		softmaxRowsInPlace(out, e.vocabSize)
	}
	return out, nil
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
	if batch.lossMask != nil {
		return objectiveBatch{}, fmt.Errorf("distillation only supports causal objective batches")
	}
	probs, err := e.TeacherProbs(batch, batchSize, seqLen)
	if err != nil {
		return objectiveBatch{}, err
	}
	batch.teacherProbs = probs
	return batch, nil
}

func (t *distillationTeacher) Logits(xTok, yTok []int, batchSize, seqLen int) ([]float32, error) {
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return nil, err
	}
	return gpu.EvalProgramOutput(t.prog, t.handles, inputs, "logits")
}

func (t *distillationTeacher) makeInputs(xTok, yTok []int, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	need := batchSize * seqLen
	if len(xTok) < need || len(yTok) < need {
		return nil, fmt.Errorf("input size mismatch: tokens=%d targets=%d need=%d", len(xTok), len(yTok), need)
	}
	if len(t.tokBuf) < need {
		t.tokBuf = make([]int32, need)
		t.tgtBuf = make([]int32, need)
		t.bigramBuf = make([]int32, need)
		t.trigramBuf = make([]int32, need)
	}
	for i := 0; i < need; i++ {
		t.tokBuf[i] = int32(xTok[i])
		t.tgtBuf[i] = int32(yTok[i])
	}
	inputs := []gpu.TensorInput{
		{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.tokBuf[:need]},
		{Name: "targets", DType: gpu.TensorInt32, Shape: []int{need}, Data: t.tgtBuf[:need]},
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

func addLogSoftmaxRows(dst, logits []float32, vocabSize int) {
	if vocabSize <= 0 {
		return
	}
	for row := 0; row*vocabSize < len(logits); row++ {
		start := row * vocabSize
		end := start + vocabSize
		maxVal := logits[start]
		for _, v := range logits[start+1 : end] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float64
		for _, v := range logits[start:end] {
			sum += math.Exp(float64(v - maxVal))
		}
		logNorm := maxVal + float32(math.Log(sum))
		for i := start; i < end; i++ {
			dst[i] += logits[i] - logNorm
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
