//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"runtime"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// mlxGPUTrainer wraps the MLX IR trainer for the training loop.
type mlxGPUTrainer struct {
	handle                     gpu.TrainerHandle
	prog                       *gpu.Program
	activeIRProg               *ir.Program
	programCache               map[*ir.Program]*gpu.Program
	handles                    []int64 // GPU weight array handles
	shapes                     []WeightShape
	optimizerSpec              gpu.TrainerOptimizerSpec
	baseLR                     float32
	evalLossOutputName         string
	componentLossOutputs       []string
	tttDiagnosticOutputs       []string
	captureComponentLosses     bool
	vocabSize                  int
	charVocabSize              int
	charMaxPerToken            int
	bigramVocabSize            int
	trigramVocabSize           int
	declaredTargetSize         int
	targetsInput               bool
	targetsInputKnown          bool
	rtdGeneratorBatchSize      int
	rtdGeneratorSeqLen         int
	rtdGeneratorMaskSlots      int
	charInput                  bool
	firstByteMaskInput         bool
	lossMaskInput              bool
	distillLossMaskInput       bool
	wordStructInput            bool
	invarianceInput            bool
	pllMarginInput             bool
	energySpanMaskInput        bool
	attentionCausalInput       bool
	segmentIDsInput            bool
	teacherProbsInput          bool
	data2VecInput              bool
	rtdGeneratorInput          bool
	rtdGeneratorPositionsInput bool
	diffusionBlockStartInput   bool
	diffusionBlockEndInput     bool
	diffusionTimestepInput     bool
	tttInnerLRScaleInput       bool
	classificationLabelsInput  bool
	classificationMaskInput    bool
	classificationPosInput     bool
	rcTokensInput              bool
	rcAlignmentInput           bool
	rcComplementInput          bool
	tttInnerLRScaleCount       int
	dropoutKeyCount            int
	trainingSeed               uint64
	trainingStep               int
	// Pre-allocated input buffers to avoid per-step allocation.
	tokBuf                 []int32
	tgtBuf                 []int32
	lossMaskBuf            []float32
	distillLossMaskBuf     []float32
	wordStructTargetBuf    []int32
	wordStructLossMaskBuf  []float32
	invarianceLossMaskBuf  []float32
	pllMarginLossMaskBuf   []float32
	energySpanMaskBuf      []float32
	attentionCausalBuf     []int32
	segmentIDBuf           []int32
	diffusionBlockStartBuf []int32
	diffusionBlockEndBuf   []int32
	diffusionTimestepBuf   []float32
	teacherProbBuf         []float32
	data2VecTargetBuf      []float32
	data2VecMaskBuf        []float32
	rtdGeneratorTokBuf     []int32
	rtdGeneratorPosBuf     []int32
	rtdGeneratorTgtBuf     []int32
	rtdGeneratorLossBuf    []float32
	charBuf                []int32
	bigramBuf              []int32
	trigramBuf             []int32
	tttInnerLRScaleBuf     []float32
	dropoutKeyBuf          []int32
	classificationLabelBuf []int32
	classificationMaskBuf  []float32
	classificationPosBuf   []int32
	rcTokenBuf             []int32
	rcAlignmentBuf         []int32
	rcComplementIDs        []int32
	charFeatures           []int32
	firstByteValid         []int32
	// MLX registers GPU streams per OS thread; keep trainer setup and steps pinned.
	lockedOSThread bool
}

// initMLXGPUTrainer creates a GPU trainer backed by the MLX IR interpreter.
// If loadedWeights is non-nil, those weights are used instead of random init.
func initMLXGPUTrainer(
	irProg *ir.Program,
	cfg *ArchConfig,
	loadedWeights [][]float32,
	optimizerOverride func(gpu.TrainerOptimizerSpec, []WeightShape) (gpu.TrainerOptimizerSpec, error),
) (*mlxGPUTrainer, error) {
	runtime.LockOSThread()
	releaseOSThread := true
	defer func() {
		if releaseOSThread {
			runtime.UnlockOSThread()
		}
	}()

	if !gpu.Available() {
		return nil, fmt.Errorf("MLX backend unavailable; rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab")
	}

	// Compute weight shapes and initialize data
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return nil, fmt.Errorf("compute weight shapes: %w", err)
	}
	if len(shapes) != irProg.NumWeights {
		return nil, fmt.Errorf("weight count mismatch: shapes=%d program=%d", len(shapes), irProg.NumWeights)
	}

	var weightData [][]float32
	if loadedWeights != nil {
		if len(loadedWeights) != len(shapes) {
			return nil, fmt.Errorf("loaded weight count mismatch: loaded=%d expected=%d", len(loadedWeights), len(shapes))
		}
		// Verify sizes match
		for i, ws := range shapes {
			expected := shapeProduct(ws.Shape)
			if len(loadedWeights[i]) != expected {
				return nil, fmt.Errorf("loaded weight %d (%s) size mismatch: got=%d expected=%d", i, ws.Name, len(loadedWeights[i]), expected)
			}
		}
		weightData = loadedWeights
	} else {
		weightData = initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	}
	if loadedWeights == nil && cfg.MTPUntieEnabled() && cfg.EffectiveMTPUntieStep() <= 0 {
		if err := copyWeightDataByName(weightData, shapes, "head", "embed"); err != nil {
			return nil, fmt.Errorf("initialize untied LM head: %w", err)
		}
	}

	// Upload weights to GPU
	handles := make([]int64, len(weightData))
	for i, data := range weightData {
		rows, cols := 1, len(data)
		if len(shapes[i].Shape) == 2 {
			rows = shapes[i].Shape[0]
			cols = shapes[i].Shape[1]
		}
		h, err := gpu.FromData(data, rows, cols)
		if err != nil {
			gpu.FreeHandles(handles[:i])
			return nil, fmt.Errorf("upload weight %d (%s): %w", i, shapes[i].Name, err)
		}
		handles[i] = h
	}

	// Convert IR program to GPU program
	gpuProg, err := lowerIRToGPU(irProg)
	if err != nil {
		gpu.FreeHandles(handles)
		return nil, fmt.Errorf("lower IR to GPU: %w", err)
	}

	optimizerSpec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		gpuProg.Destroy()
		gpu.FreeHandles(handles)
		return nil, fmt.Errorf("build optimizer spec: %w", err)
	}
	if optimizerOverride != nil {
		optimizerSpec, err = optimizerOverride(optimizerSpec, shapes)
		if err != nil {
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("optimizer override failed: %w", err)
		}
		if err := validateOptimizerSpecCoverage(optimizerSpec, shapes); err != nil {
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("optimizer override returned invalid spec: %w", err)
		}
	}
	computeDType, err := gpuComputeDTypeForTraining(cfg)
	if err != nil {
		gpuProg.Destroy()
		gpu.FreeHandles(handles)
		return nil, err
	}
	optimizerSpec.ComputeDType = computeDType

	trainerHandle, err := gpu.CreateTrainer(gpuProg, handles, optimizerSpec)
	if err != nil {
		gpuProg.Destroy()
		gpu.FreeHandles(handles)
		return nil, fmt.Errorf("create GPU trainer: %w", err)
	}
	if err := gpu.TrainerSetQAT(trainerHandle, qatModeForStep(cfg.Training, 0)); err != nil {
		gpu.TrainerDestroy(trainerHandle)
		gpuProg.Destroy()
		gpu.FreeHandles(handles)
		return nil, fmt.Errorf("set GPU trainer QAT mode: %w", err)
	}
	componentLossOutputs := declaredTrainingStepComponentLossOutputs(irProg)
	tttDiagnosticOutputs := declaredTTTDiagnosticOutputs(irProg)

	batchElems := cfg.Training.BatchTokens
	declaredTargetSize := 0
	targetsInput := false
	declaredBatchSize := 0
	declaredSeqLen := 0
	rtdGeneratorBatchSize := 0
	rtdGeneratorSeqLen := 0
	rtdGeneratorMaskSlots := 0
	charInput := false
	firstByteMaskInput := false
	lossMaskInput := false
	distillLossMaskInput := false
	wordStructInput := false
	invarianceInput := false
	pllMarginInput := false
	energySpanMaskInput := false
	attentionCausalInput := false
	segmentIDsInput := false
	teacherProbsInput := false
	data2VecInput := false
	rtdGeneratorInput := false
	rtdGeneratorPositionsInput := false
	diffusionBlockStartInput := false
	diffusionBlockEndInput := false
	diffusionTimestepInput := false
	tttInnerLRScaleInput := false
	classificationLabelsInput := false
	classificationMaskInput := false
	classificationPosInput := false
	rcTokensInput := false
	rcAlignmentInput := false
	rcComplementInput := false
	tttInnerLRScaleCount := 0
	dropoutKeyCount := 0
	for _, inp := range irProg.Inputs {
		if inp.Name == "tokens" && len(inp.Shape) == 2 {
			declaredBatchSize = inp.Shape[0]
			declaredSeqLen = inp.Shape[1]
			batchElems = declaredBatchSize * declaredSeqLen
		}
		if inp.Name == "targets" && len(inp.Shape) == 1 {
			targetsInput = true
			declaredTargetSize = inp.Shape[0]
		}
		if inp.Name == "ttt_inner_lr_scale" && len(inp.Shape) == 1 {
			tttInnerLRScaleInput = true
			tttInnerLRScaleCount = inp.Shape[0]
		}
		if inp.Name == ir.DropoutKeysInput && len(inp.Shape) == 2 && inp.Shape[1] == 2 {
			dropoutKeyCount = inp.Shape[0]
		}
		if inp.Name == "first_byte_valid" {
			firstByteMaskInput = true
		}
		if inp.Name == "loss_mask" {
			lossMaskInput = true
		}
		if inp.Name == "distill_loss_mask" {
			distillLossMaskInput = true
		}
		if inp.Name == "word_struct_targets" {
			wordStructInput = true
		}
		if inp.Name == "invariance_loss_mask" {
			invarianceInput = true
		}
		if inp.Name == "pll_margin_loss_mask" {
			pllMarginInput = true
		}
		if inp.Name == "energy_span_mask" {
			energySpanMaskInput = true
		}
		if inp.Name == "attention_causal_mask" {
			attentionCausalInput = true
		}
		if inp.Name == "segment_ids" {
			segmentIDsInput = true
		}
		if inp.Name == "teacher_probs" {
			teacherProbsInput = true
		}
		if inp.Name == "data2vec_targets" {
			data2VecInput = true
		}
		if inp.Name == "rtd_generator_tokens" {
			rtdGeneratorInput = true
			if len(inp.Shape) == 2 {
				rtdGeneratorBatchSize = inp.Shape[0]
				rtdGeneratorSeqLen = inp.Shape[1]
			}
		}
		if inp.Name == "rtd_generator_positions" {
			rtdGeneratorPositionsInput = true
			if len(inp.Shape) == 1 {
				rtdGeneratorMaskSlots = inp.Shape[0]
			}
		}
		if inp.Name == "diffusion_block_start" {
			diffusionBlockStartInput = true
		}
		if inp.Name == "diffusion_block_end" {
			diffusionBlockEndInput = true
		}
		if inp.Name == "diffusion_timestep" {
			diffusionTimestepInput = true
		}
		if inp.Name == "char_ids" {
			charInput = true
		}
		if inp.Name == "classification_labels" {
			classificationLabelsInput = true
		}
		if inp.Name == "classification_valid_mask" {
			classificationMaskInput = true
		}
		if inp.Name == "classification_positions" {
			classificationPosInput = true
		}
		if inp.Name == "rc_tokens" {
			rcTokensInput = true
		}
		if inp.Name == "rc_alignment_positions" {
			rcAlignmentInput = true
		}
		if inp.Name == "rc_complement_ids" {
			rcComplementInput = true
		}
	}
	charFeatures := []int32(nil)
	if charInput {
		if cfg.CharVocabSize <= 0 {
			gpu.TrainerDestroy(trainerHandle)
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("program requires char_ids but char_vocab_size is disabled")
		}
		if cfg.CharMaxPerToken <= 0 {
			gpu.TrainerDestroy(trainerHandle)
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("program requires char_ids but char_max_per_token=%d", cfg.CharMaxPerToken)
		}
		want := cfg.VocabSize * cfg.CharMaxPerToken
		if len(cfg.CharFeatureIDs) != want {
			gpu.TrainerDestroy(trainerHandle)
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("char feature lookup size=%d does not match vocab_size*char_max_per_token=%d", len(cfg.CharFeatureIDs), want)
		}
		charFeatures = append([]int32(nil), cfg.CharFeatureIDs...)
	}
	firstByteValid := []int32(nil)
	if firstByteMaskInput {
		firstByteValid = cfg.Training.FirstByteMaskValid
		if len(firstByteValid) == 0 {
			firstByteValid = identityFirstByteMaskValid(cfg.VocabSize)
		}
		if len(firstByteValid) != cfg.VocabSize {
			gpu.TrainerDestroy(trainerHandle)
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("first-byte mask size=%d does not match vocab_size=%d", len(firstByteValid), cfg.VocabSize)
		}
		firstByteValid = append([]int32(nil), firstByteValid...)
	}
	rcComplementIDs := []int32(nil)
	if rcComplementInput {
		if len(cfg.Training.DatasetNucleotideComplement) != cfg.VocabSize {
			gpu.TrainerDestroy(trainerHandle)
			gpuProg.Destroy()
			gpu.FreeHandles(handles)
			return nil, fmt.Errorf("program requires rc_complement_ids but DNA complement lookup has %d entries, want vocab_size=%d", len(cfg.Training.DatasetNucleotideComplement), cfg.VocabSize)
		}
		rcComplementIDs = make([]int32, cfg.VocabSize)
		for i, tokenID := range cfg.Training.DatasetNucleotideComplement {
			rcComplementIDs[i] = int32(tokenID)
		}
	}
	rtdGeneratorPosBufSize := rtdGeneratorMaskSlots
	if rtdGeneratorPosBufSize < 1 {
		rtdGeneratorPosBufSize = 1
	}
	trainer := &mlxGPUTrainer{
		handle:                     trainerHandle,
		prog:                       gpuProg,
		activeIRProg:               irProg,
		programCache:               map[*ir.Program]*gpu.Program{irProg: gpuProg},
		handles:                    handles,
		shapes:                     shapes,
		optimizerSpec:              optimizerSpec,
		baseLR:                     optimizerSpec.DefaultBaseLR,
		evalLossOutputName:         preferredEvalLossOutputName(irProg),
		componentLossOutputs:       componentLossOutputs,
		tttDiagnosticOutputs:       tttDiagnosticOutputs,
		vocabSize:                  cfg.VocabSize,
		charVocabSize:              cfg.CharVocabSize,
		charMaxPerToken:            cfg.CharMaxPerToken,
		bigramVocabSize:            cfg.BigramVocabSize,
		trigramVocabSize:           cfg.TrigramVocabSize,
		declaredTargetSize:         declaredTargetSize,
		targetsInput:               targetsInput,
		targetsInputKnown:          true,
		rtdGeneratorBatchSize:      rtdGeneratorBatchSize,
		rtdGeneratorSeqLen:         rtdGeneratorSeqLen,
		rtdGeneratorMaskSlots:      rtdGeneratorMaskSlots,
		charInput:                  charInput,
		firstByteMaskInput:         firstByteMaskInput,
		lossMaskInput:              lossMaskInput,
		distillLossMaskInput:       distillLossMaskInput,
		wordStructInput:            wordStructInput,
		invarianceInput:            invarianceInput,
		pllMarginInput:             pllMarginInput,
		energySpanMaskInput:        energySpanMaskInput,
		attentionCausalInput:       attentionCausalInput,
		segmentIDsInput:            segmentIDsInput,
		teacherProbsInput:          teacherProbsInput,
		data2VecInput:              data2VecInput,
		rtdGeneratorInput:          rtdGeneratorInput,
		rtdGeneratorPositionsInput: rtdGeneratorPositionsInput,
		diffusionBlockStartInput:   diffusionBlockStartInput,
		diffusionBlockEndInput:     diffusionBlockEndInput,
		diffusionTimestepInput:     diffusionTimestepInput,
		tttInnerLRScaleInput:       tttInnerLRScaleInput,
		classificationLabelsInput:  classificationLabelsInput,
		classificationMaskInput:    classificationMaskInput,
		classificationPosInput:     classificationPosInput,
		rcTokensInput:              rcTokensInput,
		rcAlignmentInput:           rcAlignmentInput,
		rcComplementInput:          rcComplementInput,
		tttInnerLRScaleCount:       tttInnerLRScaleCount,
		dropoutKeyCount:            dropoutKeyCount,
		trainingSeed:               uint64(cfg.Training.Seed),
		tokBuf:                     make([]int32, batchElems),
		tgtBuf:                     make([]int32, batchElems),
		lossMaskBuf:                make([]float32, batchElems),
		distillLossMaskBuf:         make([]float32, batchElems),
		wordStructTargetBuf:        make([]int32, batchElems),
		wordStructLossMaskBuf:      make([]float32, batchElems),
		invarianceLossMaskBuf:      make([]float32, batchElems),
		pllMarginLossMaskBuf:       make([]float32, batchElems),
		energySpanMaskBuf:          make([]float32, batchElems),
		attentionCausalBuf:         make([]int32, batchElems),
		segmentIDBuf:               make([]int32, batchElems),
		diffusionBlockStartBuf:     make([]int32, batchElems),
		diffusionBlockEndBuf:       make([]int32, batchElems),
		diffusionTimestepBuf:       make([]float32, batchElems),
		teacherProbBuf:             make([]float32, batchElems*cfg.VocabSize),
		data2VecTargetBuf:          make([]float32, batchElems*cfg.ModelDim),
		data2VecMaskBuf:            make([]float32, batchElems),
		rtdGeneratorTokBuf:         make([]int32, cfg.Training.BatchTokens),
		rtdGeneratorPosBuf:         make([]int32, rtdGeneratorPosBufSize),
		rtdGeneratorTgtBuf:         make([]int32, cfg.Training.BatchTokens),
		rtdGeneratorLossBuf:        make([]float32, cfg.Training.BatchTokens),
		charBuf:                    make([]int32, batchElems*cfg.CharMaxPerToken),
		bigramBuf:                  make([]int32, batchElems),
		trigramBuf:                 make([]int32, batchElems),
		tttInnerLRScaleBuf:         make([]float32, tttInnerLRScaleCount),
		dropoutKeyBuf:              make([]int32, dropoutKeyCount*2),
		classificationLabelBuf:     make([]int32, declaredBatchSize),
		classificationMaskBuf:      make([]float32, batchElems),
		classificationPosBuf:       make([]int32, declaredBatchSize),
		rcTokenBuf:                 make([]int32, batchElems),
		rcAlignmentBuf:             make([]int32, batchElems),
		rcComplementIDs:            rcComplementIDs,
		charFeatures:               charFeatures,
		firstByteValid:             firstByteValid,
		lockedOSThread:             true,
	}
	releaseOSThread = false
	return trainer, nil
}

func weightIndexByName(shapes []WeightShape, name string) (int, error) {
	for i, shape := range shapes {
		if shape.Name == name {
			return i, nil
		}
	}
	return -1, fmt.Errorf("unknown weight %q", name)
}

func (t *mlxGPUTrainer) shapesModelDim() int {
	for _, shape := range t.shapes {
		if shape.Name == "final_norm" && len(shape.Shape) == 1 {
			return shape.Shape[0]
		}
	}
	if len(t.shapes) > 0 && len(t.shapes[0].Shape) == 2 {
		return t.shapes[0].Shape[1]
	}
	return 0
}

func copyWeightDataByName(weights [][]float32, shapes []WeightShape, dstName, srcName string) error {
	dstIdx, err := weightIndexByName(shapes, dstName)
	if err != nil {
		return err
	}
	srcIdx, err := weightIndexByName(shapes, srcName)
	if err != nil {
		return err
	}
	return copyWeightData(weights[dstIdx], shapes[dstIdx].Shape, weights[srcIdx], shapes[srcIdx].Shape, dstName, srcName)
}

func copyWeightData(dst []float32, dstShape []int, src []float32, srcShape []int, dstName, srcName string) error {
	if len(src) != len(dst) {
		return fmt.Errorf("cannot copy weight %q to %q: size mismatch source=%d destination=%d", srcName, dstName, len(src), len(dst))
	}
	if len(srcShape) == 2 && len(dstShape) == 2 && srcShape[0] == dstShape[1] && srcShape[1] == dstShape[0] {
		rows, cols := srcShape[0], srcShape[1]
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				dst[c*rows+r] = src[r*cols+c]
			}
		}
		return nil
	}
	if len(srcShape) != len(dstShape) {
		return fmt.Errorf("cannot copy weight %q to %q: rank mismatch source=%d destination=%d", srcName, dstName, len(srcShape), len(dstShape))
	}
	for i := range srcShape {
		if srcShape[i] != dstShape[i] {
			return fmt.Errorf("cannot copy weight %q to %q: shape mismatch source=%v destination=%v", srcName, dstName, srcShape, dstShape)
		}
	}
	copy(dst, src)
	return nil
}

// makeInputs creates GPU tensor inputs from token arrays, reusing pre-allocated buffers.
func (t *mlxGPUTrainer) setLRScale(lr float32) {
	lrScale := float32(1.0)
	if t.baseLR > 0 {
		lrScale = lr / t.baseLR
	}
	gpu.TrainerSetLRScale(t.handle, lrScale)
}

// TrainStepGPU runs one training step and returns the loss.
func (t *mlxGPUTrainer) TrainStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) (float32, error) {
	t.setLRScale(lr)
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	loss, err := gpu.TrainerStep(t.handle, inputs)
	if err == nil {
		t.trainingStep++
	}
	return loss, err
}

func (t *mlxGPUTrainer) TrainObjectiveStepGPU(batch objectiveBatch, batchSize, seqLen int, lr float32) (float32, error) {
	t.setLRScale(lr)
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	loss, err := gpu.TrainerStep(t.handle, inputs)
	if err == nil {
		t.trainingStep++
	}
	return loss, err
}

// SubmitStepGPU submits one training step without blocking on loss readback.
func (t *mlxGPUTrainer) SubmitStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) error {
	t.setLRScale(lr)
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return err
	}
	if err := gpu.TrainerSubmitStep(t.handle, inputs); err != nil {
		return err
	}
	t.trainingStep++
	return nil
}

func (t *mlxGPUTrainer) SubmitObjectiveStepGPU(batch objectiveBatch, batchSize, seqLen int, lr float32) error {
	t.setLRScale(lr)
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return err
	}
	if err := gpu.TrainerSubmitStep(t.handle, inputs); err != nil {
		return err
	}
	t.trainingStep++
	return nil
}

// CollectLossGPU blocks until the oldest uncollected submitted step completes.
func (t *mlxGPUTrainer) CollectLossGPU() (float32, error) {
	return gpu.TrainerCollectLoss(t.handle)
}

// FlushGPU waits for any submitted work and discards uncollected losses.
func (t *mlxGPUTrainer) FlushGPU() error {
	return gpu.TrainerFlush(t.handle)
}

func (t *mlxGPUTrainer) SetQATGPU(mode string) error {
	return gpu.TrainerSetQAT(t.handle, mode)
}

func (t *mlxGPUTrainer) SetWeightGPU(name string, data []float32) error {
	for i, shape := range t.shapes {
		if shape.Name == name {
			return gpu.TrainerSetWeight(t.handle, i, data)
		}
	}
	return fmt.Errorf("unknown weight %q", name)
}

func (t *mlxGPUTrainer) CopyWeightGPU(dstName, srcName string) error {
	if err := t.FlushGPU(); err != nil {
		return err
	}
	dstIdx, err := weightIndexByName(t.shapes, dstName)
	if err != nil {
		return err
	}
	srcIdx, err := weightIndexByName(t.shapes, srcName)
	if err != nil {
		return err
	}
	srcSize, err := gpu.TrainerWeightSize(t.handle, srcIdx)
	if err != nil {
		return fmt.Errorf("source weight %q size: %w", srcName, err)
	}
	dstSize, err := gpu.TrainerWeightSize(t.handle, dstIdx)
	if err != nil {
		return fmt.Errorf("destination weight %q size: %w", dstName, err)
	}
	srcData := make([]float32, srcSize)
	if err := gpu.TrainerReadWeight(t.handle, srcIdx, srcData); err != nil {
		return fmt.Errorf("read source weight %q: %w", srcName, err)
	}
	dstData := make([]float32, dstSize)
	if err := copyWeightData(dstData, t.shapes[dstIdx].Shape, srcData, t.shapes[srcIdx].Shape, dstName, srcName); err != nil {
		return err
	}
	if err := gpu.TrainerSetWeight(t.handle, dstIdx, dstData); err != nil {
		return fmt.Errorf("write destination weight %q: %w", dstName, err)
	}
	return nil
}

func (t *mlxGPUTrainer) SetProgramGPU(irProg *ir.Program) error {
	if irProg == nil {
		return fmt.Errorf("nil IR program")
	}
	if irProg.NumWeights != len(t.shapes) {
		return fmt.Errorf("program weight count mismatch: program=%d expected=%d", irProg.NumWeights, len(t.shapes))
	}
	if t.activeIRProg == irProg {
		return nil
	}
	if t.programCache == nil {
		t.programCache = make(map[*ir.Program]*gpu.Program)
		if t.activeIRProg != nil && t.prog != nil {
			t.programCache[t.activeIRProg] = t.prog
		}
	}
	gpuProg := t.programCache[irProg]
	newlyLowered := false
	if gpuProg == nil {
		var err error
		gpuProg, err = lowerIRToGPU(irProg)
		if err != nil {
			return fmt.Errorf("lower IR to GPU: %w", err)
		}
		t.programCache[irProg] = gpuProg
		newlyLowered = true
	}
	if err := gpu.TrainerSetProgram(t.handle, gpuProg); err != nil {
		if newlyLowered {
			delete(t.programCache, irProg)
			gpuProg.Destroy()
		}
		return err
	}
	componentLossOutputs := declaredTrainingStepComponentLossOutputs(irProg)
	tttDiagnosticOutputs := declaredTTTDiagnosticOutputs(irProg)
	if t.captureComponentLosses {
		if err := gpu.TrainerSetStepOutputNames(t.handle, componentLossOutputs); err != nil {
			return fmt.Errorf("configure GPU training step outputs: %w", err)
		}
	}
	t.prog = gpuProg
	t.activeIRProg = irProg
	t.evalLossOutputName = preferredEvalLossOutputName(irProg)
	t.componentLossOutputs = componentLossOutputs
	t.tttDiagnosticOutputs = tttDiagnosticOutputs
	t.charInput = programDeclaresInput(irProg, "char_ids")
	t.firstByteMaskInput = programDeclaresInput(irProg, "first_byte_valid")
	t.lossMaskInput = programDeclaresInput(irProg, "loss_mask")
	t.distillLossMaskInput = programDeclaresInput(irProg, "distill_loss_mask")
	t.wordStructInput = programDeclaresInput(irProg, "word_struct_targets")
	t.invarianceInput = programDeclaresInput(irProg, "invariance_loss_mask")
	t.pllMarginInput = programDeclaresInput(irProg, "pll_margin_loss_mask")
	t.energySpanMaskInput = programDeclaresInput(irProg, "energy_span_mask")
	t.attentionCausalInput = programDeclaresInput(irProg, "attention_causal_mask")
	t.segmentIDsInput = programDeclaresInput(irProg, "segment_ids")
	t.teacherProbsInput = programDeclaresInput(irProg, "teacher_probs")
	t.data2VecInput = programDeclaresInput(irProg, "data2vec_targets")
	t.rtdGeneratorInput = programDeclaresInput(irProg, "rtd_generator_tokens")
	t.rtdGeneratorPositionsInput = programDeclaresInput(irProg, "rtd_generator_positions")
	t.tttInnerLRScaleInput = programDeclaresInput(irProg, "ttt_inner_lr_scale")
	t.targetsInput = programDeclaresInput(irProg, "targets")
	t.targetsInputKnown = true
	t.classificationLabelsInput = programDeclaresInput(irProg, "classification_labels")
	t.classificationMaskInput = programDeclaresInput(irProg, "classification_valid_mask")
	t.classificationPosInput = programDeclaresInput(irProg, "classification_positions")
	t.rcTokensInput = programDeclaresInput(irProg, "rc_tokens")
	t.rcAlignmentInput = programDeclaresInput(irProg, "rc_alignment_positions")
	t.rcComplementInput = programDeclaresInput(irProg, "rc_complement_ids")
	t.tttInnerLRScaleCount = 0
	t.dropoutKeyCount = 0
	t.rtdGeneratorBatchSize = 0
	t.rtdGeneratorSeqLen = 0
	t.rtdGeneratorMaskSlots = 0
	for _, inp := range irProg.Inputs {
		if inp.Name == "ttt_inner_lr_scale" && len(inp.Shape) == 1 {
			t.tttInnerLRScaleCount = inp.Shape[0]
			if len(t.tttInnerLRScaleBuf) < t.tttInnerLRScaleCount {
				t.tttInnerLRScaleBuf = make([]float32, t.tttInnerLRScaleCount)
			}
		}
		if inp.Name == ir.DropoutKeysInput && len(inp.Shape) == 2 && inp.Shape[1] == 2 {
			t.dropoutKeyCount = inp.Shape[0]
			if len(t.dropoutKeyBuf) < t.dropoutKeyCount*2 {
				t.dropoutKeyBuf = make([]int32, t.dropoutKeyCount*2)
			}
		}
		if inp.Name == "rtd_generator_tokens" && len(inp.Shape) == 2 {
			t.rtdGeneratorBatchSize = inp.Shape[0]
			t.rtdGeneratorSeqLen = inp.Shape[1]
		}
		if inp.Name == "rtd_generator_positions" && len(inp.Shape) == 1 {
			t.rtdGeneratorMaskSlots = inp.Shape[0]
		}
	}
	t.diffusionBlockStartInput = programDeclaresInput(irProg, "diffusion_block_start")
	t.diffusionBlockEndInput = programDeclaresInput(irProg, "diffusion_block_end")
	t.diffusionTimestepInput = programDeclaresInput(irProg, "diffusion_timestep")
	if t.charInput && len(t.charFeatures) != t.vocabSize*t.charMaxPerToken {
		return fmt.Errorf("char feature lookup size=%d does not match vocab_size*char_max_per_token=%d", len(t.charFeatures), t.vocabSize*t.charMaxPerToken)
	}
	if t.firstByteMaskInput && len(t.firstByteValid) == 0 {
		t.firstByteValid = identityFirstByteMaskValid(t.vocabSize)
	}
	return nil
}

// EnableComponentLossCapture retains declared scalar component losses with
// each training step for telemetry. It must be configured before submission so
// regular runs retain their historical loss-only capture behavior.
func (t *mlxGPUTrainer) EnableComponentLossCapture() error {
	if t.captureComponentLosses {
		return nil
	}
	if err := gpu.TrainerSetStepOutputNames(t.handle, t.componentLossOutputs); err != nil {
		return fmt.Errorf("configure GPU training step outputs: %w", err)
	}
	t.captureComponentLosses = true
	return nil
}

// EvaluateGPU runs a forward pass without gradients and returns the loss.
func (t *mlxGPUTrainer) EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error) {
	return evaluateTokensViaObjectiveGPU(t, xTok, yTok, batchSize, seqLen)
}

// EvaluateObjectiveGPU runs an objective-batch forward pass without gradients and returns the loss.
func (t *mlxGPUTrainer) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	return t.evaluatePreparedInputs(inputs)
}

func (t *mlxGPUTrainer) EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, outputNames []string) (float32, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	if len(outputNames) == 0 {
		return t.evaluatePreparedInputs(inputs)
	}
	loss, err := gpu.TrainerEvaluateWithOutputs(t.handle, inputs, outputNames)
	if err != nil || t.evalLossOutputName == "" || t.evalLossOutputName == "loss" {
		return loss, err
	}
	out, err := gpu.TrainerReadOutput(t.handle, t.evalLossOutputName, []int{1})
	if err != nil {
		return 0, err
	}
	if len(out) != 1 {
		return 0, fmt.Errorf("eval output %q returned %d values, want 1", t.evalLossOutputName, len(out))
	}
	return out[0], nil
}

func (t *mlxGPUTrainer) CompileStatsGPU() (gpu.TrainerCompileStats, error) {
	return gpu.TrainerCompileStatsSnapshot(t.handle)
}

func (t *mlxGPUTrainer) OptimizerStatsGPU() (gpu.TrainerOptimizerStats, error) {
	return gpu.TrainerOptimizerStatsSnapshot(t.handle)
}

// EvaluateObjectiveTrainingLossGPU evaluates the graph's optimizer loss output
// directly, even when a separate dense eval_loss output is available.
func (t *mlxGPUTrainer) EvaluateObjectiveTrainingLossGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	return gpu.TrainerEvaluate(t.handle, inputs)
}

func (t *mlxGPUTrainer) evaluatePreparedInputs(inputs []gpu.TensorInput) (float32, error) {
	loss, err := gpu.TrainerEvaluate(t.handle, inputs)
	if err != nil || t.evalLossOutputName == "" || t.evalLossOutputName == "loss" {
		return loss, err
	}
	out, err := gpu.TrainerReadOutput(t.handle, t.evalLossOutputName, []int{1})
	if err != nil {
		return 0, err
	}
	if len(out) != 1 {
		return 0, fmt.Errorf("eval output %q returned %d values, want 1", t.evalLossOutputName, len(out))
	}
	return out[0], nil
}

// EvaluatePerTokenGPU runs a forward pass without gradients and returns per-token NLLs.
func (t *mlxGPUTrainer) EvaluatePerTokenGPU(xTok, yTok []int, batchSize, seqLen int) ([]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	inputs, err := t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
	if err != nil {
		return nil, err
	}
	return gpu.TrainerEvaluatePerToken(t.handle, inputs)
}

// EvaluateLoRATTTGPU runs per-batch LoRA TTT without mutating the base trainer weights.
func (t *mlxGPUTrainer) EvaluateLoRATTTGPU(xTok, yTok []int, batchSize, seqLen, tttSteps int, tttLR float32, tttRank int) (float32, error) {
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	return gpu.TrainerEvaluateLoRA(t.handle, inputs, tttRank, tttSteps, tttLR)
}

// CloseTrainer releases GPU resources.
func (t *mlxGPUTrainer) CloseTrainer() {
	if t.handle != 0 {
		_ = gpu.TrainerFlush(t.handle)
		gpu.TrainerDestroy(t.handle)
		t.handle = 0
	}
	if t.prog != nil {
		if len(t.programCache) == 0 {
			t.prog.Destroy()
		}
		t.prog = nil
	}
	if len(t.programCache) > 0 {
		seen := make(map[*gpu.Program]struct{}, len(t.programCache))
		for _, prog := range t.programCache {
			if prog == nil {
				continue
			}
			if _, ok := seen[prog]; ok {
				continue
			}
			seen[prog] = struct{}{}
			prog.Destroy()
		}
		t.programCache = nil
	}
	t.activeIRProg = nil
	if len(t.handles) > 0 {
		gpu.FreeHandles(t.handles)
		t.handles = nil
	}
	if t.lockedOSThread {
		t.lockedOSThread = false
		runtime.UnlockOSThread()
	}
}

// ReadWeights reads all weight tensors back from the GPU trainer.
func (t *mlxGPUTrainer) ReadWeights() ([][]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	nWeights, err := gpu.TrainerNumWeights(t.handle)
	if err != nil {
		return nil, err
	}
	if nWeights != len(t.shapes) {
		return nil, fmt.Errorf("weight count mismatch: trainer=%d expected=%d", nWeights, len(t.shapes))
	}
	weights := make([][]float32, nWeights)
	for i := 0; i < nWeights; i++ {
		size, err := gpu.TrainerWeightSize(t.handle, i)
		if err != nil {
			return nil, fmt.Errorf("weight %d size: %w", i, err)
		}
		data := make([]float32, size)
		if err := gpu.TrainerReadWeight(t.handle, i, data); err != nil {
			return nil, fmt.Errorf("read weight %d: %w", i, err)
		}
		weights[i] = data
	}
	return weights, nil
}

// ReadOutput reads a named output tensor cached by the last trainer step or eval.
func (t *mlxGPUTrainer) ReadOutput(name string, shape []int) ([]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	return gpu.TrainerReadOutput(t.handle, name, shape)
}

// ReadComponentLossesGPU returns the declared scalar component losses from the
// most recently collected training step without flushing a lookahead step.
func (t *mlxGPUTrainer) ReadComponentLossesGPU() (map[string]float64, error) {
	if !t.captureComponentLosses {
		return nil, fmt.Errorf("component loss capture is not enabled")
	}
	if len(t.componentLossOutputs) == 0 {
		return nil, nil
	}
	result := make(map[string]float64, len(t.componentLossOutputs))
	for _, name := range t.componentLossOutputs {
		out, err := gpu.TrainerReadCachedOutput(t.handle, name, []int{1})
		if err != nil {
			return nil, fmt.Errorf("read cached component loss %q: %w", name, err)
		}
		if len(out) != 1 {
			return nil, fmt.Errorf("cached component loss %q returned %d values, want 1", name, len(out))
		}
		result[name] = float64(out[0])
	}
	return result, nil
}
