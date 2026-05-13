//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"log"
	"runtime"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// mlxGPUTrainer wraps the MLX IR trainer for the training loop.
type mlxGPUTrainer struct {
	handle             gpu.TrainerHandle
	prog               *gpu.Program
	handles            []int64 // GPU weight array handles
	shapes             []WeightShape
	baseLR             float32
	evalLossOutputName string
	bigramVocabSize    int
	trigramVocabSize   int
	declaredTargetSize int
	// Pre-allocated input buffers to avoid per-step allocation.
	tokBuf     []int32
	tgtBuf     []int32
	bigramBuf  []int32
	trigramBuf []int32
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

	batchElems := cfg.Training.BatchTokens
	declaredTargetSize := 0
	for _, inp := range irProg.Inputs {
		if inp.Name == "targets" && len(inp.Shape) == 1 {
			declaredTargetSize = inp.Shape[0]
			break
		}
	}
	trainer := &mlxGPUTrainer{
		handle:             trainerHandle,
		prog:               gpuProg,
		handles:            handles,
		shapes:             shapes,
		baseLR:             optimizerSpec.DefaultBaseLR,
		evalLossOutputName: preferredEvalLossOutputName(irProg),
		bigramVocabSize:    cfg.BigramVocabSize,
		trigramVocabSize:   cfg.TrigramVocabSize,
		declaredTargetSize: declaredTargetSize,
		tokBuf:             make([]int32, batchElems),
		tgtBuf:             make([]int32, batchElems),
		bigramBuf:          make([]int32, batchElems),
		trigramBuf:         make([]int32, batchElems),
		lockedOSThread:     true,
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
func (t *mlxGPUTrainer) makeInputs(xTok, yTok []int, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	need := batchSize * seqLen
	if len(xTok) < need || len(yTok) < need {
		return nil, fmt.Errorf("input size mismatch: tokens=%d targets=%d need=%d", len(xTok), len(yTok), need)
	}
	// Grow buffers if needed (shouldn't happen with consistent batch size).
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
	targetData, targetShape, err := t.prepareTargets(batchSize, seqLen, need)
	if err != nil {
		return nil, err
	}
	inputs := []gpu.TensorInput{
		{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.tokBuf[:need]},
		{Name: "targets", DType: gpu.TensorInt32, Shape: targetShape, Data: targetData},
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

func (t *mlxGPUTrainer) prepareTargets(batchSize, seqLen, targetSize int) ([]int32, []int, error) {
	if t.declaredTargetSize <= 0 || t.declaredTargetSize == targetSize {
		return t.tgtBuf[:targetSize], []int{targetSize}, nil
	}
	if t.declaredTargetSize < targetSize {
		return nil, nil, fmt.Errorf("declared targets size=%d smaller than batch target size=%d", t.declaredTargetSize, targetSize)
	}
	if batchSize <= 0 {
		return nil, nil, fmt.Errorf("invalid batch size=%d", batchSize)
	}
	extraTargets := t.declaredTargetSize - targetSize
	if extraTargets%batchSize != 0 {
		return nil, nil, fmt.Errorf("declared targets size=%d adds %d extras not divisible by batch size=%d", t.declaredTargetSize, extraTargets, batchSize)
	}
	kPerBatch := extraTargets / batchSize
	if kPerBatch <= 0 {
		return t.tgtBuf[:targetSize], []int{targetSize}, nil
	}
	if seqLen%kPerBatch != 0 {
		return nil, nil, fmt.Errorf("invalid extended targets: seq len=%d not divisible by extra targets per batch=%d", seqLen, kPerBatch)
	}
	stride := seqLen / kPerBatch
	extBuf := make([]int32, t.declaredTargetSize)
	copy(extBuf, t.tgtBuf[:targetSize])
	dst := targetSize
	for b := 0; b < batchSize; b++ {
		base := b * seqLen
		for i := 0; i < kPerBatch; i++ {
			extBuf[dst] = t.tgtBuf[base+i*stride]
			dst++
		}
	}
	return extBuf, []int{t.declaredTargetSize}, nil
}

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
	return gpu.TrainerStep(t.handle, inputs)
}

// SubmitStepGPU submits one training step without blocking on loss readback.
func (t *mlxGPUTrainer) SubmitStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) error {
	t.setLRScale(lr)
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return err
	}
	return gpu.TrainerSubmitStep(t.handle, inputs)
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
	gpuProg, err := lowerIRToGPU(irProg)
	if err != nil {
		return fmt.Errorf("lower IR to GPU: %w", err)
	}
	if err := gpu.TrainerSetProgram(t.handle, gpuProg); err != nil {
		gpuProg.Destroy()
		return err
	}
	if t.prog != nil {
		t.prog.Destroy()
	}
	t.prog = gpuProg
	t.evalLossOutputName = preferredEvalLossOutputName(irProg)
	return nil
}

// EvaluateGPU runs a forward pass without gradients and returns the loss.
func (t *mlxGPUTrainer) EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error) {
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
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
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
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
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
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
		t.prog.Destroy()
		t.prog = nil
	}
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

// lowerIRToGPU converts a mixlab ir.Program to a gpu.Program.
func lowerIRToGPU(prog *ir.Program) (*gpu.Program, error) {
	gpuProg, err := gpu.NewProgram(prog.NumWeights)
	if err != nil {
		return nil, err
	}

	// Declare inputs
	for _, inp := range prog.Inputs {
		dtype := gpu.TensorInt32
		if inp.DType == ir.TensorFloat32 {
			dtype = gpu.TensorFloat32
		}
		if err := gpuProg.DeclareInput(inp.Name, dtype, inp.Shape); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("declare input %q: %w", inp.Name, err)
		}
	}

	// Emit ops
	for _, op := range prog.Ops {
		if err := gpuProg.AddOp(op.Code, op.Inputs, op.Outputs, op.FloatParams, op.IntParams); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("add op code=%d: %w", op.Code, err)
		}
	}

	// Declare outputs
	for _, out := range prog.Outputs {
		dtype := gpu.TensorInt32
		if out.DType == ir.TensorFloat32 {
			dtype = gpu.TensorFloat32
		}
		if err := gpuProg.DeclareOutput(out.Name, dtype, out.Shape); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("declare output %q: %w", out.Name, err)
		}
	}

	return gpuProg, nil
}

func preferredEvalLossOutputName(prog *ir.Program) string {
	if prog != nil {
		for _, out := range prog.Outputs {
			if out.Name == "eval_loss" {
				return "eval_loss"
			}
		}
	}
	return "loss"
}

func buildTrainerOptimizerSpec(cfg *ArchConfig, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
	if cfg == nil {
		return gpu.TrainerOptimizerSpec{}, fmt.Errorf("nil config")
	}
	if cfg.TieEmbeddings && !cfg.MTPUntieEnabled() && cfg.Training.HeadLR != cfg.Training.EmbedLR {
		log.Printf("warning: tie_embeddings=true ignores head_lr; using embed_lr for shared embedding/head weight")
	}
	muonNesterov := true
	if cfg.Training.MuonNesterov != nil {
		muonNesterov = *cfg.Training.MuonNesterov
	}
	matrixOptimizerName := matrixOptimizer(cfg)
	cautiousWeightDecay := cfg.Training.CautiousWeightDecay
	cautiousWeightDecayActivationStep := cfg.Training.EffectiveCautiousWeightDecayActivationStep()
	wmeta := make([]gpu.OptimizerWeightMetadata, len(shapes))
	for i, s := range shapes {
		wmeta[i] = gpu.OptimizerWeightMetadata{
			Name:        s.Name,
			Shape:       s.Shape,
			IsNormScale: s.IsNormScale,
		}
	}
	return gpu.BuildTrainerOptimizerSpec(gpu.TrainerOptimizerConfig{
		Weights: wmeta,
		Embed: gpu.OptimizerSettings{
			Name:                              "adamw",
			LR:                                cfg.Training.EmbedLR,
			Beta1:                             cfg.Training.Beta1,
			Beta2:                             cfg.Training.Beta2,
			Epsilon:                           cfg.Training.Epsilon,
			WeightDecay:                       cfg.Training.EmbedWeightDecay,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
		},
		Head: gpu.OptimizerSettings{
			Name:                              "adamw",
			LR:                                cfg.Training.HeadLR,
			Beta1:                             cfg.Training.Beta1,
			Beta2:                             cfg.Training.Beta2,
			Epsilon:                           cfg.Training.Epsilon,
			WeightDecay:                       cfg.Training.HeadWeightDecay,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
		},
		Scalar: gpu.OptimizerSettings{
			Name:                              "adamw",
			LR:                                cfg.Training.ScalarLR,
			Beta1:                             cfg.Training.Beta1,
			Beta2:                             cfg.Training.Beta2,
			Epsilon:                           cfg.Training.Epsilon,
			WeightDecay:                       cfg.Training.ScalarWeightDecay,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
		},
		Matrix: gpu.OptimizerSettings{
			Name:                              matrixOptimizerName,
			LR:                                cfg.Training.MatrixLR,
			Beta1:                             cfg.Training.MuonMomentum,
			Beta2:                             cfg.Training.Beta2,
			Epsilon:                           cfg.Training.Epsilon,
			WeightDecay:                       cfg.Training.MatrixWeightDecay,
			CautiousWeightDecay:               cautiousWeightDecay,
			CautiousWeightDecayActivationStep: cautiousWeightDecayActivationStep,
			BackendSteps:                      cfg.Training.MuonBackendSteps,
			NewtonSchulzVariant:               cfg.Training.NewtonSchulzVariant,
			Nesterov:                          muonNesterov,
			MuonNormalization:                 matrixMuonNormalization(matrixOptimizerName),
			RowNormalize:                      matrixOptimizerName == "muon_eq_r",
		},
		MaxGradNorm:   cfg.Training.GradClip,
		DefaultBaseLR: float32(cfg.Training.LR),
	})
}

func matrixOptimizer(cfg *ArchConfig) string {
	switch cfg.Training.Optimizer {
	case "adamw":
		return "adamw"
	case "muon_eq_r":
		return "muon_eq_r"
	case "normuon":
		return "normuon"
	default:
		return "muon"
	}
}

func matrixMuonNormalization(name string) gpu.MuonNormalization {
	switch name {
	case "muon_eq_r":
		return gpu.MuonNormalizationRowL2
	case "normuon":
		return gpu.MuonNormalizationNorMuon
	default:
		return gpu.MuonNormalizationNone
	}
}
