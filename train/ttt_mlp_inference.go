package train

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

var errTTTMLPStateClosed = fmt.Errorf("TTT-MLP inference state is closed")

type tttMLPStateHandles struct {
	mlp  int64
	grad int64
	conv int64
}

type tttMLPProgramCacheEntry struct {
	program *gpu.Program
	layouts []arch.TTTMLPStateLayout
}

// TTTMLPInferenceSession owns immutable model weights and cached native
// programs. Each TTTMLPInferenceState created from it owns independent online
// adaptation state.
type TTTMLPInferenceSession struct {
	cfg              *ArchConfig
	weightHandles    []int64
	initialStateData [][]float32
	baseLayouts      []arch.TTTMLPStateLayout
	programs         map[string]*tttMLPProgramCacheEntry
	states           map[*TTTMLPInferenceState]struct{}
	closed           bool
	totalTokens      uint64
	evaluations      uint64
}

// TTTMLPInferenceStats is a cheap host-side snapshot of stateful execution.
type TTTMLPInferenceStats struct {
	Tokens          uint64
	Evaluations     uint64
	ProgramVariants int
	LiveStates      int
}

// TTTMLPInferenceState is a request-owned recurrent cache. It must not be
// shared between sessions or concurrent requests.
type TTTMLPInferenceState struct {
	session *TTTMLPInferenceSession
	handles []tttMLPStateHandles
	offsets []int
	closed  bool
}

// NewTTTMLPInferenceSession loads a causal TTT-MLP checkpoint for stateful
// batch-one prefill and token decode. Pointwise GLU/MLP blocks may be composed
// with TTT-MLP blocks; token mixers without a cache contract are rejected.
func NewTTTMLPInferenceSession(configPath, safetensorsLoad string) (*TTTMLPInferenceSession, error) {
	runtime.LockOSThread()
	fail := func(err error) (*TTTMLPInferenceSession, error) {
		runtime.UnlockOSThread()
		return nil, err
	}
	if configPath == "" {
		return fail(fmt.Errorf("config path is required"))
	}
	if safetensorsLoad == "" {
		return fail(fmt.Errorf("safetensors path is required"))
	}
	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return fail(err)
	}
	if err := configureCharFeaturesForConfigPath(cfg, configPath, safetensorsLoad); err != nil {
		return fail(err)
	}
	baseProgram, layouts, err := arch.BuildTTTMLPStatefulInferenceIRProgram(cfg, 1, make([]int, countTTTMLPBlocks(cfg.Blocks)))
	if err != nil {
		return fail(fmt.Errorf("build TTT-MLP stateful IR: %w", err))
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return fail(fmt.Errorf("compute weight shapes: %w", err))
	}
	weights, err := loadSafetensorsWeights(safetensorsLoad, shapes)
	if err != nil {
		return fail(fmt.Errorf("load safetensors %q: %w", safetensorsLoad, err))
	}
	handles, err := uploadWeightHandles(shapes, weights)
	if err != nil {
		return fail(err)
	}
	gpuProgram, err := gpu.LowerIRProgram(baseProgram)
	if err != nil {
		gpu.FreeHandles(handles)
		return fail(fmt.Errorf("lower TTT-MLP stateful IR: %w", err))
	}
	initial, err := buildTTTMLPInitialStateData(layouts, weights)
	if err != nil {
		gpuProgram.Destroy()
		gpu.FreeHandles(handles)
		return fail(err)
	}
	s := &TTTMLPInferenceSession{
		cfg:              cfg,
		weightHandles:    handles,
		initialStateData: initial,
		baseLayouts:      layouts,
		programs: map[string]*tttMLPProgramCacheEntry{
			tttMLPProgramKey(1, make([]int, len(layouts))): {program: gpuProgram, layouts: layouts},
		},
		states: make(map[*TTTMLPInferenceState]struct{}),
	}
	return s, nil
}

func countTTTMLPBlocks(blocks []BlockSpec) int {
	count := 0
	for _, block := range blocks {
		if strings.EqualFold(strings.TrimSpace(block.Type), "ttt_mlp") {
			count++
		}
	}
	return count
}

func buildTTTMLPInitialStateData(layouts []arch.TTTMLPStateLayout, weights [][]float32) ([][]float32, error) {
	out := make([][]float32, len(layouts))
	for i, layout := range layouts {
		packed := make([]float32, 0, layout.StateSize)
		for _, weightIndex := range layout.InitialWeightIndices {
			if weightIndex < 0 || weightIndex >= len(weights) {
				return nil, fmt.Errorf("TTT block %d initial weight index %d out of range", layout.BlockIndex, weightIndex)
			}
			packed = append(packed, weights[weightIndex]...)
		}
		if len(packed) != layout.StateSize {
			return nil, fmt.Errorf("TTT block %d packed state size=%d, want %d", layout.BlockIndex, len(packed), layout.StateSize)
		}
		out[i] = packed
	}
	return out, nil
}

// Config returns the loaded config. Treat it as read-only.
func (s *TTTMLPInferenceSession) Config() *ArchConfig {
	if s == nil {
		return nil
	}
	return s.cfg
}

// Stats returns cumulative stateful runtime counters.
func (s *TTTMLPInferenceSession) Stats() TTTMLPInferenceStats {
	if s == nil {
		return TTTMLPInferenceStats{}
	}
	return TTTMLPInferenceStats{Tokens: s.totalTokens, Evaluations: s.evaluations, ProgramVariants: len(s.programs), LiveStates: len(s.states)}
}

// NewState allocates a reset request-local cache on the GPU.
func (s *TTTMLPInferenceSession) NewState() (*TTTMLPInferenceState, error) {
	if s == nil || s.closed {
		return nil, errInferenceSessionClosed
	}
	state := &TTTMLPInferenceState{session: s, handles: make([]tttMLPStateHandles, len(s.baseLayouts)), offsets: make([]int, len(s.baseLayouts))}
	if err := state.allocateResetHandles(); err != nil {
		return nil, err
	}
	s.states[state] = struct{}{}
	return state, nil
}

func (state *TTTMLPInferenceState) allocateResetHandles() error {
	for i, layout := range state.session.baseLayouts {
		mlp, err := gpu.FromDataShape(state.session.initialStateData[i], []int{1, layout.StateSize})
		if err != nil {
			state.freeHandles()
			return fmt.Errorf("allocate TTT block %d MLP state: %w", layout.BlockIndex, err)
		}
		grad, err := gpu.FromDataShape(make([]float32, layout.StateSize), []int{1, layout.StateSize})
		if err != nil {
			gpu.FreeHandle(mlp)
			state.freeHandles()
			return fmt.Errorf("allocate TTT block %d gradient state: %w", layout.BlockIndex, err)
		}
		convSize := 2 * 3 * state.session.cfg.ModelDim
		conv, err := gpu.FromDataShape(make([]float32, convSize), []int{1, 2, 3, state.session.cfg.ModelDim})
		if err != nil {
			gpu.FreeHandles([]int64{mlp, grad})
			state.freeHandles()
			return fmt.Errorf("allocate TTT block %d convolution state: %w", layout.BlockIndex, err)
		}
		state.handles[i] = tttMLPStateHandles{mlp: mlp, grad: grad, conv: conv}
	}
	clear(state.offsets)
	return nil
}

// Reset discards all adaptation and convolution history.
func (state *TTTMLPInferenceState) Reset() error {
	if err := state.validate(); err != nil {
		return err
	}
	state.freeHandles()
	return state.allocateResetHandles()
}

// Prefill adapts on tokens in order and returns row-major [len(tokens), vocab]
// logits. Calls may continue a partial chunk left by earlier calls.
func (s *TTTMLPInferenceSession) Prefill(state *TTTMLPInferenceState, tokens []int) ([]float32, error) {
	return s.prefill(state, tokens, true)
}

// PrefillLast adapts on all supplied tokens but retains only the final
// position's logits on the host. This is the memory-bounded streaming path.
func (s *TTTMLPInferenceSession) PrefillLast(state *TTTMLPInferenceState, tokens []int) ([]float32, error) {
	return s.prefill(state, tokens, false)
}

func (s *TTTMLPInferenceSession) prefill(state *TTTMLPInferenceState, tokens []int, retainAll bool) ([]float32, error) {
	if err := s.validateState(state); err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("prefill requires at least one token")
	}
	for i, token := range tokens {
		if token < 0 || token >= s.cfg.VocabSize {
			return nil, fmt.Errorf("token[%d]=%d outside vocab_size=%d", i, token, s.cfg.VocabSize)
		}
	}
	var out []float32
	if retainAll {
		out = make([]float32, 0, len(tokens)*s.cfg.VocabSize)
	}
	for start := 0; start < len(tokens); {
		segmentLen := len(tokens) - start
		for i, layout := range s.baseLayouts {
			remaining := layout.ChunkSize - state.offsets[i]
			if remaining < segmentLen {
				segmentLen = remaining
			}
		}
		logits, err := s.evalSegment(state, tokens[start:start+segmentLen])
		if err != nil {
			return nil, fmt.Errorf("prefill tokens [%d,%d): %w", start, start+segmentLen, err)
		}
		if retainAll {
			out = append(out, logits...)
		} else {
			out = append(out[:0], logits[len(logits)-s.cfg.VocabSize:]...)
		}
		start += segmentLen
	}
	return out, nil
}

// Decode consumes one token and returns its next-token logits.
func (s *TTTMLPInferenceSession) Decode(state *TTTMLPInferenceState, token int) ([]float32, error) {
	return s.Prefill(state, []int{token})
}

func (s *TTTMLPInferenceSession) evalSegment(state *TTTMLPInferenceState, tokens []int) ([]float32, error) {
	entry, err := s.programFor(len(tokens), state.offsets)
	if err != nil {
		return nil, err
	}
	tokenData := make([]int32, len(tokens))
	for i, token := range tokens {
		tokenData[i] = int32(token)
	}
	handleInputs := make([]gpu.HandleInput, 0, 3*len(entry.layouts))
	outputNames := make([]string, 0, 1+3*len(entry.layouts))
	outputNames = append(outputNames, "logits")
	for i, layout := range entry.layouts {
		handles := state.handles[i]
		handleInputs = append(handleInputs,
			gpu.HandleInput{Name: layout.StateInput, Handle: handles.mlp},
			gpu.HandleInput{Name: layout.GradientInput, Handle: handles.grad},
			gpu.HandleInput{Name: layout.ConvInput, Handle: handles.conv})
		outputNames = append(outputNames, layout.StateOutput, layout.GradientOutput, layout.ConvOutput)
	}
	outputs, err := gpu.EvalProgramHandleOutputs(entry.program, s.weightHandles, []gpu.TensorInput{{
		Name: "tokens", DType: gpu.TensorInt32, Shape: []int{1, len(tokens)}, Data: tokenData,
	}}, handleInputs, outputNames)
	if err != nil {
		return nil, err
	}
	allNew := make([]int64, 0, len(outputs))
	for _, handle := range outputs {
		allNew = append(allNew, handle)
	}
	committed := false
	defer func() {
		if !committed {
			gpu.FreeHandles(allNew)
		}
	}()
	logits, err := gpu.ReadHandle(outputs["logits"])
	if err != nil {
		return nil, err
	}
	if len(logits) != len(tokens)*s.cfg.VocabSize {
		return nil, fmt.Errorf("logits size=%d, want %d", len(logits), len(tokens)*s.cfg.VocabSize)
	}
	for i, layout := range entry.layouts {
		next := tttMLPStateHandles{
			mlp: outputs[layout.StateOutput], grad: outputs[layout.GradientOutput], conv: outputs[layout.ConvOutput],
		}
		gpu.FreeHandles([]int64{state.handles[i].mlp, state.handles[i].grad, state.handles[i].conv})
		state.handles[i] = next
		state.offsets[i] = (state.offsets[i] + len(tokens)) % layout.ChunkSize
	}
	gpu.FreeHandle(outputs["logits"])
	s.totalTokens += uint64(len(tokens))
	s.evaluations++
	committed = true
	return logits, nil
}

func (s *TTTMLPInferenceSession) programFor(tokenCount int, offsets []int) (*tttMLPProgramCacheEntry, error) {
	key := tttMLPProgramKey(tokenCount, offsets)
	if entry := s.programs[key]; entry != nil {
		return entry, nil
	}
	irProgram, layouts, err := arch.BuildTTTMLPStatefulInferenceIRProgram(s.cfg, tokenCount, append([]int(nil), offsets...))
	if err != nil {
		return nil, err
	}
	program, err := gpu.LowerIRProgram(irProgram)
	if err != nil {
		return nil, err
	}
	entry := &tttMLPProgramCacheEntry{program: program, layouts: layouts}
	s.programs[key] = entry
	return entry, nil
}

func tttMLPProgramKey(tokenCount int, offsets []int) string {
	var b strings.Builder
	b.WriteString(strconv.Itoa(tokenCount))
	for _, offset := range offsets {
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(offset))
	}
	return b.String()
}

func (state *TTTMLPInferenceState) validate() error {
	if state == nil {
		return fmt.Errorf("nil TTT-MLP inference state")
	}
	if state.closed || state.session == nil {
		return errTTTMLPStateClosed
	}
	if state.session.closed {
		return errInferenceSessionClosed
	}
	return nil
}

func (s *TTTMLPInferenceSession) validateState(state *TTTMLPInferenceState) error {
	if s == nil || s.closed {
		return errInferenceSessionClosed
	}
	if err := state.validate(); err != nil {
		return err
	}
	if state.session != s {
		return fmt.Errorf("TTT-MLP inference state belongs to a different session")
	}
	return nil
}

func (state *TTTMLPInferenceState) freeHandles() {
	for i := range state.handles {
		gpu.FreeHandles([]int64{state.handles[i].mlp, state.handles[i].grad, state.handles[i].conv})
		state.handles[i] = tttMLPStateHandles{}
	}
}

// Close releases a request's recurrent cache. It is idempotent.
func (state *TTTMLPInferenceState) Close() error {
	if state == nil || state.closed {
		return nil
	}
	state.freeHandles()
	if state.session != nil {
		delete(state.session.states, state)
	}
	state.closed = true
	state.session = nil
	return nil
}

// Close releases all states, cached programs, and model weights. It is
// idempotent and must run on the same goroutine as construction.
func (s *TTTMLPInferenceSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	for state := range s.states {
		_ = state.Close()
	}
	for _, entry := range s.programs {
		entry.program.Destroy()
	}
	gpu.FreeHandles(s.weightHandles)
	s.weightHandles = nil
	s.closed = true
	runtime.UnlockOSThread()
	return nil
}
