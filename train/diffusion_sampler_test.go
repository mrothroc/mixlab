package train

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestCommitThresholdDiffusionSampler(t *testing.T) {
	state, err := newDiffusionSamplerState([]int{10, 99, 99, 13}, 1, 3, 99, nil)
	if err != nil {
		t.Fatalf("newDiffusionSamplerState: %v", err)
	}
	probs := make([]float32, 4*8)
	setDiffusionTestProb(probs, 8, 1, 4, 0.92)
	setDiffusionTestProb(probs, 8, 2, 5, 0.80)
	predictions, err := diffusionPredictionsFromProbabilities(probs, state.unresolvedPositions(), 8)
	if err != nil {
		t.Fatalf("diffusionPredictionsFromProbabilities: %v", err)
	}

	commits, err := state.applyDiffusionCommitStep(predictions, 0.90, 1)
	if err != nil {
		t.Fatalf("applyDiffusionCommitStep: %v", err)
	}

	wantCommits := []diffusionCommit{{position: 1, token: 4, confidence: 0.92}}
	if !reflect.DeepEqual(commits, wantCommits) {
		t.Fatalf("commits = %#v, want %#v", commits, wantCommits)
	}
	if wantTokens := []int{10, 4, 99, 13}; !reflect.DeepEqual(state.tokens, wantTokens) {
		t.Fatalf("tokens = %v, want %v", state.tokens, wantTokens)
	}
	if wantUnresolved := []int{2}; !reflect.DeepEqual(state.unresolvedPositions(), wantUnresolved) {
		t.Fatalf("unresolved = %v, want %v", state.unresolvedPositions(), wantUnresolved)
	}
}

func TestCommitFloorFallbackDiffusionSampler(t *testing.T) {
	state, err := newDiffusionSamplerState([]int{10, 99, 99, 99, 14}, 1, 4, 99, nil)
	if err != nil {
		t.Fatalf("newDiffusionSamplerState: %v", err)
	}
	probs := make([]float32, 5*8)
	setDiffusionTestProb(probs, 8, 1, 1, 0.20)
	setDiffusionTestProb(probs, 8, 2, 2, 0.70)
	setDiffusionTestProb(probs, 8, 3, 3, 0.70)
	predictions, err := diffusionPredictionsFromProbabilities(probs, state.unresolvedPositions(), 8)
	if err != nil {
		t.Fatalf("diffusionPredictionsFromProbabilities: %v", err)
	}

	commits, err := state.applyDiffusionCommitStep(predictions, 0.95, 2)
	if err != nil {
		t.Fatalf("applyDiffusionCommitStep: %v", err)
	}

	wantCommits := []diffusionCommit{
		{position: 2, token: 2, confidence: 0.70, forced: true},
		{position: 3, token: 3, confidence: 0.70, forced: true},
	}
	if !reflect.DeepEqual(commits, wantCommits) {
		t.Fatalf("commits = %#v, want %#v", commits, wantCommits)
	}
	if wantTokens := []int{10, 99, 2, 3, 14}; !reflect.DeepEqual(state.tokens, wantTokens) {
		t.Fatalf("tokens = %v, want %v", state.tokens, wantTokens)
	}
	if wantUnresolved := []int{1}; !reflect.DeepEqual(state.unresolvedPositions(), wantUnresolved) {
		t.Fatalf("unresolved = %v, want %v", state.unresolvedPositions(), wantUnresolved)
	}
}

func TestCommitDiffusionSamplerPreservesCommittedPosition(t *testing.T) {
	committed := []bool{true, true, false, true}
	state, err := newDiffusionSamplerState([]int{1, 7, 99, 4}, 1, 3, 99, committed)
	if err != nil {
		t.Fatalf("newDiffusionSamplerState: %v", err)
	}
	predictions := []diffusionTokenPrediction{
		{position: 1, token: 6, confidence: 0.99},
		{position: 2, token: 8, confidence: 0.99},
	}

	commits, err := state.applyDiffusionCommitStep(predictions, 0.90, 1)
	if err != nil {
		t.Fatalf("applyDiffusionCommitStep: %v", err)
	}

	wantCommits := []diffusionCommit{{position: 2, token: 8, confidence: 0.99}}
	if !reflect.DeepEqual(commits, wantCommits) {
		t.Fatalf("commits = %#v, want %#v", commits, wantCommits)
	}
	if wantTokens := []int{1, 7, 8, 4}; !reflect.DeepEqual(state.tokens, wantTokens) {
		t.Fatalf("tokens = %v, want %v", state.tokens, wantTokens)
	}
}

func TestDiffusionSamplerLogitPredictions(t *testing.T) {
	state, err := newDiffusionSamplerState([]int{3, 99, 5}, 1, 2, 99, nil)
	if err != nil {
		t.Fatalf("newDiffusionSamplerState: %v", err)
	}
	logits := make([]float32, 3*4)
	logits[1*4+0] = -1
	logits[1*4+1] = -1
	logits[1*4+2] = 3
	logits[1*4+3] = -1
	predictions, err := diffusionPredictionsFromLogits(logits, state.unresolvedPositions(), 4)
	if err != nil {
		t.Fatalf("diffusionPredictionsFromLogits: %v", err)
	}
	if len(predictions) != 1 {
		t.Fatalf("predictions = %#v, want one", predictions)
	}
	wantConfidence := float32(1 / (1 + 3*math.Exp(-4)))
	if predictions[0].position != 1 || predictions[0].token != 2 || math.Abs(float64(predictions[0].confidence-wantConfidence)) > 1e-6 {
		t.Fatalf("prediction = %#v, want position=1 token=2 confidence=%g", predictions[0], wantConfidence)
	}

	commits, err := state.applyDiffusionCommitStep(predictions, 0.90, 1)
	if err != nil {
		t.Fatalf("applyDiffusionCommitStep: %v", err)
	}
	if wantCommits := []diffusionCommit{{position: 1, token: 2, confidence: predictions[0].confidence}}; !reflect.DeepEqual(commits, wantCommits) {
		t.Fatalf("commits = %#v, want %#v", commits, wantCommits)
	}
}

func TestDiffusionSamplerSampledLogitPredictionsDeterministic(t *testing.T) {
	logits := make([]float32, 3*4)
	logits[1*4+0] = 0
	logits[1*4+1] = 2
	logits[1*4+2] = 4
	logits[1*4+3] = 6
	first, err := diffusionPredictionsFromLogitsSampled(logits, []int{1}, 4, 1.0, 2, rand.New(rand.NewSource(123)))
	if err != nil {
		t.Fatalf("diffusionPredictionsFromLogitsSampled first: %v", err)
	}
	second, err := diffusionPredictionsFromLogitsSampled(logits, []int{1}, 4, 1.0, 2, rand.New(rand.NewSource(123)))
	if err != nil {
		t.Fatalf("diffusionPredictionsFromLogitsSampled second: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("sampled predictions are not deterministic: %#v vs %#v", first, second)
	}
	if len(first) != 1 {
		t.Fatalf("predictions=%#v, want one", first)
	}
	if first[0].token != 3 && first[0].token != 2 {
		t.Fatalf("top-k sampled token=%d, want one of top-2 tokens 2 or 3", first[0].token)
	}
	if first[0].confidence <= 0 || first[0].confidence > 1 {
		t.Fatalf("sampled confidence=%g, want in (0,1]", first[0].confidence)
	}
}

func TestDiffusionSamplerResolvesBlockWithinSteps(t *testing.T) {
	cfg := diffusionSamplerConfig{
		blockIndex:          2,
		blockStart:          2,
		blockEnd:            6,
		stepsPerBlock:       4,
		confidenceThreshold: 0.90,
		commitFloor:         1,
		maskTokenID:         99,
	}
	start := []int{10, 11, 99, 99, 99, 99}
	result, err := runDiffusionSamplerSteps(cfg, start, nil, func(input diffusionSamplerStepInput) ([]diffusionTokenPrediction, error) {
		if len(input.unresolved) == 0 {
			return nil, nil
		}
		probs := make([]float32, len(input.tokens)*16)
		for i, pos := range input.unresolved {
			confidence := float32(0.20)
			if i == 0 {
				confidence = 0.95
			}
			setDiffusionTestProb(probs, 16, pos, pos+1, confidence)
		}
		return diffusionPredictionsFromProbabilities(probs, input.unresolved, 16)
	})
	if err != nil {
		t.Fatalf("runDiffusionSamplerSteps: %v", err)
	}

	if !result.complete {
		t.Fatalf("complete = false, want true")
	}
	if result.steps != 4 {
		t.Fatalf("steps = %d, want 4", result.steps)
	}
	if wantTokens := []int{10, 11, 3, 4, 5, 6}; !reflect.DeepEqual(result.tokens, wantTokens) {
		t.Fatalf("tokens = %v, want %v", result.tokens, wantTokens)
	}
	if len(result.commits) != 4 {
		t.Fatalf("commits = %#v, want 4 commits", result.commits)
	}
	if len(result.trace) != 4 {
		t.Fatalf("trace len=%d, want 4", len(result.trace))
	}
	firstTrace := result.trace[0]
	if firstTrace.Block != 2 || firstTrace.BlockStart != 2 || firstTrace.BlockEnd != 6 || firstTrace.Step != 0 {
		t.Fatalf("first trace location = %+v", firstTrace)
	}
	if firstTrace.UnresolvedBefore != 4 || firstTrace.Committed != 1 || firstTrace.UnresolvedAfter != 3 || firstTrace.Forced != 0 {
		t.Fatalf("first trace counts = %+v", firstTrace)
	}
	lastTrace := result.trace[len(result.trace)-1]
	if !lastTrace.Complete || lastTrace.UnresolvedAfter != 0 {
		t.Fatalf("last trace = %+v, want complete with no unresolved positions", lastTrace)
	}
	if !reflect.DeepEqual(start, []int{10, 11, 99, 99, 99, 99}) {
		t.Fatalf("input tokens mutated: %v", start)
	}
}

func setDiffusionTestProb(probs []float32, vocab, position, token int, confidence float32) {
	row := probs[position*vocab : (position+1)*vocab]
	row[token] = confidence
}
