package train

import (
	"fmt"
	"math"
	"slices"
)

type diffusionSamplerConfig struct {
	blockStart          int
	blockEnd            int
	stepsPerBlock       int
	confidenceThreshold float32
	commitFloor         int
	maskTokenID         int
}

type diffusionSamplerState struct {
	tokens      []int
	committed   []bool
	blockStart  int
	blockEnd    int
	maskTokenID int
}

type diffusionTokenPrediction struct {
	position   int
	token      int
	confidence float32
}

type diffusionCommit struct {
	position   int
	token      int
	confidence float32
	forced     bool
}

type diffusionSamplerStepInput struct {
	step       int
	tokens     []int
	unresolved []int
}

type diffusionSamplerResult struct {
	tokens   []int
	commits  []diffusionCommit
	steps    int
	complete bool
}

type diffusionSamplerPredictFunc func(diffusionSamplerStepInput) ([]diffusionTokenPrediction, error)

func newDiffusionSamplerState(tokens []int, blockStart, blockEnd, maskTokenID int, committed []bool) (diffusionSamplerState, error) {
	if len(tokens) == 0 {
		return diffusionSamplerState{}, fmt.Errorf("empty token buffer")
	}
	if blockStart < 0 || blockEnd <= blockStart || blockEnd > len(tokens) {
		return diffusionSamplerState{}, fmt.Errorf("invalid diffusion block [%d,%d) for %d tokens", blockStart, blockEnd, len(tokens))
	}
	if maskTokenID < 0 {
		return diffusionSamplerState{}, fmt.Errorf("invalid mask_token_id=%d", maskTokenID)
	}

	state := diffusionSamplerState{
		tokens:      append([]int(nil), tokens...),
		committed:   make([]bool, len(tokens)),
		blockStart:  blockStart,
		blockEnd:    blockEnd,
		maskTokenID: maskTokenID,
	}
	if committed != nil {
		if len(committed) != len(tokens) {
			return diffusionSamplerState{}, fmt.Errorf("committed mask length=%d want %d", len(committed), len(tokens))
		}
		copy(state.committed, committed)
	}
	for pos := range state.tokens {
		if pos < blockStart || pos >= blockEnd {
			state.committed[pos] = true
			continue
		}
		if !state.committed[pos] {
			state.tokens[pos] = maskTokenID
		}
	}
	return state, nil
}

func newDiffusionSamplerStateFromConfig(cfg diffusionSamplerConfig, tokens []int, committed []bool) (diffusionSamplerState, error) {
	if cfg.stepsPerBlock < 0 {
		return diffusionSamplerState{}, fmt.Errorf("steps_per_block must be >= 0, got %d", cfg.stepsPerBlock)
	}
	if cfg.commitFloor < 0 {
		return diffusionSamplerState{}, fmt.Errorf("commit_floor must be >= 0, got %d", cfg.commitFloor)
	}
	if !finiteProbability(cfg.confidenceThreshold) {
		return diffusionSamplerState{}, fmt.Errorf("invalid confidence_threshold=%g", cfg.confidenceThreshold)
	}
	return newDiffusionSamplerState(tokens, cfg.blockStart, cfg.blockEnd, cfg.maskTokenID, committed)
}

func (s diffusionSamplerState) clone() diffusionSamplerState {
	return diffusionSamplerState{
		tokens:      append([]int(nil), s.tokens...),
		committed:   append([]bool(nil), s.committed...),
		blockStart:  s.blockStart,
		blockEnd:    s.blockEnd,
		maskTokenID: s.maskTokenID,
	}
}

func (s diffusionSamplerState) tokenSnapshot() []int {
	return append([]int(nil), s.tokens...)
}

func (s diffusionSamplerState) unresolvedPositions() []int {
	positions := make([]int, 0, s.blockEnd-s.blockStart)
	for pos := s.blockStart; pos < s.blockEnd; pos++ {
		if !s.committed[pos] {
			positions = append(positions, pos)
		}
	}
	return positions
}

func (s diffusionSamplerState) complete() bool {
	for pos := s.blockStart; pos < s.blockEnd; pos++ {
		if !s.committed[pos] {
			return false
		}
	}
	return true
}

func (s *diffusionSamplerState) applyDiffusionCommitStep(predictions []diffusionTokenPrediction, threshold float32, commitFloor int) ([]diffusionCommit, error) {
	if s == nil {
		return nil, fmt.Errorf("nil diffusion sampler state")
	}
	commits, err := selectDiffusionCommits(s.committed, s.blockStart, s.blockEnd, predictions, threshold, commitFloor)
	if err != nil {
		return nil, err
	}
	for _, commit := range commits {
		if s.committed[commit.position] {
			continue
		}
		s.tokens[commit.position] = commit.token
		s.committed[commit.position] = true
	}
	return commits, nil
}

func selectDiffusionCommits(committed []bool, blockStart, blockEnd int, predictions []diffusionTokenPrediction, threshold float32, commitFloor int) ([]diffusionCommit, error) {
	if blockStart < 0 || blockEnd < blockStart || blockEnd > len(committed) {
		return nil, fmt.Errorf("invalid diffusion block [%d,%d) for committed length %d", blockStart, blockEnd, len(committed))
	}
	if !finiteProbability(threshold) {
		return nil, fmt.Errorf("invalid confidence_threshold=%g", threshold)
	}
	if commitFloor < 0 {
		return nil, fmt.Errorf("commit_floor must be >= 0, got %d", commitFloor)
	}

	candidates := make([]diffusionCommit, 0, len(predictions))
	seen := make([]bool, len(committed))
	for _, pred := range predictions {
		if pred.position < 0 || pred.position >= len(committed) {
			return nil, fmt.Errorf("prediction position %d out of range [0,%d)", pred.position, len(committed))
		}
		if pred.token < 0 {
			return nil, fmt.Errorf("prediction token %d at position %d is invalid", pred.token, pred.position)
		}
		if !finiteProbability(pred.confidence) {
			return nil, fmt.Errorf("prediction confidence %g at position %d is invalid", pred.confidence, pred.position)
		}
		if pred.position < blockStart || pred.position >= blockEnd || committed[pred.position] {
			continue
		}
		if seen[pred.position] {
			return nil, fmt.Errorf("duplicate prediction for position %d", pred.position)
		}
		seen[pred.position] = true
		candidates = append(candidates, diffusionCommit{
			position:   pred.position,
			token:      pred.token,
			confidence: pred.confidence,
		})
	}

	commits := candidates[:0]
	for _, candidate := range candidates {
		if candidate.confidence >= threshold {
			commits = append(commits, candidate)
		}
	}
	if len(commits) > 0 {
		sortDiffusionCommitsByPosition(commits)
		return append([]diffusionCommit(nil), commits...), nil
	}
	if commitFloor == 0 || len(candidates) == 0 {
		return nil, nil
	}

	slices.SortFunc(candidates, func(a, b diffusionCommit) int {
		switch {
		case a.confidence > b.confidence:
			return -1
		case a.confidence < b.confidence:
			return 1
		case a.position < b.position:
			return -1
		case a.position > b.position:
			return 1
		case a.token < b.token:
			return -1
		case a.token > b.token:
			return 1
		default:
			return 0
		}
	})
	limit := commitFloor
	if limit > len(candidates) {
		limit = len(candidates)
	}
	commits = append([]diffusionCommit(nil), candidates[:limit]...)
	for i := range commits {
		commits[i].forced = true
	}
	sortDiffusionCommitsByPosition(commits)
	return commits, nil
}

func diffusionPredictionsFromProbabilities(probabilities []float32, positions []int, vocab int) ([]diffusionTokenPrediction, error) {
	if vocab <= 0 {
		return nil, fmt.Errorf("invalid vocab=%d", vocab)
	}
	predictions := make([]diffusionTokenPrediction, 0, len(positions))
	for _, pos := range positions {
		row, err := diffusionRow(probabilities, pos, vocab, "probabilities")
		if err != nil {
			return nil, err
		}
		bestToken := 0
		bestProb := row[0]
		for token, prob := range row {
			if !finiteProbability(prob) {
				return nil, fmt.Errorf("probability %g at position %d token %d is invalid", prob, pos, token)
			}
			if prob > bestProb {
				bestToken = token
				bestProb = prob
			}
		}
		predictions = append(predictions, diffusionTokenPrediction{
			position:   pos,
			token:      bestToken,
			confidence: bestProb,
		})
	}
	return predictions, nil
}

func diffusionPredictionsFromLogits(logits []float32, positions []int, vocab int) ([]diffusionTokenPrediction, error) {
	if vocab <= 0 {
		return nil, fmt.Errorf("invalid vocab=%d", vocab)
	}
	predictions := make([]diffusionTokenPrediction, 0, len(positions))
	for _, pos := range positions {
		row, err := diffusionRow(logits, pos, vocab, "logits")
		if err != nil {
			return nil, err
		}
		bestToken := 0
		maxLogit := row[0]
		for token, logit := range row {
			if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 0) {
				return nil, fmt.Errorf("logit %g at position %d token %d is invalid", logit, pos, token)
			}
			if logit > maxLogit {
				bestToken = token
				maxLogit = logit
			}
		}
		sumExp := 0.0
		for _, logit := range row {
			sumExp += math.Exp(float64(logit - maxLogit))
		}
		if sumExp <= 0 || math.IsNaN(sumExp) || math.IsInf(sumExp, 0) {
			return nil, fmt.Errorf("non-finite softmax denominator at position %d", pos)
		}
		confidence := float32(1.0 / sumExp)
		predictions = append(predictions, diffusionTokenPrediction{
			position:   pos,
			token:      bestToken,
			confidence: confidence,
		})
	}
	return predictions, nil
}

func runDiffusionSamplerSteps(cfg diffusionSamplerConfig, tokens []int, committed []bool, predict diffusionSamplerPredictFunc) (diffusionSamplerResult, error) {
	if predict == nil {
		return diffusionSamplerResult{}, fmt.Errorf("nil diffusion sampler predictor")
	}
	state, err := newDiffusionSamplerStateFromConfig(cfg, tokens, committed)
	if err != nil {
		return diffusionSamplerResult{}, err
	}
	return runDiffusionSamplerStateSteps(state, cfg, predict)
}

func runDiffusionSamplerStateSteps(state diffusionSamplerState, cfg diffusionSamplerConfig, predict diffusionSamplerPredictFunc) (diffusionSamplerResult, error) {
	if predict == nil {
		return diffusionSamplerResult{}, fmt.Errorf("nil diffusion sampler predictor")
	}
	if cfg.stepsPerBlock < 0 {
		return diffusionSamplerResult{}, fmt.Errorf("steps_per_block must be >= 0, got %d", cfg.stepsPerBlock)
	}
	if cfg.commitFloor < 0 {
		return diffusionSamplerResult{}, fmt.Errorf("commit_floor must be >= 0, got %d", cfg.commitFloor)
	}
	if !finiteProbability(cfg.confidenceThreshold) {
		return diffusionSamplerResult{}, fmt.Errorf("invalid confidence_threshold=%g", cfg.confidenceThreshold)
	}
	state = state.clone()

	var allCommits []diffusionCommit
	steps := 0
	for step := 0; step < cfg.stepsPerBlock && !state.complete(); step++ {
		unresolved := state.unresolvedPositions()
		predictions, err := predict(diffusionSamplerStepInput{
			step:       step,
			tokens:     state.tokenSnapshot(),
			unresolved: append([]int(nil), unresolved...),
		})
		if err != nil {
			return diffusionSamplerResult{}, err
		}
		commits, err := state.applyDiffusionCommitStep(predictions, cfg.confidenceThreshold, cfg.commitFloor)
		if err != nil {
			return diffusionSamplerResult{}, err
		}
		allCommits = append(allCommits, commits...)
		steps++
	}

	result := diffusionSamplerResult{
		tokens:   state.tokenSnapshot(),
		commits:  allCommits,
		steps:    steps,
		complete: state.complete(),
	}
	if !result.complete {
		return result, fmt.Errorf("diffusion sampler left %d unresolved positions after %d steps", len(state.unresolvedPositions()), cfg.stepsPerBlock)
	}
	return result, nil
}

func diffusionRow(values []float32, position, vocab int, label string) ([]float32, error) {
	if position < 0 {
		return nil, fmt.Errorf("%s position %d is invalid", label, position)
	}
	start := position * vocab
	end := start + vocab
	if start < 0 || end < start || end > len(values) {
		return nil, fmt.Errorf("%s row for position %d out of range: need %d values, got %d", label, position, end, len(values))
	}
	return values[start:end], nil
}

func finiteProbability(v float32) bool {
	return v >= 0 && v <= 1 && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func sortDiffusionCommitsByPosition(commits []diffusionCommit) {
	slices.SortFunc(commits, func(a, b diffusionCommit) int {
		switch {
		case a.position < b.position:
			return -1
		case a.position > b.position:
			return 1
		case a.token < b.token:
			return -1
		case a.token > b.token:
			return 1
		default:
			return 0
		}
	})
}
