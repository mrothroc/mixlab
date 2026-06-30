package train

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestDiffusionScoringBatchWiring(t *testing.T) {
	batch, err := diffusionScoringBatch([]int{3, 4, 5, 6, 7}, []int{1, 3, 4}, 6, 99, 2, 4)
	if err != nil {
		t.Fatalf("diffusionScoringBatch: %v", err)
	}
	wantX := []int{
		3, 99, 5, 6, 7, 0,
		3, 4, 5, 99, 7, 0,
		3, 4, 5, 6, 99, 0,
		3, 99, 5, 6, 7, 0,
	}
	if !reflect.DeepEqual(batch.x, wantX) {
		t.Fatalf("x = %v, want %v", batch.x, wantX)
	}
	wantY := []int{
		3, 4, 5, 6, 7, 0,
		3, 4, 5, 6, 7, 0,
		3, 4, 5, 6, 7, 0,
		3, 4, 5, 6, 7, 0,
	}
	if !reflect.DeepEqual(batch.y, wantY) {
		t.Fatalf("y = %v, want %v", batch.y, wantY)
	}
	wantMask := []float32{
		0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0,
	}
	if !reflect.DeepEqual(batch.lossMask, wantMask) {
		t.Fatalf("lossMask = %v, want %v", batch.lossMask, wantMask)
	}
	if !reflect.DeepEqual(batch.diffusionBlockStart, []int32{0, 2, 4, 0}) {
		t.Fatalf("block starts = %v", batch.diffusionBlockStart)
	}
	if !reflect.DeepEqual(batch.diffusionBlockEnd, []int32{2, 4, 5, 2}) {
		t.Fatalf("block ends = %v", batch.diffusionBlockEnd)
	}
}

func TestScoreDiffusionChunkedMatchesSinglePositionOracle(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	tokens := []int{4, 5, 6, 7, 8}
	scoreFrom := 1

	chunkedEval := &fakeDiffusionGenerationEvaluator{logitsForBatch: scoreDiffusionTestLogits(cfg.SeqLen, cfg.VocabSize)}
	chunked, err := scoreDiffusionTokens(cfg, chunkedEval, tokens, scoreFrom, 2)
	if err != nil {
		t.Fatalf("scoreDiffusionTokens chunked: %v", err)
	}
	singleEval := &fakeDiffusionGenerationEvaluator{logitsForBatch: scoreDiffusionTestLogits(cfg.SeqLen, cfg.VocabSize)}
	single, err := scoreDiffusionTokens(cfg, singleEval, tokens, scoreFrom, 1)
	if err != nil {
		t.Fatalf("scoreDiffusionTokens single: %v", err)
	}
	requireFloat64SliceNear(t, chunked, single, 1e-7)
	if len(chunkedEval.batches) != 2 {
		t.Fatalf("chunked calls=%d, want 2", len(chunkedEval.batches))
	}
	if !reflect.DeepEqual(chunkedEval.batchSizes, []int{2, 2}) {
		t.Fatalf("batch sizes=%v, want [2 2]", chunkedEval.batchSizes)
	}
	first := chunkedEval.batches[0]
	if !reflect.DeepEqual(first.diffusionBlockStart, []int32{0, 2}) || !reflect.DeepEqual(first.diffusionBlockEnd, []int32{2, 4}) {
		t.Fatalf("first batch blocks=%v/%v, want [0 2]/[2 4]", first.diffusionBlockStart, first.diffusionBlockEnd)
	}
	second := chunkedEval.batches[1]
	if !reflect.DeepEqual(second.diffusionBlockStart, []int32{2, 4}) || !reflect.DeepEqual(second.diffusionBlockEnd, []int32{4, 5}) {
		t.Fatalf("second batch blocks=%v/%v, want [2 4]/[4 5]", second.diffusionBlockStart, second.diffusionBlockEnd)
	}
	if !reflect.DeepEqual(chunkedEval.readShapes, [][]int{{2 * cfg.SeqLen, cfg.VocabSize}, {2 * cfg.SeqLen, cfg.VocabSize}}) {
		t.Fatalf("read shapes=%v", chunkedEval.readShapes)
	}
}

func TestScoreDiffusionJSONLSkipAndDeterminism(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	input := strings.Join([]string{
		`{"id":"a","tokens":[1,2,3,4],"score_from":1}`,
		`{"id":"b","tokens":[1],"score_from":1}`,
		``,
	}, "\n")

	run := func() string {
		t.Helper()
		eval := &fakeDiffusionGenerationEvaluator{logitsForBatch: scoreDiffusionTestLogits(cfg.SeqLen, cfg.VocabSize)}
		var out bytes.Buffer
		if err := scoreDiffusionJSONL(strings.NewReader(input), &out, cfg, eval, 0, 2); err != nil {
			t.Fatalf("scoreDiffusionJSONL: %v", err)
		}
		return out.String()
	}
	first := run()
	second := run()
	if first != second {
		t.Fatalf("score output not deterministic\nfirst=%s\nsecond=%s", first, second)
	}
	lines := strings.Split(strings.TrimSpace(first), "\n")
	if len(lines) != 2 {
		t.Fatalf("output lines=%d, want 2: %q", len(lines), first)
	}
	var a, b scoreDiffusionOutputRecord
	if err := json.Unmarshal([]byte(lines[0]), &a); err != nil {
		t.Fatalf("unmarshal first output: %v", err)
	}
	if a.ID != "a" || a.ScoreFrom != 1 || a.NTokens != 3 || len(a.PerToken) != 3 {
		t.Fatalf("first output = %+v", a)
	}
	if a.LogprobMean != a.LogprobSum/3 {
		t.Fatalf("mean=%g sum=%g", a.LogprobMean, a.LogprobSum)
	}
	if err := json.Unmarshal([]byte(lines[1]), &b); err != nil {
		t.Fatalf("unmarshal second output: %v", err)
	}
	if b.ID != "b" || b.ScoreFrom != 1 || b.NTokens != 0 || b.LogprobSum != 0 || b.LogprobMean != 0 || len(b.PerToken) != 0 {
		t.Fatalf("second output = %+v", b)
	}
}

func TestScoreDiffusionValidationErrors(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	tests := []struct {
		name string
		rec  scoreDiffusionInputRecord
		want string
	}{
		{name: "missing id", rec: scoreDiffusionInputRecord{Tokens: []int{1}}, want: "id"},
		{name: "empty tokens", rec: scoreDiffusionInputRecord{ID: "x"}, want: "tokens"},
		{name: "too long", rec: scoreDiffusionInputRecord{ID: "x", Tokens: []int{1, 2, 3, 4, 5, 6, 7}}, want: "exceeds seq_len"},
		{name: "bad token", rec: scoreDiffusionInputRecord{ID: "x", Tokens: []int{1, 64}}, want: "out of range"},
		{name: "bad score_from", rec: scoreDiffusionInputRecord{ID: "x", Tokens: []int{1}}, want: "score_from"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := scoreDiffusionRecord(cfg, &fakeDiffusionGenerationEvaluator{}, tt.rec, 2, 1)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("scoreDiffusionRecord error=%v, want %q", err, tt.want)
			}
		})
	}

	nonDiffusion := *cfg
	nonDiffusion.Training.Objective = arch.ObjectiveCausal
	if err := validateScoreDiffusionConfig(&nonDiffusion); err == nil || !strings.Contains(err.Error(), "score-diffusion requires") {
		t.Fatalf("validate non-diffusion error=%v", err)
	}

	var out bytes.Buffer
	err := scoreDiffusionJSONL(strings.NewReader(`{"id":`), &out, cfg, &fakeDiffusionGenerationEvaluator{}, 0, 1)
	if err == nil || !strings.Contains(err.Error(), "line 1") || !strings.Contains(err.Error(), "invalid JSON") {
		t.Fatalf("malformed JSON error=%v", err)
	}
}

func TestRunScoreDiffusionOptionsValidation(t *testing.T) {
	tests := []struct {
		name string
		opts ScoreDiffusionOptions
		want string
	}{
		{name: "unsupported mode", opts: ScoreDiffusionOptions{ScoreMode: "mlm"}, want: "-score-mode"},
		{name: "missing config", opts: ScoreDiffusionOptions{}, want: "-config"},
		{name: "missing weights", opts: ScoreDiffusionOptions{ConfigPath: "cfg.json"}, want: "-safetensors-load"},
		{name: "missing input", opts: ScoreDiffusionOptions{ConfigPath: "cfg.json", SafetensorsLoad: "w.safetensors"}, want: "-score-in"},
		{name: "missing output", opts: ScoreDiffusionOptions{ConfigPath: "cfg.json", SafetensorsLoad: "w.safetensors", ScoreIn: "in.jsonl"}, want: "-score-out"},
		{name: "negative skip", opts: ScoreDiffusionOptions{ConfigPath: "cfg.json", SafetensorsLoad: "w.safetensors", ScoreIn: "in.jsonl", ScoreOut: "out.jsonl", ScoreSkipFirst: -1}, want: "-score-skip-first"},
		{name: "same paths", opts: ScoreDiffusionOptions{ConfigPath: "cfg.json", SafetensorsLoad: "w.safetensors", ScoreIn: "same.jsonl", ScoreOut: "same.jsonl"}, want: "different paths"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := runScoreDiffusionWithOptions(tt.opts)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("runScoreDiffusionWithOptions error=%v, want %q", err, tt.want)
			}
		})
	}
}

func TestScoreDiffusionAcceptsHybridBlockDiffusionConfig(t *testing.T) {
	cfg := testGenerateDiffusionConfig()
	cfg.Training.Objective = arch.ObjectiveHybrid
	cfg.Training.HybridSecondaryObjective = arch.ObjectiveBlockDiffusion
	cfg.Training.HybridCLMFraction = 0.5
	if err := validateScoreDiffusionConfig(cfg); err != nil {
		t.Fatalf("validateScoreDiffusionConfig hybrid: %v", err)
	}
}

func TestEffectiveScorePositionBatch(t *testing.T) {
	if got, err := effectiveScorePositionBatch(16, 32, 7); err != nil || got != 7 {
		t.Fatalf("explicit batch = %d, %v; want 7", got, err)
	}
	if got, err := effectiveScorePositionBatch(1024, 32768, 0); err != nil || got != 2 {
		t.Fatalf("auto batch = %d, %v; want 2", got, err)
	}
	if got, err := effectiveScorePositionBatch(4096, 65536, 0); err != nil || got != 1 {
		t.Fatalf("large auto batch = %d, %v; want 1", got, err)
	}
}

func scoreDiffusionTestLogits(seqLen, vocab int) func(objectiveBatch, int) []float32 {
	return func(batch objectiveBatch, call int) []float32 {
		rows := len(batch.x) / seqLen
		logits := make([]float32, rows*seqLen*vocab)
		for row := 0; row < rows; row++ {
			for pos := 0; pos < seqLen; pos++ {
				base := (row*seqLen + pos) * vocab
				for token := 0; token < vocab; token++ {
					logits[base+token] = float32(row)*0.05 + float32(pos)*0.1 + float32(token)*0.003 + float32(call)*0.02
				}
			}
		}
		return logits
	}
}

func requireFloat64SliceNear(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len got=%d want=%d; got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("value[%d]=%.12g want %.12g diff=%g tol=%g", i, got[i], want[i], math.Abs(got[i]-want[i]), tol)
		}
	}
}
