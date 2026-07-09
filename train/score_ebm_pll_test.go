package train

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestScoreEBMPLLDifferingSpanAndDependentWindow(t *testing.T) {
	cfg := parseTrainMinimalPairPLLConfig(t)
	rec := scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 7, 4}, Corrupt: []int{1, 5, 8, 4},
		CleanSpan: []int{2, 3}, CorruptSpan: []int{2, 3}, Family: "agreement",
	}
	for _, tc := range []struct {
		name        string
		aggregation string
		window      int
		cleanPos    []int
		corruptPos  []int
	}{
		{
			name:        "differing_span",
			aggregation: scoreEBMPLLAggregationDifferingSpan,
			cleanPos:    []int{2},
			corruptPos:  []int{2},
		},
		{
			name:        "dependent_window",
			aggregation: scoreEBMPLLAggregationDependentWin,
			window:      1,
			cleanPos:    []int{1, 2, 3},
			corruptPos:  []int{1, 2, 3},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eval := &fakeEBMEvaluator{logitsForBatch: scoreEBMPLLTestLogits(cfg.SeqLen, cfg.VocabSize)}
			opts := scoreEBMRuntimeOptions{
				scorePositionBatch: 2,
				pllAggregation:     tc.aggregation,
				pllWindow:          tc.window,
				pllSkipTokenIDs:    map[int]bool{1: true},
			}
			out, err := scoreEBMRecordWithOptions(cfg, eval, rec, opts)
			if err != nil {
				t.Fatalf("scoreEBMRecordWithOptions(%s): %v", tc.name, err)
			}
			if !reflect.DeepEqual(eval.requestedOutputs, []string{"head_scorer_logits"}) {
				t.Fatalf("requested outputs=%v", eval.requestedOutputs)
			}
			wantClean := scoreEBMPLLTestOraclePositions(t, cfg.SeqLen, cfg.VocabSize, rec.Clean, tc.cleanPos)
			wantCorrupt := scoreEBMPLLTestOraclePositions(t, cfg.SeqLen, cfg.VocabSize, rec.Corrupt, tc.corruptPos)
			requireFloat64SliceNear(t, []float64{*out.ScoreClean, *out.ScoreCorrupt}, []float64{float64(float32(wantClean)), float64(float32(wantCorrupt))}, 1e-6)
		})
	}
}

func TestScoreEBMPLLAttributionDump(t *testing.T) {
	cfg := parseTrainMinimalPairPLLConfig(t)
	eval := &fakeEBMEvaluator{logitsForBatch: scoreEBMPLLTestLogits(cfg.SeqLen, cfg.VocabSize)}
	var attr bytes.Buffer
	rec := scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 31, 4}, Corrupt: []int{1, 5, 31, 4}, Family: "agreement",
	}
	out, err := scoreEBMRecordWithOptions(cfg, eval, rec, scoreEBMRuntimeOptions{
		scorePositionBatch: 2,
		pllAggregation:     scoreEBMPLLAggregationDifferingSpan,
		pllSkipTokenIDs:    map[int]bool{1: true},
		pllAttributionEnc:  json.NewEncoder(&attr),
	})
	if err != nil {
		t.Fatalf("scoreEBMRecordWithOptions(attribution): %v", err)
	}
	var got scoreEBMPLLAttributionRecord
	if err := json.Unmarshal(bytes.TrimSpace(attr.Bytes()), &got); err != nil {
		t.Fatalf("unmarshal attribution: %v\n%s", err, attr.String())
	}
	if got.ID != rec.ID || got.Family != rec.Family || got.Aggregation != scoreEBMPLLAggregationDifferingSpan {
		t.Fatalf("attribution metadata=%+v", got)
	}
	if !reflect.DeepEqual(got.DifferingSpanClean, []int{1, 2}) || !reflect.DeepEqual(got.DifferingSpanCorrupt, []int{1, 2}) {
		t.Fatalf("attribution spans=%v/%v", got.DifferingSpanClean, got.DifferingSpanCorrupt)
	}
	if got.CleanLogprobs[0] != nil || got.CleanLogprobs[2] != nil || got.CorruptLogprobs[0] != nil || got.CorruptLogprobs[2] != nil {
		t.Fatalf("skipped attribution logprobs clean=%v corrupt=%v", got.CleanLogprobs, got.CorruptLogprobs)
	}
	if got.CleanLogprobs[1] == nil || got.CleanLogprobs[3] == nil || got.CorruptLogprobs[1] == nil || got.CorruptLogprobs[3] == nil {
		t.Fatalf("missing evaluated attribution logprobs clean=%v corrupt=%v", got.CleanLogprobs, got.CorruptLogprobs)
	}
	if out.ScoreClean == nil || out.ScoreCorrupt == nil || got.ScoreClean != *out.ScoreClean || got.ScoreCorrupt != *out.ScoreCorrupt {
		t.Fatalf("attribution scores=%+v output=%+v", got, out)
	}
	if got.Margin != *out.Margin || got.Correct != *out.Correct {
		t.Fatalf("attribution margin/correct=%+v output=%+v", got, out)
	}
	if !reflect.DeepEqual(got.SkippedTokenIDs, []int{1, 31}) {
		t.Fatalf("skipped IDs=%v", got.SkippedTokenIDs)
	}
}

func TestScoreEBMFullSeqPLLSingleObjective(t *testing.T) {
	cfg := parseSinglePLLConfig(t, arch.ObjectiveMLM)
	if err := validateScoreEBMConfig(cfg); err != nil {
		t.Fatalf("validate single MLM score-ebm: %v", err)
	}
	if got, err := effectiveScoreEBMPLLAggregation(cfg, ""); err != nil || got != scoreEBMPLLAggregationFullSeq {
		t.Fatalf("single config aggregation=%q err=%v, want full_seq", got, err)
	}
	opts := scoreEBMRuntimeOptions{
		scorePositionBatch: 2,
		pllAggregation:     scoreEBMPLLAggregationConfig,
		pllSkipTokenIDs:    map[int]bool{1: true},
	}
	eval := &fakeEBMEvaluator{logitsForBatch: scoreEBMPLLTestLogits(cfg.SeqLen, cfg.VocabSize)}
	out, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 31, 4}, Span: []int{1, 2}}, opts)
	if err != nil {
		t.Fatalf("scoreEBMRecordWithOptions(single full_seq): %v", err)
	}
	if !reflect.DeepEqual(eval.requestedOutputs, []string{"logits"}) {
		t.Fatalf("requested outputs=%v", eval.requestedOutputs)
	}
	if eval.batchSize != 2 || len(eval.batch.x) != 2*cfg.SeqLen {
		t.Fatalf("single eval batch size=%d len=%d", eval.batchSize, len(eval.batch.x))
	}
	if !reflect.DeepEqual(eval.batch.x, []int{1, 31, 31, 4, 1, 2, 31, 31}) {
		t.Fatalf("single full-seq rows=%v", eval.batch.x)
	}
	if !reflect.DeepEqual(eval.batch.diffusionBlockStart, []int32{0, 0}) ||
		!reflect.DeepEqual(eval.batch.diffusionBlockEnd, []int32{int32(cfg.SeqLen), int32(cfg.SeqLen)}) {
		t.Fatalf("single full-seq boundaries=%v/%v, want full-sequence rows", eval.batch.diffusionBlockStart, eval.batch.diffusionBlockEnd)
	}
	if out.Score == nil {
		t.Fatalf("missing sequence score: %+v", out)
	}
	want := scoreEBMPLLTestOracle(t, cfg.SeqLen, cfg.VocabSize, []int{1, 2, 31, 4}, map[int]bool{1: true, 31: true})
	requireFloat64SliceNear(t, []float64{*out.Score}, []float64{float64(float32(want))}, 1e-6)
}
