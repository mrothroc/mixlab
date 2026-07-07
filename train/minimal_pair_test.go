package train

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestMinimalPairAttachUsesFixedCleanCorruptRows(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl", "pair_batch_fraction": 0.5}`)
	sampler := &minimalPairSampler{records: []minimalPairRecord{
		{ID: "p0", Clean: []int{1, 2, 3}, Corrupt: []int{1, 4, 3}},
	}}
	raw := trainBatch{
		x: []int{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		y: []int{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	got, err := maybeAttachMinimalPairs(sampler, cfg, 0, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen)
	if err != nil {
		t.Fatalf("maybeAttachMinimalPairs: %v", err)
	}
	need := cfg.Training.BatchTokens
	energyOffset := need
	if got := got.x[energyOffset : energyOffset+8]; !intSlicesEqualTrain(got, []int{1, 2, 3, 31, 1, 4, 3, 31}) {
		t.Fatalf("energy first pair x=%v", got)
	}
	if got.lossMask[energyOffset] != 1 || got.lossMask[energyOffset+cfg.SeqLen] != 1 {
		t.Fatalf("active clean/corrupt masks=%g/%g, want 1/1", got.lossMask[energyOffset], got.lossMask[energyOffset+cfg.SeqLen])
	}
	if got.lossMask[energyOffset+2*cfg.SeqLen] != 0 || got.lossMask[energyOffset+3*cfg.SeqLen] != 0 {
		t.Fatalf("inactive pair masks=%g/%g, want 0/0", got.lossMask[energyOffset+2*cfg.SeqLen], got.lossMask[energyOffset+3*cfg.SeqLen])
	}
	rowOffset := cfg.Training.BatchTokens / cfg.SeqLen
	if got.diffusionBlockStart[rowOffset] != 0 || got.diffusionBlockEnd[rowOffset] != int32(cfg.SeqLen) {
		t.Fatalf("energy boundaries=%d/%d, want 0/%d", got.diffusionBlockStart[rowOffset], got.diffusionBlockEnd[rowOffset], cfg.SeqLen)
	}
}

func TestMinimalPairAttachDifferingSpanMask(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl", "energy_aggregation": "differing_span", "pair_batch_fraction": 0.5}`)
	sampler := &minimalPairSampler{records: []minimalPairRecord{
		{ID: "p0", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, Family: "agreement"},
	}}
	raw := trainBatch{
		x: []int{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		y: []int{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	got, err := maybeAttachMinimalPairs(sampler, cfg, 0, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen)
	if err != nil {
		t.Fatalf("maybeAttachMinimalPairs: %v", err)
	}
	energyOffset := cfg.Training.BatchTokens
	if !reflect.DeepEqual(got.energySpanMask[energyOffset:energyOffset+8], []float32{0, 1, 0, 0, 0, 1, 0, 0}) {
		t.Fatalf("active energy span mask=%v", got.energySpanMask[energyOffset:energyOffset+8])
	}
	if !reflect.DeepEqual(got.energySpanMask[energyOffset+8:energyOffset+16], []float32{0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Fatalf("inactive energy span mask=%v", got.energySpanMask[energyOffset+8:energyOffset+16])
	}
}

func TestMinimalPairAttachMLMSpanPLLAppendedRows(t *testing.T) {
	cfg := parseTrainMinimalPairPLLConfig(t)
	sampler := &minimalPairSampler{records: []minimalPairRecord{
		{ID: "p0", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, CleanSpan: []int{1, 2}, CorruptSpan: []int{1, 2}, Family: "agreement"},
	}}
	raw := trainBatch{
		x: []int{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		y: []int{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMultihead)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	if got, want := len(prepared.x), 48; got != want {
		t.Fatalf("prepared len=%d, want %d", got, want)
	}
	if !reflect.DeepEqual(prepared.x[:16], raw.x) {
		t.Fatalf("normal scorer rows changed before pair attach: %v", prepared.x[:16])
	}
	got, err := maybeAttachMinimalPairs(sampler, cfg, 0, prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen)
	if err != nil {
		t.Fatalf("maybeAttachMinimalPairs: %v", err)
	}
	if !reflect.DeepEqual(got.x[:16], raw.x) {
		t.Fatalf("normal scorer rows changed after pair attach: %v", got.x[:16])
	}
	pairOffset := len(cfg.Training.Heads) * cfg.Training.BatchTokens
	if got := got.x[pairOffset : pairOffset+8]; !intSlicesEqualTrain(got, []int{1, 31, 3, 31, 1, 31, 3, 31}) {
		t.Fatalf("active PLL pair x=%v", got)
	}
	if got := got.y[pairOffset : pairOffset+8]; !intSlicesEqualTrain(got, []int{1, 2, 3, 31, 1, 9, 3, 31}) {
		t.Fatalf("active PLL pair y=%v", got)
	}
	wantMask := []float32{0, 1, 0, 0, 0, 1, 0, 0}
	if !reflect.DeepEqual(got.energySpanMask[pairOffset:pairOffset+8], wantMask) {
		t.Fatalf("PLL span mask=%v want %v", got.energySpanMask[pairOffset:pairOffset+8], wantMask)
	}
	if !reflect.DeepEqual(got.lossMask[pairOffset:pairOffset+8], wantMask) {
		t.Fatalf("PLL loss mask=%v want %v", got.lossMask[pairOffset:pairOffset+8], wantMask)
	}
	if !reflect.DeepEqual(got.energySpanMask[pairOffset+8:pairOffset+16], []float32{0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Fatalf("inactive PLL span mask=%v", got.energySpanMask[pairOffset+8:pairOffset+16])
	}
	rowOffset := len(cfg.Training.Heads) * (cfg.Training.BatchTokens / cfg.SeqLen)
	if got.diffusionBlockStart[rowOffset] != 0 || got.diffusionBlockEnd[rowOffset] != int32(cfg.SeqLen) {
		t.Fatalf("PLL boundaries=%d/%d, want 0/%d", got.diffusionBlockStart[rowOffset], got.diffusionBlockEnd[rowOffset], cfg.SeqLen)
	}
}

func TestDecodeMinimalPairJSONLValidation(t *testing.T) {
	records, err := decodeMinimalPairJSONL(strings.NewReader(`{"id":"p0","clean":[1,2],"corrupt":[1,3],"family":"agreement"}`+"\n"), "pairs", 8)
	if err != nil {
		t.Fatalf("decodeMinimalPairJSONL: %v", err)
	}
	if len(records) != 1 || records[0].Family != "agreement" {
		t.Fatalf("records=%+v", records)
	}
	_, err = decodeMinimalPairJSONL(strings.NewReader(`{"id":"bad","clean":[1],"corrupt":[9]}`+"\n"), "pairs", 8)
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("error=%v, want out of range", err)
	}
}

func TestMinimalPairDifferingSpanDerivationAndValidation(t *testing.T) {
	tests := []struct {
		name        string
		clean       []int
		corrupt     []int
		cleanSpan   []int
		corruptSpan []int
	}{
		{"substitution", []int{1, 2, 3}, []int{1, 9, 3}, []int{1, 2}, []int{1, 2}},
		{"insertion", []int{1, 2, 3}, []int{1, 2, 9, 3}, []int{2, 3}, []int{2, 3}},
		{"deletion", []int{1, 2, 9, 3}, []int{1, 2, 3}, []int{2, 3}, []int{2, 3}},
		{"multi-token", []int{1, 2, 3, 4}, []int{1, 8, 9, 4}, []int{1, 3}, []int{1, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := minimalPairRecord{ID: tt.name, Clean: tt.clean, Corrupt: tt.corrupt}
			if err := ensureMinimalPairRecordSpans(&rec); err != nil {
				t.Fatalf("ensureMinimalPairRecordSpans: %v", err)
			}
			if !reflect.DeepEqual(rec.CleanSpan, tt.cleanSpan) || !reflect.DeepEqual(rec.CorruptSpan, tt.corruptSpan) {
				t.Fatalf("spans=%v/%v want %v/%v", rec.CleanSpan, rec.CorruptSpan, tt.cleanSpan, tt.corruptSpan)
			}
		})
	}
	rec := minimalPairRecord{ID: "same", Clean: []int{1, 2}, Corrupt: []int{1, 2}}
	if err := ensureMinimalPairRecordSpans(&rec); err == nil || !strings.Contains(err.Error(), "identical") {
		t.Fatalf("identical pair error=%v", err)
	}
	_, err := decodeMinimalPairJSONLWithOptions(strings.NewReader(`{"id":"bad","clean":[1],"corrupt":[2],"clean_span":[1,1]}`+"\n"), "pairs", minimalPairDecodeOptions{
		VocabSize:         8,
		EnergyAggregation: arch.MinimalPairEnergySpan,
	})
	if err == nil || !strings.Contains(err.Error(), "clean_span") {
		t.Fatalf("invalid span error=%v", err)
	}
}

func TestMinimalPairBinaryRoundTrip(t *testing.T) {
	records := []minimalPairRecord{
		{ID: "p0", Clean: []int{1, 2, 3}, Corrupt: []int{1, 4, 3}, CleanSpan: []int{1, 2}, CorruptSpan: []int{1, 2}, Family: "agreement"},
		{ID: "p1", Clean: []int{5, 6}, Corrupt: []int{6, 5}, CleanSpan: []int{0, 2}, CorruptSpan: []int{0, 2}, Family: "word_order"},
	}
	var buf bytes.Buffer
	if err := writeMinimalPairBinary(&buf, records, 32, 4); err != nil {
		t.Fatalf("writeMinimalPairBinary: %v", err)
	}
	got, err := decodeMinimalPairBinary(bytes.NewReader(buf.Bytes()), "pairs.mpair", minimalPairDecodeOptions{
		VocabSize:     32,
		MaxLen:        4,
		RequireFamily: true,
	})
	if err != nil {
		t.Fatalf("decodeMinimalPairBinary: %v", err)
	}
	if len(got) != len(records) || got[1].ID != "p1" || got[1].Family != "word_order" || !intSlicesEqualTrain(got[1].Corrupt, []int{6, 5}) {
		t.Fatalf("round-trip records=%+v", got)
	}
	if !reflect.DeepEqual(got[0].CleanSpan, []int{1, 2}) || !reflect.DeepEqual(got[0].CorruptSpan, []int{1, 2}) {
		t.Fatalf("round-trip spans=%+v", got[0])
	}
	if _, err := decodeMinimalPairBinary(bytes.NewReader(buf.Bytes()), "pairs.mpair", minimalPairDecodeOptions{VocabSize: 31}); err == nil || !strings.Contains(err.Error(), "vocab_size") {
		t.Fatalf("vocab mismatch error=%v, want vocab_size", err)
	}
}

func TestMinimalPairBinaryV1DerivesSpans(t *testing.T) {
	var buf bytes.Buffer
	if err := writeMinimalPairBinaryV1ForTest(&buf, []minimalPairRecord{
		{ID: "p0", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, Family: "agreement"},
	}, 32, 4); err != nil {
		t.Fatalf("write v1 fixture: %v", err)
	}
	got, err := decodeMinimalPairBinary(bytes.NewReader(buf.Bytes()), "pairs-v1.mpair", minimalPairDecodeOptions{
		VocabSize:         32,
		MaxLen:            4,
		RequireFamily:     true,
		EnergyAggregation: arch.MinimalPairEnergySpan,
	})
	if err != nil {
		t.Fatalf("decode v1 binary: %v", err)
	}
	if !reflect.DeepEqual(got[0].CleanSpan, []int{1, 2}) || !reflect.DeepEqual(got[0].CorruptSpan, []int{1, 2}) {
		t.Fatalf("derived spans from v1=%+v", got[0])
	}
}

func TestPreparePairsValidateAndCompile(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "pairs.jsonl")
	out := filepath.Join(dir, "pairs.mpair")
	if err := os.WriteFile(in, []byte(`{"id":"p0","clean":[1,2],"corrupt":[1,3],"family":"agreement"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write pair jsonl: %v", err)
	}
	if err := runPreparePairsWithOptions(PreparePairsOptions{
		PairIn:    in,
		PairOut:   out,
		VocabSize: 32,
		MaxLen:    4,
	}); err != nil {
		t.Fatalf("runPreparePairsWithOptions: %v", err)
	}
	records, err := loadMinimalPairs(out, arch.MinimalPairSourceBinary, minimalPairDecodeOptions{VocabSize: 32, RequireFamily: true})
	if err != nil {
		t.Fatalf("load compiled pair shard: %v", err)
	}
	if len(records) != 1 || records[0].ID != "p0" || records[0].Family != "agreement" {
		t.Fatalf("compiled records=%+v", records)
	}

	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"source": "bin", "path": "`+strings.ReplaceAll(out, `\`, `\\`)+`"}`)
	sampler, err := newMinimalPairSampler(cfg)
	if err != nil {
		t.Fatalf("newMinimalPairSampler(bin): %v", err)
	}
	if len(sampler.records) != 1 || sampler.records[0].ID != "p0" {
		t.Fatalf("sampler records=%+v", sampler.records)
	}
}

func TestPreparePairsRejectsLengthAndMissingFamily(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "pairs.jsonl")
	if err := os.WriteFile(in, []byte(`{"id":"p0","clean":[1,2],"corrupt":[1,3]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write pair jsonl: %v", err)
	}
	err := runPreparePairsWithOptions(PreparePairsOptions{PairIn: in, VocabSize: 8, MaxLen: 4})
	if err == nil || !strings.Contains(err.Error(), "family") {
		t.Fatalf("missing family error=%v, want family", err)
	}
	if err := os.WriteFile(in, []byte(`{"id":"p0","clean":[1,2],"corrupt":[1,3],"family":"agreement"}`+"\n"), 0o644); err != nil {
		t.Fatalf("rewrite pair jsonl: %v", err)
	}
	err = runPreparePairsWithOptions(PreparePairsOptions{PairIn: in, VocabSize: 8, MaxLen: 1})
	if err == nil || !strings.Contains(err.Error(), "exceeds max length") {
		t.Fatalf("length error=%v, want exceeds max length", err)
	}
	err = runPreparePairsWithOptions(PreparePairsOptions{PairIn: in, PairOut: in, VocabSize: 8, MaxLen: 4})
	if err == nil || !strings.Contains(err.Error(), "must be different") {
		t.Fatalf("same path error=%v, want must be different", err)
	}
}

func TestScoreEBMRecordSequenceAndPair(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl"}`)
	eval := &fakeEBMEvaluator{output: []float32{0.25, 1.25}}
	pairOut, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2}, Corrupt: []int{1, 3}, Family: "agreement",
	}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig})
	if err != nil {
		t.Fatalf("scoreEBMRecord(pair): %v", err)
	}
	if pairOut.EnergyClean == nil || pairOut.EnergyCorrupt == nil || *pairOut.EnergyClean != 0.25 || *pairOut.EnergyCorrupt != 1.25 {
		t.Fatalf("pair energies=%+v", pairOut)
	}
	if pairOut.Correct == nil || !*pairOut.Correct || pairOut.Margin == nil || *pairOut.Margin != 1.0 {
		t.Fatalf("pair correctness/margin=%+v", pairOut)
	}
	energyOffset := 2 * cfg.SeqLen
	if eval.batch.x[energyOffset] != 1 || eval.batch.x[energyOffset+cfg.SeqLen] != 1 {
		t.Fatalf("energy scoring rows not populated: %v", eval.batch.x)
	}
	if !reflect.DeepEqual(eval.requestedOutputs, []string{"head_energy_logits"}) {
		t.Fatalf("requested outputs=%v, want head_energy_logits", eval.requestedOutputs)
	}
	if eval.batchSize != 4 || len(eval.batch.x) != eval.batchSize*cfg.SeqLen {
		t.Fatalf("energy eval batchSize=%d len(x)=%d, want full row count 4", eval.batchSize, len(eval.batch.x))
	}

	eval.output = []float32{0.5, 0.0}
	eval.requestedOutputs = nil
	seqOut, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig})
	if err != nil {
		t.Fatalf("scoreEBMRecord(seq): %v", err)
	}
	if seqOut.Energy == nil || *seqOut.Energy != 0.5 || seqOut.NTokens != 3 {
		t.Fatalf("seq output=%+v", seqOut)
	}
}

func TestScoreEBMDifferingSpanAndTokenEnergy(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl", "energy_aggregation": "differing_span"}`)
	eval := &fakeEBMEvaluator{outputs: map[string][]float32{
		"head_energy_logits":       {0.25, 1.25},
		"head_energy_token_energy": {10, 11, 12, 13, 20, 21, 22, 23},
	}}
	out, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, Family: "agreement",
	}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig, emitTokenEnergy: true})
	if err != nil {
		t.Fatalf("scoreEBMRecord(pair): %v", err)
	}
	if !reflect.DeepEqual(eval.requestedOutputs, []string{"head_energy_logits", "head_energy_token_energy"}) {
		t.Fatalf("requested outputs=%v", eval.requestedOutputs)
	}
	energyOffset := 2 * cfg.SeqLen
	if !reflect.DeepEqual(eval.batch.energySpanMask[energyOffset:energyOffset+8], []float32{0, 1, 0, 0, 0, 1, 0, 0}) {
		t.Fatalf("score energy span mask=%v", eval.batch.energySpanMask[energyOffset:energyOffset+8])
	}
	if out.EnergyClean == nil || *out.EnergyClean != 0.25 || out.EnergyCorrupt == nil || *out.EnergyCorrupt != 1.25 {
		t.Fatalf("energies=%+v", out)
	}
	if !reflect.DeepEqual(out.CleanTokenEnergy, []float64{10, 11, 12}) || !reflect.DeepEqual(out.CorruptTokenEnergy, []float64{20, 21, 22}) {
		t.Fatalf("token energies clean=%v corrupt=%v", out.CleanTokenEnergy, out.CorruptTokenEnergy)
	}

	eval.outputs = map[string][]float32{
		"head_energy_logits":       {0.5, 0},
		"head_energy_token_energy": {30, 31, 32, 33, 0, 0, 0, 0},
	}
	eval.requestedOutputs = nil
	seqOut, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}, Span: []int{2, 3}}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig, emitTokenEnergy: true})
	if err != nil {
		t.Fatalf("scoreEBMRecord(seq): %v", err)
	}
	if !reflect.DeepEqual(eval.batch.energySpanMask[energyOffset:energyOffset+4], []float32{0, 0, 1, 0}) {
		t.Fatalf("sequence span mask=%v", eval.batch.energySpanMask[energyOffset:energyOffset+4])
	}
	if seqOut.Energy == nil || *seqOut.Energy != 0.5 || !reflect.DeepEqual(seqOut.TokenEnergy, []float64{30, 31, 32}) {
		t.Fatalf("sequence output=%+v", seqOut)
	}
}

func TestScoreEBMMLMSpanPLL(t *testing.T) {
	cfg := parseTrainMinimalPairPLLConfig(t)
	eval := &fakeEBMEvaluator{outputs: map[string][]float32{
		"head_scorer_minimal_pair_scores": {2.5, 1.0},
	}}
	out, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, Family: "agreement",
	}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig})
	if err != nil {
		t.Fatalf("scoreEBMRecord(pair): %v", err)
	}
	if !reflect.DeepEqual(eval.requestedOutputs, []string{"head_scorer_minimal_pair_scores"}) {
		t.Fatalf("requested outputs=%v", eval.requestedOutputs)
	}
	if eval.batchSize != 6 || len(eval.batch.x) != eval.batchSize*cfg.SeqLen {
		t.Fatalf("PLL eval batchSize=%d len(x)=%d, want normal plus appended pair rows", eval.batchSize, len(eval.batch.x))
	}
	pairOffset := len(cfg.Training.Heads) * 2 * cfg.SeqLen
	if !reflect.DeepEqual(eval.batch.x[pairOffset:pairOffset+8], []int{1, 31, 3, 31, 1, 31, 3, 31}) {
		t.Fatalf("PLL scoring rows=%v", eval.batch.x[pairOffset:pairOffset+8])
	}
	if !reflect.DeepEqual(eval.batch.y[pairOffset:pairOffset+8], []int{1, 2, 3, 31, 1, 9, 3, 31}) {
		t.Fatalf("PLL scoring targets=%v", eval.batch.y[pairOffset:pairOffset+8])
	}
	if !reflect.DeepEqual(eval.batch.energySpanMask[pairOffset:pairOffset+8], []float32{0, 1, 0, 0, 0, 1, 0, 0}) {
		t.Fatalf("PLL scoring span mask=%v", eval.batch.energySpanMask[pairOffset:pairOffset+8])
	}
	if out.ScoreClean == nil || *out.ScoreClean != 2.5 || out.ScoreCorrupt == nil || *out.ScoreCorrupt != 1.0 {
		t.Fatalf("scores=%+v", out)
	}
	if out.Margin == nil || *out.Margin != 1.5 || out.Correct == nil || !*out.Correct {
		t.Fatalf("margin/correct=%+v", out)
	}

	eval.outputs = map[string][]float32{"head_scorer_minimal_pair_scores": {3.0, 0}}
	eval.requestedOutputs = nil
	seqOut, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}, Span: []int{1, 3}}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig})
	if err != nil {
		t.Fatalf("scoreEBMRecord(seq): %v", err)
	}
	if seqOut.Score == nil || *seqOut.Score != 3.0 || seqOut.Energy != nil {
		t.Fatalf("seq PLL output=%+v", seqOut)
	}
	if _, err := scoreEBMRecordWithOptions(cfg, eval, scoreEBMInputRecord{ID: "seq2", Tokens: []int{1, 2, 3}}, scoreEBMRuntimeOptions{scoreBatch: 2, pllAggregation: scoreEBMPLLAggregationConfig, emitTokenEnergy: true}); err == nil || !strings.Contains(err.Error(), "native energy") {
		t.Fatalf("token energy error=%v, want native energy rejection", err)
	}
}

func TestScoreEBMFullSeqPLLMultihead(t *testing.T) {
	cfg := parseTrainMinimalPairPLLConfig(t)
	opts := scoreEBMRuntimeOptions{
		scorePositionBatch: 2,
		pllAggregation:     scoreEBMPLLAggregationFullSeq,
		pllSkipTokenIDs:    map[int]bool{1: true},
	}
	eval := &fakeEBMEvaluator{logitsForBatch: scoreEBMPLLTestLogits(cfg.SeqLen, cfg.VocabSize)}
	rec := scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 31, 4}, Corrupt: []int{1, 5, 31, 4},
		CleanSpan: []int{1, 2}, CorruptSpan: []int{1, 2}, Family: "agreement",
	}
	out, err := scoreEBMRecordWithOptions(cfg, eval, rec, opts)
	if err != nil {
		t.Fatalf("scoreEBMRecordWithOptions(full_seq pair): %v", err)
	}
	if !reflect.DeepEqual(eval.requestedOutputs, []string{"head_scorer_logits"}) || eval.calls != 2 {
		t.Fatalf("requested outputs=%v", eval.requestedOutputs)
	}
	if !reflect.DeepEqual(eval.batchSizes, []int{6, 6}) {
		t.Fatalf("eval batch sizes=%v, want [6 6]", eval.batchSizes)
	}
	if len(eval.batches) != 2 {
		t.Fatalf("eval batches=%d, want 2", len(eval.batches))
	}
	first := eval.batches[0]
	if first.batchSizeOverride != 6 || len(first.x) != 6*cfg.SeqLen {
		t.Fatalf("first batch override=%d len=%d, want six rows", first.batchSizeOverride, len(first.x))
	}
	if !reflect.DeepEqual(first.x[:8], []int{1, 31, 31, 4, 1, 2, 31, 31}) {
		t.Fatalf("full-seq score-head rows=%v", first.x[:8])
	}
	if !reflect.DeepEqual(first.lossMask[:8], []float32{0, 1, 0, 0, 0, 0, 0, 1}) {
		t.Fatalf("full-seq loss mask=%v", first.lossMask[:8])
	}
	if !reflect.DeepEqual(eval.readShapes, [][]int{{2 * cfg.SeqLen, cfg.VocabSize}, {2 * cfg.SeqLen, cfg.VocabSize}}) {
		t.Fatalf("read shapes=%v", eval.readShapes)
	}
	skip := map[int]bool{1: true, 31: true}
	wantClean := scoreEBMPLLTestOracle(t, cfg.SeqLen, cfg.VocabSize, rec.Clean, skip)
	wantCorrupt := scoreEBMPLLTestOracle(t, cfg.SeqLen, cfg.VocabSize, rec.Corrupt, skip)
	if out.ScoreClean == nil || out.ScoreCorrupt == nil {
		t.Fatalf("missing scores: %+v", out)
	}
	requireFloat64SliceNear(t, []float64{*out.ScoreClean, *out.ScoreCorrupt}, []float64{float64(float32(wantClean)), float64(float32(wantCorrupt))}, 1e-6)
	if out.Margin == nil || out.Correct == nil || *out.Margin != *out.ScoreClean-*out.ScoreCorrupt || *out.Correct != (*out.ScoreClean > *out.ScoreCorrupt) {
		t.Fatalf("margin/correct=%+v", out)
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

func TestScoreEBMFullSeqPLLSingleParallelGroupBuilds(t *testing.T) {
	body := `{
		"name": "single_pll_parallel_group",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"block_scales": true,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional", "parallel_group": 3},
			{"type": "hgrn2", "heads": 2, "residual_scale_init": 0.0},
			{"type": "geglu"}
		],
		"training": {
			"objective": "mntp",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31
		}
	}`
	cfg, err := ParseArchConfig([]byte(body), "single_pll_parallel_group")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if err := validateScoreEBMConfig(cfg); err != nil {
		t.Fatalf("validateScoreEBMConfig: %v", err)
	}
	prog, err := buildScoreEBMIRProgram(cfg, scoreEBMModeSinglePLL, scoreEBMPLLAggregationFullSeq)
	if err != nil {
		t.Fatalf("buildScoreEBMIRProgram: %v", err)
	}
	if got := countProgramOps(prog, arch.OpHGRN2Scan); got != 1 {
		t.Fatalf("HGRN2Scan ops=%d want 1", got)
	}
	if !trainProgramDeclaresOutput(prog, "logits") {
		t.Fatal("single full-seq PLL scoring program missing logits output")
	}
}

func TestScoreEBMSingleFullSeqPLLSharedDebertaSuppressesDenseEvalLoss(t *testing.T) {
	cfg := parseSinglePLLSharedDebertaConfig(t)
	scoreCfg := *cfg
	scoreCfg.Training.BatchTokens = scoreCfg.SeqLen
	prog, err := buildScoreEBMIRProgram(&scoreCfg, scoreEBMModeSinglePLL, scoreEBMPLLAggregationFullSeq)
	if err != nil {
		t.Fatalf("buildScoreEBMIRProgram: %v", err)
	}
	if got := trainPreferredEvalLossOutputName(prog); got != "loss" {
		t.Fatalf("preferred eval loss=%q, want loss", got)
	}
	if trainProgramDeclaresOutput(prog, "eval_loss") {
		t.Fatal("single full-seq PLL scoring program should not declare dense eval_loss")
	}
	if trainProgramProducesOutput(prog, "eval_loss") {
		t.Fatal("single full-seq PLL scoring program should not produce dense eval_loss")
	}
	for _, name := range []string{"diffusion_block_start", "diffusion_block_end"} {
		if !trainProgramDeclaresInput(prog, name) {
			t.Fatalf("single full-seq PLL scoring program missing input %q", name)
		}
	}
	if !trainProgramDeclaresOutput(prog, "logits") {
		t.Fatal("single full-seq PLL scoring program missing logits output")
	}
	if got := trainCountOps(prog, arch.OpDebertaRelativeBias); got != 1 {
		t.Fatalf("DeBERTa relative-bias ops=%d, want 1", got)
	}
	if got := trainCountOps(prog, arch.OpBlockDiffusionMask); got != 1 {
		t.Fatalf("block-diffusion mask ops=%d, want 1", got)
	}
	if got := trainCountOps(prog, arch.OpMaskedCrossEntropy); got != 1 {
		t.Fatalf("masked CE ops=%d, want 1", got)
	}
	if got := trainCountOps(prog, arch.OpCrossEntropy); got != 1 {
		t.Fatalf("dense CE op should remain prunable with renamed output; got %d", got)
	}
}

func TestScoreEBMSingleFullSeqPLLExplicitCausalKeepsCausalGraph(t *testing.T) {
	cfg := parseSinglePLLConfig(t, arch.ObjectiveMNTP)
	cfg.Blocks[0].AttentionMask = arch.AttentionMaskCausal
	scoreCfg := *cfg
	scoreCfg.Training.BatchTokens = scoreCfg.SeqLen
	prog, err := buildScoreEBMIRProgram(&scoreCfg, scoreEBMModeSinglePLL, scoreEBMPLLAggregationFullSeq)
	if err != nil {
		t.Fatalf("buildScoreEBMIRProgram: %v", err)
	}
	if trainProgramDeclaresInput(prog, "diffusion_block_start") || trainProgramDeclaresInput(prog, "diffusion_block_end") {
		t.Fatal("explicit causal single full-seq PLL graph should not declare block-diffusion boundaries")
	}
	if got := trainCountOps(prog, arch.OpBlockDiffusionMask); got != 0 {
		t.Fatalf("block-diffusion mask ops=%d, want 0", got)
	}
	if got := trainCountOps(prog, arch.OpCausalMask); got != 1 {
		t.Fatalf("causal mask ops=%d, want 1", got)
	}
}

func TestScoreEBMFullSeqPLLValidation(t *testing.T) {
	energyCfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl"}`)
	if _, err := effectiveScoreEBMPLLAggregation(energyCfg, scoreEBMPLLAggregationFullSeq); err == nil || !strings.Contains(err.Error(), "native energy") {
		t.Fatalf("energy full_seq error=%v, want native energy rejection", err)
	}
	singleCfg := parseSinglePLLConfig(t, arch.ObjectiveMNTP)
	if _, err := effectiveScoreEBMPLLAggregation(singleCfg, scoreEBMPLLAggregationDifferingSpan); err == nil || !strings.Contains(err.Error(), "mlm_span_pll") {
		t.Fatalf("single differing_span error=%v, want mlm_span_pll rejection", err)
	}
	if _, err := effectiveScoreEBMPLLAggregation(singleCfg, "bad"); err == nil || !strings.Contains(err.Error(), "score-pll-aggregation") {
		t.Fatalf("invalid aggregation error=%v", err)
	}
	if got, err := parseScoreEBMSkipTokenIDs("1, 2", 8); err != nil || !got[1] || !got[2] || len(got) != 2 {
		t.Fatalf("parse skip ids=%v err=%v", got, err)
	}
	if _, err := parseScoreEBMSkipTokenIDs("8", 8); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("bad skip id error=%v", err)
	}
}

func TestScoreEBMJSONLSummaryIncludesFamilies(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl"}`)
	eval := &fakeEBMEvaluator{output: []float32{0.1, 0.9}}
	var out bytes.Buffer
	err := scoreEBMJSONL(strings.NewReader(`{"id":"p0","clean":[1],"corrupt":[2],"family":"agreement"}`+"\n"), &out, cfg, eval, 2, false)
	if err != nil {
		t.Fatalf("scoreEBMJSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines=%d output=%s", len(lines), out.String())
	}
	var summary map[string]interface{}
	if err := json.Unmarshal([]byte(lines[1]), &summary); err != nil {
		t.Fatalf("unmarshal summary: %v", err)
	}
	if summary["id"] != "__summary__" {
		t.Fatalf("summary=%v", summary)
	}
	if _, ok := summary["families"].(map[string]interface{})["agreement"]; !ok {
		t.Fatalf("summary missing family: %v", summary)
	}
}

type fakeEBMEvaluator struct {
	batch            objectiveBatch
	batches          []objectiveBatch
	batchSize        int
	batchSizes       []int
	seqLen           int
	output           []float32
	outputs          map[string][]float32
	logitsForBatch   func(objectiveBatch, int) []float32
	calls            int
	readShapes       [][]int
	requestedOutputs []string
}

func (f *fakeEBMEvaluator) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	f.batch = batch
	f.batches = append(f.batches, batch)
	f.batchSize = batchSize
	f.batchSizes = append(f.batchSizes, batchSize)
	f.seqLen = seqLen
	f.calls++
	if len(batch.x) != batchSize*seqLen {
		return 0, fmt.Errorf("batch x length=%d does not match batchSize=%d seqLen=%d", len(batch.x), batchSize, seqLen)
	}
	return 0, nil
}

func (f *fakeEBMEvaluator) EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, outputNames []string) (float32, error) {
	f.batch = batch
	f.batches = append(f.batches, batch)
	f.batchSize = batchSize
	f.batchSizes = append(f.batchSizes, batchSize)
	f.seqLen = seqLen
	f.calls++
	if len(batch.x) != batchSize*seqLen {
		return 0, fmt.Errorf("batch x length=%d does not match batchSize=%d seqLen=%d", len(batch.x), batchSize, seqLen)
	}
	f.requestedOutputs = append([]string(nil), outputNames...)
	return 0, nil
}

func (f *fakeEBMEvaluator) ReadOutput(name string, shape []int) ([]float32, error) {
	f.readShapes = append(f.readShapes, append([]int(nil), shape...))
	if f.outputs != nil {
		if out, ok := f.outputs[name]; ok {
			return append([]float32(nil), out...), nil
		}
	}
	if f.logitsForBatch != nil {
		batch := f.batch
		if len(shape) >= 1 && shape[0] >= 0 && shape[0] < len(batch.x) {
			batch.x = batch.x[:shape[0]]
		}
		return f.logitsForBatch(batch, f.calls), nil
	}
	return append([]float32(nil), f.output...), nil
}

func parseTrainMinimalPairConfig(t *testing.T, minimalPair string) *ArchConfig {
	t.Helper()
	body := `{
		"name": "minimal_pair_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 16,
			"mlm_mask_token_id": 31,
			"export_head": "scorer",
			` + minimalPair + `,
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
				{"name": "energy", "objective": "energy", "loss_weight": 0.3}
			]
		}
	}`
	cfg, err := ParseArchConfig([]byte(body), "minimal_pair_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func parseTrainMinimalPairPLLConfig(t *testing.T) *ArchConfig {
	t.Helper()
	body := `{
		"name": "minimal_pair_pll_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 16,
			"mlm_mask_token_id": 31,
			"mlm_mask_prob": 0,
			"export_head": "scorer",
			"minimal_pair": {
				"path": "pairs.jsonl",
				"energy_aggregation": "differing_span",
				"score_source": "mlm_span_pll",
				"pair_batch_fraction": 0.5
			},
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
				{"name": "aux", "objective": "causal", "loss_weight": 0.3}
			]
		}
	}`
	cfg, err := ParseArchConfig([]byte(body), "minimal_pair_pll_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func parseSinglePLLConfig(t *testing.T, objective string) *ArchConfig {
	t.Helper()
	body := `{
		"name": "single_pll_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}],
		"training": {
			"objective": "` + objective + `",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31
		}
	}`
	cfg, err := ParseArchConfig([]byte(body), "single_pll_test")
	if err != nil {
		t.Fatalf("ParseArchConfig single PLL: %v", err)
	}
	return cfg
}

func parseSinglePLLSharedDebertaConfig(t *testing.T) *ArchConfig {
	t.Helper()
	body := `{
		"name": "single_pll_shared_deberta_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [{
			"type": "plain",
			"heads": 2,
			"attention_mask": "bidirectional",
			"relative_attention": "deberta_p2c_c2p",
			"relative_attention_window": 3,
			"relative_attention_parameterization": "shared_qk_reuse",
			"ffn_activation": "geglu",
			"attn_bias": true,
			"attn_value_gate": true
		}],
		"training": {
			"objective": "mntp",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31,
			"mlm_head": "bert",
			"z_loss": 0.1
		}
	}`
	cfg, err := ParseArchConfig([]byte(body), "single_pll_shared_deberta_test")
	if err != nil {
		t.Fatalf("ParseArchConfig single PLL shared DeBERTa: %v", err)
	}
	return cfg
}

func scoreEBMPLLTestLogits(seqLen, vocab int) func(objectiveBatch, int) []float32 {
	return func(batch objectiveBatch, _ int) []float32 {
		rows := len(batch.x) / seqLen
		logits := make([]float32, rows*seqLen*vocab)
		for row := 0; row < rows; row++ {
			for pos := 0; pos < seqLen; pos++ {
				base := (row*seqLen + pos) * vocab
				for token := 0; token < vocab; token++ {
					logits[base+token] = float32(pos)*0.1 + float32(token)*0.003
				}
			}
		}
		return logits
	}
}

func scoreEBMPLLTestOracle(t *testing.T, seqLen, vocab int, tokens []int, skip map[int]bool) float64 {
	t.Helper()
	logitsFn := scoreEBMPLLTestLogits(seqLen, vocab)
	positions := scoreEBMFullSeqPLLPositions(tokens, skip)
	var total float64
	for _, pos := range positions {
		batch := objectiveBatch{x: make([]int, seqLen)}
		logits := logitsFn(batch, 0)
		start := pos * vocab
		lp, err := targetLogProbFromLogits(logits[start:start+vocab], tokens[pos])
		if err != nil {
			t.Fatalf("targetLogProbFromLogits: %v", err)
		}
		total += lp
	}
	return total
}

func trainProgramDeclaresInput(prog *arch.Program, name string) bool {
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

func trainProgramDeclaresOutput(prog *arch.Program, name string) bool {
	if prog == nil {
		return false
	}
	for _, out := range prog.Outputs {
		if out.Name == name {
			return true
		}
	}
	return false
}

func countProgramOps(prog *arch.Program, code int) int {
	if prog == nil {
		return 0
	}
	count := 0
	for _, op := range prog.Ops {
		if op.Code == code {
			count++
		}
	}
	return count
}

func trainProgramProducesOutput(prog *arch.Program, name string) bool {
	if prog == nil {
		return false
	}
	for _, op := range prog.Ops {
		for _, out := range op.Outputs {
			if out == name {
				return true
			}
		}
	}
	return false
}

func trainPreferredEvalLossOutputName(prog *arch.Program) string {
	if trainProgramDeclaresOutput(prog, "eval_loss") {
		return "eval_loss"
	}
	return "loss"
}

func trainCountOps(prog *arch.Program, code int) int {
	if prog == nil {
		return 0
	}
	n := 0
	for _, op := range prog.Ops {
		if op.Code == code {
			n++
		}
	}
	return n
}

func intSlicesEqualTrain(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func writeMinimalPairBinaryV1ForTest(w *bytes.Buffer, records []minimalPairRecord, vocabSize, maxLen int) error {
	header := []uint32{
		minimalPairBinaryMagic,
		minimalPairBinaryVersion1,
		uint32(vocabSize),
		uint32(maxLen),
		uint32(len(records)),
		0,
	}
	for _, v := range header {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	for _, rec := range records {
		fields := []uint32{uint32(len(rec.Clean)), uint32(len(rec.Corrupt)), uint32(len(rec.ID)), uint32(len(rec.Family))}
		for _, v := range fields {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
		for _, tok := range rec.Clean {
			if err := binary.Write(w, binary.LittleEndian, uint32(tok)); err != nil {
				return err
			}
		}
		for _, tok := range rec.Corrupt {
			if err := binary.Write(w, binary.LittleEndian, uint32(tok)); err != nil {
				return err
			}
		}
		if _, err := w.WriteString(rec.ID); err != nil {
			return err
		}
		if _, err := w.WriteString(rec.Family); err != nil {
			return err
		}
	}
	return nil
}
