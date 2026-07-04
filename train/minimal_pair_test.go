package train

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
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
	pairOut, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2}, Corrupt: []int{1, 3}, Family: "agreement",
	}, 2, false)
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

	eval.output = []float32{0.5, 0.0}
	eval.requestedOutputs = nil
	seqOut, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}}, 2, false)
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
	out, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, Family: "agreement",
	}, 2, true)
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
	seqOut, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}, Span: []int{2, 3}}, 2, true)
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
	out, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2, 3}, Corrupt: []int{1, 9, 3}, Family: "agreement",
	}, 2, false)
	if err != nil {
		t.Fatalf("scoreEBMRecord(pair): %v", err)
	}
	if !reflect.DeepEqual(eval.requestedOutputs, []string{"head_scorer_minimal_pair_scores"}) {
		t.Fatalf("requested outputs=%v", eval.requestedOutputs)
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
	seqOut, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}, Span: []int{1, 3}}, 2, false)
	if err != nil {
		t.Fatalf("scoreEBMRecord(seq): %v", err)
	}
	if seqOut.Score == nil || *seqOut.Score != 3.0 || seqOut.Energy != nil {
		t.Fatalf("seq PLL output=%+v", seqOut)
	}
	if _, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{ID: "seq2", Tokens: []int{1, 2, 3}}, 2, true); err == nil || !strings.Contains(err.Error(), "native energy") {
		t.Fatalf("token energy error=%v, want native energy rejection", err)
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
	output           []float32
	outputs          map[string][]float32
	requestedOutputs []string
}

func (f *fakeEBMEvaluator) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	f.batch = batch
	return 0, nil
}

func (f *fakeEBMEvaluator) EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, outputNames []string) (float32, error) {
	f.batch = batch
	f.requestedOutputs = append([]string(nil), outputNames...)
	return 0, nil
}

func (f *fakeEBMEvaluator) ReadOutput(name string, shape []int) ([]float32, error) {
	if f.outputs != nil {
		if out, ok := f.outputs[name]; ok {
			return append([]float32(nil), out...), nil
		}
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
