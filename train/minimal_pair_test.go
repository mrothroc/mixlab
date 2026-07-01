package train

import (
	"bytes"
	"encoding/json"
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

func TestScoreEBMRecordSequenceAndPair(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl"}`)
	eval := &fakeEBMEvaluator{output: []float32{0.25, 1.25}}
	pairOut, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{
		ID: "pair", Clean: []int{1, 2}, Corrupt: []int{1, 3}, Family: "agreement",
	}, 2)
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

	eval.output = []float32{0.5, 0.0}
	seqOut, err := scoreEBMRecord(cfg, eval, scoreEBMInputRecord{ID: "seq", Tokens: []int{1, 2, 3}}, 2)
	if err != nil {
		t.Fatalf("scoreEBMRecord(seq): %v", err)
	}
	if seqOut.Energy == nil || *seqOut.Energy != 0.5 || seqOut.NTokens != 3 {
		t.Fatalf("seq output=%+v", seqOut)
	}
}

func TestScoreEBMJSONLSummaryIncludesFamilies(t *testing.T) {
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl"}`)
	eval := &fakeEBMEvaluator{output: []float32{0.1, 0.9}}
	var out bytes.Buffer
	err := scoreEBMJSONL(strings.NewReader(`{"id":"p0","clean":[1],"corrupt":[2],"family":"agreement"}`+"\n"), &out, cfg, eval, 2)
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
	batch  objectiveBatch
	output []float32
}

func (f *fakeEBMEvaluator) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	f.batch = batch
	return 0, nil
}

func (f *fakeEBMEvaluator) ReadOutput(name string, shape []int) ([]float32, error) {
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
