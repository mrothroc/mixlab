package train

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestInvariancePairJSONLBinaryAndPreparePairs(t *testing.T) {
	jsonl := `{"id":"p0","family":"distractor_agreement","view_a":[1,2,3,4],"view_a_pos":2,"view_b":[1,5,3,4],"view_b_pos":2}` + "\n"
	opts := invariancePairDecodeOptions{VocabSize: 16, MaxLen: 4, MaskTokenID: 15, RequireFamily: true}
	records, err := decodeInvariancePairJSONL(strings.NewReader(jsonl), "pairs.jsonl", opts)
	if err != nil {
		t.Fatalf("decode JSONL: %v", err)
	}
	var blob bytes.Buffer
	if err := writeInvariancePairBinary(&blob, records, 16, 4); err != nil {
		t.Fatalf("write binary: %v", err)
	}
	roundTrip, err := decodeInvariancePairBinary(bytes.NewReader(blob.Bytes()), "pairs.bin", opts)
	if err != nil {
		t.Fatalf("decode binary: %v", err)
	}
	if !reflect.DeepEqual(roundTrip, records) {
		t.Fatalf("round trip=%+v, want %+v", roundTrip, records)
	}

	dir := t.TempDir()
	in := filepath.Join(dir, "pairs.jsonl")
	out := filepath.Join(dir, "pairs.bin")
	if err := os.WriteFile(in, []byte(jsonl), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runPreparePairsWithOptions(PreparePairsOptions{PairIn: in, PairOut: out, VocabSize: 16, MaxLen: 4}); err != nil {
		t.Fatalf("prepare invariance pairs: %v", err)
	}
	if _, err := loadInvariancePairs(out, "bin", opts); err != nil {
		t.Fatalf("load prepared binary: %v", err)
	}
}

func TestInvariancePairValidationRejectsBadAnnotations(t *testing.T) {
	opts := invariancePairDecodeOptions{VocabSize: 16, MaxLen: 4, MaskTokenID: 15, RequireFamily: true}
	for _, tc := range []struct {
		name string
		line string
		want string
	}{
		{"missing position", `{"id":"p","family":"f","view_a":[1,2],"view_b":[1,3],"view_b_pos":1}`, "view_a_pos"},
		{"target changed", `{"id":"p","family":"f","view_a":[1,2],"view_a_pos":1,"view_b":[1,3],"view_b_pos":1}`, "target tokens must match"},
		{"identical", `{"id":"p","family":"f","view_a":[1,2],"view_a_pos":1,"view_b":[1,2],"view_b_pos":1}`, "identical"},
		{"masked target", `{"id":"p","family":"f","view_a":[1,15],"view_a_pos":1,"view_b":[2,15],"view_b_pos":1}`, "must not contain"},
		{"skipped target", `{"id":"p","family":"f","view_a":[1,2],"view_a_pos":1,"view_b":[3,2],"view_b_pos":1}`, "skip_token_ids"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			caseOpts := opts
			if tc.name == "skipped target" {
				caseOpts.SkipTokenIDs = map[int]bool{2: true}
			}
			_, err := decodeInvariancePairJSONL(strings.NewReader(tc.line+"\n"), "pairs", caseOpts)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestAttachInvariancePairsDeterministicMLMAndMNTP(t *testing.T) {
	record := invariancePairRecord{
		ID: "p", Family: "distractor_agreement",
		ViewA: []int{1, 2, 3, 4}, ViewAPos: 2,
		ViewB: []int{1, 5, 3, 4}, ViewBPos: 2,
		viewAPosSet: true, viewBPosSet: true,
	}
	sampler := &invariancePairSampler{records: []invariancePairRecord{record}}
	for _, objective := range []string{"mlm", "mntp"} {
		t.Run(objective, func(t *testing.T) {
			cfg := parseTrainInvarianceConfig(t, objective)
			raw := trainBatch{x: []int{8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11}, y: make([]int, 16)}
			for i := range raw.y {
				raw.y[i] = raw.x[i]
			}
			prepared, err := prepareObjectiveBatch(cfg, raw, 3, objective)
			if err != nil {
				t.Fatalf("prepare objective: %v", err)
			}
			got, err := maybeAttachInvariancePairs(sampler, cfg, 3, prepared, 4, 4, objective)
			if err != nil {
				t.Fatalf("attach invariance: %v", err)
			}
			wantX := []int{1, 2, 15, 4, 1, 5, 15, 4}
			if !reflect.DeepEqual(got.x[:8], wantX) {
				t.Fatalf("pair views x=%v, want %v", got.x[:8], wantX)
			}
			if !reflect.DeepEqual(got.unmaskedX[:8], []int{1, 2, 3, 4, 1, 5, 3, 4}) {
				t.Fatalf("unmasked=%v", got.unmaskedX[:8])
			}
			if !reflect.DeepEqual(got.invarianceLossMask[:8], []float32{0, 0, 1, 0, 0, 0, 1, 0}) {
				t.Fatalf("invariance mask=%v", got.invarianceLossMask[:8])
			}
			if got.invarianceLossMask[8] != 0 || got.invarianceLossMask[12] != 0 {
				t.Fatalf("inactive rows have invariance mask=%v", got.invarianceLossMask[8:])
			}
			again, err := maybeAttachInvariancePairs(sampler, cfg, 3, prepared, 4, 4, objective)
			if err != nil {
				t.Fatalf("attach invariance again: %v", err)
			}
			if !reflect.DeepEqual(got, again) {
				t.Fatal("same seed and step produced different invariance batch")
			}
		})
	}
}

func TestInvarianceWeightZeroDoesNotAttachPairs(t *testing.T) {
	cfg := parseTrainInvarianceConfig(t, "mlm")
	cfg.Training.Invariance.Weight = 0
	batch := objectiveBatch{x: []int{1, 2, 3, 4, 5, 6, 7, 8}, y: []int{1, 2, 3, 4, 5, 6, 7, 8}, lossMask: make([]float32, 8)}
	got, err := maybeAttachInvariancePairs(&invariancePairSampler{}, cfg, 0, batch, 2, 4, "mlm")
	if err != nil {
		t.Fatalf("attach zero-weight invariance: %v", err)
	}
	if !reflect.DeepEqual(got, batch) {
		t.Fatalf("zero-weight invariance changed batch: %+v", got)
	}
}

func parseTrainInvarianceConfig(t *testing.T, objective string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name":"invariance_test", "model_dim":16, "vocab_size":32, "seq_len":4,
		"blocks":[{"type":"plain", "heads":2}],
		"training": {
			"steps":1, "lr":0.001, "seed":42, "batch_tokens":16,
			"objective":"`+objective+`", "mlm_mask_token_id":15, "mlm_mask_prob":0,
			"invariance":{"path":"pairs.bin", "weight":0.1, "batch_fraction":0.5}
		}
	}`), "invariance_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}
