package train

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestPLLMarginPairJSONLBinaryAndPreparePairs(t *testing.T) {
	jsonl := `{"id":"p0","family":"agreement","view_pos":[1,2,3,4],"target_pos_positions":[2],"view_neg":[1,5,3,4],"target_neg_positions":[2],"target_ids":[3]}` + "\n"
	opts := pllMarginPairDecodeOptions{VocabSize: 16, MaxLen: 4, MaskTokenID: 15, RequireFamily: true}
	records, err := decodePLLMarginPairJSONL(strings.NewReader(jsonl), "pairs.jsonl", opts)
	if err != nil {
		t.Fatalf("decode JSONL: %v", err)
	}
	var blob bytes.Buffer
	if err := writePLLMarginPairBinary(&blob, records, 16, 4); err != nil {
		t.Fatalf("write binary: %v", err)
	}
	roundTrip, err := decodePLLMarginPairBinary(bytes.NewReader(blob.Bytes()), "pairs.bin", opts)
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
		t.Fatalf("prepare PLL margin pairs: %v", err)
	}
	if _, err := loadPLLMarginPairs(out, "bin", opts); err != nil {
		t.Fatalf("load prepared binary: %v", err)
	}
}

func TestPLLMarginPairValidationRejectsBadAnnotations(t *testing.T) {
	opts := pllMarginPairDecodeOptions{VocabSize: 16, MaxLen: 4, MaskTokenID: 15, RequireFamily: true}
	for _, tc := range []struct {
		name string
		line string
		want string
	}{
		{"missing positions", `{"id":"p","family":"f","view_pos":[1,2],"view_neg":[3,2],"target_neg_positions":[1],"target_ids":[2]}`, "target_pos_positions"},
		{"target mismatch", `{"id":"p","family":"f","view_pos":[1,2],"target_pos_positions":[1],"view_neg":[3,4],"target_neg_positions":[1],"target_ids":[2]}`, "does not select"},
		{"nonincreasing", `{"id":"p","family":"f","view_pos":[1,2,3],"target_pos_positions":[1,1],"view_neg":[4,2,3],"target_neg_positions":[1,2],"target_ids":[2,3]}`, "strictly increasing"},
		{"identical", `{"id":"p","family":"f","view_pos":[1,2],"target_pos_positions":[1],"view_neg":[1,2],"target_neg_positions":[1],"target_ids":[2]}`, "identical"},
		{"masked target", `{"id":"p","family":"f","view_pos":[1,15],"target_pos_positions":[1],"view_neg":[2,15],"target_neg_positions":[1],"target_ids":[15]}`, "must not contain"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := decodePLLMarginPairJSONL(strings.NewReader(tc.line+"\n"), "pairs", opts)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestAttachPLLMarginPairsDeterministicAndContrastRowsAreNotPrimaryCE(t *testing.T) {
	record := pllMarginPairRecord{
		ID: "p", Family: "agreement",
		ViewPos: []int{1, 2, 3, 4}, TargetPosPositions: []int{1, 2},
		ViewNeg: []int{1, 5, 2, 3}, TargetNegPositions: []int{2, 3},
		TargetIDs:  []int{2, 3},
		viewPosSet: true, targetPosPositionsSet: true, viewNegSet: true, targetNegPositionsSet: true, targetIDsSet: true,
	}
	sampler := &pllMarginPairSampler{records: []pllMarginPairRecord{record}}
	for _, objective := range []string{"mlm", "mntp"} {
		t.Run(objective, func(t *testing.T) {
			cfg := parseTrainPLLMarginConfig(t, objective)
			raw := trainBatch{x: []int{8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11}, y: make([]int, 16)}
			for i := range raw.y {
				raw.y[i] = raw.x[i]
			}
			prepared, err := prepareObjectiveBatch(cfg, raw, 3, objective)
			if err != nil {
				t.Fatalf("prepare objective: %v", err)
			}
			got, err := maybeAttachPLLMarginPairs(sampler, cfg, 3, prepared, 4, 4, objective)
			if err != nil {
				t.Fatalf("attach PLL margin: %v", err)
			}
			if want := []int{1, 15, 15, 4, 1, 5, 15, 15}; !reflect.DeepEqual(got.x[:8], want) {
				t.Fatalf("pair views x=%v, want %v", got.x[:8], want)
			}
			if want := []int{1, 2, 3, 4, 1, 5, 2, 3}; !reflect.DeepEqual(got.unmaskedX[:8], want) {
				t.Fatalf("unmasked=%v, want %v", got.unmaskedX[:8], want)
			}
			if want := []float32{0, 1, 1, 0, 0, 0, 1, 1}; !reflect.DeepEqual(got.pllMarginLossMask[:8], want) {
				t.Fatalf("PLL margin mask=%v, want %v", got.pllMarginLossMask[:8], want)
			}
			if want := []float32{0, 0, 0, 0, 0, 0, 0, 0}; !reflect.DeepEqual(got.lossMask[:8], want) {
				t.Fatalf("pair loss mask=%v, want contrast rows excluded from ordinary CE", got.lossMask[:8])
			}
			again, err := maybeAttachPLLMarginPairs(sampler, cfg, 3, prepared, 4, 4, objective)
			if err != nil {
				t.Fatalf("attach PLL margin again: %v", err)
			}
			if !reflect.DeepEqual(got, again) {
				t.Fatal("same seed and step produced different PLL margin batch")
			}
		})
	}
}

func TestPLLMarginWeightZeroDoesNotAttachPairs(t *testing.T) {
	cfg := parseTrainPLLMarginConfig(t, "mlm")
	cfg.Training.PLLMargin.Weight = 0
	batch := objectiveBatch{x: []int{1, 2, 3, 4, 5, 6, 7, 8}, y: []int{1, 2, 3, 4, 5, 6, 7, 8}, lossMask: make([]float32, 8)}
	got, err := maybeAttachPLLMarginPairs(&pllMarginPairSampler{}, cfg, 0, batch, 2, 4, "mlm")
	if err != nil {
		t.Fatalf("attach zero-weight PLL margin: %v", err)
	}
	if !reflect.DeepEqual(got, batch) {
		t.Fatalf("zero-weight PLL margin changed batch: %+v", got)
	}
}

func parseTrainPLLMarginConfig(t *testing.T, objective string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name":"pll_margin_test", "model_dim":16, "vocab_size":32, "seq_len":4,
		"blocks":[{"type":"plain", "heads":2}],
		"training": {
			"steps":1, "lr":0.001, "seed":42, "batch_tokens":16,
			"objective":"`+objective+`", "mlm_mask_token_id":15, "mlm_mask_prob":0,
			"pll_margin":{"path":"pairs.bin", "anchor_weight":0.5, "batch_fraction":0.5}
		}
	}`), "pll_margin_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}
