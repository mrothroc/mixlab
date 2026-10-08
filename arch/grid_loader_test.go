package arch

import (
	"encoding/json"
	"runtime"
	"strings"
	"testing"
)

func TestGridLoaderConfig(t *testing.T) {
	for _, tc := range []struct {
		value string
		bad   bool
	}{
		{`null`, false}, {`{}`, false}, {`{"prefetch_batches":0,"read_workers":1}`, false},
		{`{"prefetch_batches":2,"read_workers":4}`, false},
		{`{"prefetch_batches":-1}`, true}, {`{"prefetch_batches":65}`, true},
		{`{"read_workers":-1}`, true}, {`{"read_workers":257}`, true},
	} {
		c := gridTestConfig(t)
		if err := json.Unmarshal([]byte(tc.value), &c.Training.GridLoader); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(c)
		if err != nil {
			t.Fatal(err)
		}
		got, err := ParseArchConfig(raw, "grid-loader")
		if (err != nil) != tc.bad {
			t.Fatalf("%s: %v", tc.value, err)
		}
		if tc.bad {
			continue
		}
		if tc.value == `null` || tc.value == `{}` {
			if got.Training.GridLoader.EffectivePrefetchBatches() != 2 || got.Training.GridLoader.EffectiveReadWorkers(4) != min(4, runtime.GOMAXPROCS(0)) {
				t.Fatal("defaults")
			}
		} else {
			a, _ := json.Marshal(got.Training.GridLoader)
			var s GridLoaderSpec
			if err = json.Unmarshal(a, &s); err != nil || s.EffectivePrefetchBatches() != c.Training.GridLoader.EffectivePrefetchBatches() {
				t.Fatal("roundtrip", err)
			}
		}
	}
	_, err := ParseArchConfig([]byte(`{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[{"type":"plain","heads":2}],"training":{"grid_loader":{}}}`), "nongrid")
	if err == nil || !strings.Contains(err.Error(), "require dense_regression") {
		t.Fatal(err)
	}
}
