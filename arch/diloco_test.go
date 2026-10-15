package arch

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

const dilocoConfig = `{"model_dim":16,"seq_len":4,"vocab_size":32,"blocks":[{"type":"plain","heads":2}],"training":{"optimizer":"adamw","steps":20,"batch_tokens":8,"distributed":{"mode":"diloco","inner_steps":5,"outer_optimizer":"nesterov","outer_lr":1,"outer_momentum":0,"aggregation":"uniform"}}}`

func TestDiLoCoConfig(t *testing.T) {
	c, e := ParseArchConfig([]byte(dilocoConfig), "diloco")
	if e != nil {
		t.Fatal(e)
	}
	s := c.Training.Distributed
	if s.OuterMomentum == nil || *s.OuterMomentum != 0 || s.Backend != "" || s.GradientBucketBytes != 0 || s.GradientAccumulationSteps != 1 {
		t.Fatal(s)
	}
	b, e := json.Marshal(c)
	if e != nil {
		t.Fatal(e)
	}
	r, e := ParseArchConfig(b, "roundtrip")
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(r.Training.Distributed, s) {
		t.Fatal("roundtrip changed algorithm")
	}
	p, e := BuildIRProgramFromConfig(c)
	if e != nil {
		t.Fatal(e)
	}
	c.Training.Distributed = nil
	q, e := BuildIRProgramFromConfig(c)
	if e != nil || !reflect.DeepEqual(p, q) {
		t.Fatal("contracts changed model graph", e)
	}
	for _, tc := range []struct{ from, to string }{
		{`"outer_momentum":0,`, ``}, {`"outer_momentum":0`, `"outer_momentum":null`}, {`"outer_momentum":0`, `"outer_momentum":1`}, {`"outer_momentum":0`, `"outer_momentum":-0.1`},
		{`"inner_steps":5`, `"inner_steps":0`}, {`"inner_steps":5`, `"inner_steps":3`}, {`"outer_lr":1`, `"outer_lr":0`}, {`"outer_optimizer":"nesterov"`, `"outer_optimizer":"adamw"`},
		{`"aggregation":"uniform"`, `"aggregation":"quorum"`}, {`"mode":"diloco"`, `"mode":"diloco","backend":"auto"`}, {`"mode":"diloco"`, `"mode":"diloco","gradient_bucket_bytes":0`},
		{`"mode":"diloco"`, `"mode":"diloco","unknown":0`}, {`"outer_lr":1`, `"outer_lr":1,"outer_lr":2`}, {`"optimizer":"adamw"`, `"optimizer":"muon"`},
	} {
		t.Run(tc.to, func(t *testing.T) {
			if _, e := ParseArchConfig([]byte(strings.Replace(dilocoConfig, tc.from, tc.to, 1)), "bad"); e == nil {
				t.Fatal("accepted invalid config")
			}
		})
	}
	for _, v := range []float64{math.NaN(), math.Inf(1)} {
		r.Training.Distributed.OuterLR = v
		if ValidateDistributedConfig(r) == nil {
			t.Fatal("accepted nonfinite LR")
		}
	}
}

func TestDiLoCoDDPWireUnchanged(t *testing.T) {
	var s DistributedSpec
	if e := json.Unmarshal([]byte(`{"mode":"ddp"}`), &s); e != nil {
		t.Fatal(e)
	}
	b, e := json.Marshal(s)
	if e != nil {
		t.Fatal(e)
	}
	if string(b) != `{"mode":"ddp","backend":"auto","gradient_accumulation_steps":1,"gradient_bucket_bytes":33554432}` {
		t.Fatal(string(b))
	}
	for _, b := range []string{`{"mode":"ddp","outer_lr":0}`, `{"mode":"ddp","outer_momentum":0}`, `{"mode":"ddp","mode":"ddp"}`, `{"mode":"ddp","backend":null}`} {
		if json.Unmarshal([]byte(b), &s) == nil {
			t.Fatal("accepted", b)
		}
	}
}
