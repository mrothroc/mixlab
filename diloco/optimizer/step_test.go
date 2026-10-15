package optimizer

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func tensor(values ...float32) []Tensor {
	return []Tensor{{Name: "w0", Shape: []int{len(values)}, Values: append([]float32(nil), values...)}}
}
func contribution(id string, delta []Tensor, tokens uint64) Contribution {
	return Contribution{WorkerID: id, Delta: delta, Attempted: 2, Committed: 2, EffectiveTokens: tokens}
}
func near(t *testing.T, got, want []Tensor) {
	t.Helper()
	if e := ValidateMatch(got, want); e != nil {
		t.Fatal(e)
	}
	for i, w := range want {
		for j, v := range w.Values {
			if math.Abs(float64(got[i].Values[j])-float64(v)) > 1e-6+1e-6*math.Abs(float64(v)) {
				t.Fatalf("%s[%d]: got %.9g want %.9g", w.Name, j, got[i].Values[j], v)
			}
		}
	}
}

func TestOneWorkerReplacementAndDeltaSign(t *testing.T) {
	base, local := tensor(10, -2, 0), tensor(7, 4, .25)
	d, e := Delta(base, local)
	if e != nil {
		t.Fatal(e)
	}
	near(t, d, tensor(3, -6, -.25))
	zero, _ := ZerosLike(base)
	r, e := Step(Config{LR: 1, Aggregation: "uniform", InnerSteps: 2}, []string{"w"}, base, zero, []Contribution{contribution("w", d, 8)})
	if e != nil {
		t.Fatal(e)
	}
	near(t, r.Model, local)
	near(t, base, tensor(10, -2, 0))
	r.Model[0].Values[0] = 100
	near(t, local, tensor(7, 4, .25))
}

func TestQuadraticOracleFixtures(t *testing.T) {
	b, e := os.ReadFile("testdata/quadratics.json")
	if e != nil {
		t.Fatal(e)
	}
	var fixtures []struct {
		Name        string    `json:"name"`
		Momentum    float64   `json:"momentum"`
		LR          float64   `json:"outer_lr"`
		Aggregation string    `json:"aggregation"`
		Tokens      []uint64  `json:"tokens"`
		Models      []float32 `json:"models"`
		Momenta     []float32 `json:"momenta"`
	}
	if e = json.Unmarshal(b, &fixtures); e != nil {
		t.Fatal(e)
	}
	for _, f := range fixtures {
		t.Run(f.Name, func(t *testing.T) {
			model := tensor(4)
			momentum := tensor(0)
			for round, want := range f.Models {
				// Two gradient steps of size .25 on .5*(theta-target)^2, with
				// targets 0 and 2. This is an independent local-work oracle, not AdamW.
				locals := make([][]Tensor, 2)
				for i, target := range []float64{0, 2} {
					x := float64(model[0].Values[0])
					for step := 0; step < 2; step++ {
						x -= .25 * (x - target)
					}
					locals[i] = tensor(float32(x))
				}
				d0, _ := Delta(model, locals[0])
				d1, _ := Delta(model, locals[1])
				u := []Contribution{contribution("a", d0, f.Tokens[0]), contribution("b", d1, f.Tokens[1])}
				r, e := Step(Config{LR: f.LR, Momentum: f.Momentum, Aggregation: f.Aggregation, InnerSteps: 2}, []string{"a", "b"}, model, momentum, u)
				if e != nil {
					t.Fatal(e)
				}
				reverse, e := Step(Config{LR: f.LR, Momentum: f.Momentum, Aggregation: f.Aggregation, InnerSteps: 2}, []string{"a", "b"}, model, momentum, []Contribution{u[1], u[0]})
				if e != nil || !reflect.DeepEqual(r, reverse) {
					t.Fatal("arrival order changes reduction", e)
				}
				near(t, r.Model, tensor(want))
				near(t, r.Momentum, tensor(f.Momenta[round]))
				model, momentum = r.Model, r.Momentum
			}
		})
	}
}

func TestWeightedDenominatorsAndZeroDecay(t *testing.T) {
	base := tensor(5)
	mom := tensor(0)
	updates := []Contribution{contribution("a", tensor(2), 1), contribution("b", tensor(-2), 3)}
	cfg := Config{LR: 1, Aggregation: "effective_tokens", InnerSteps: 2}
	r, e := Step(cfg, []string{"a", "b"}, base, mom, updates)
	if e != nil {
		t.Fatal(e)
	}
	near(t, r.Model, tensor(6))
	if r.EffectiveTokens != 4 {
		t.Fatal(r.EffectiveTokens)
	}
	cfg.Aggregation = "uniform"
	r, e = Step(cfg, []string{"a", "b"}, base, mom, updates)
	if e != nil {
		t.Fatal(e)
	}
	near(t, r.Model, base)
}

func TestRejectBeforeMutation(t *testing.T) {
	base, mom := tensor(4), tensor(.1)
	cfg := Config{LR: .5, Momentum: .9, Aggregation: "uniform", InnerSteps: 2}
	for name, mutate := range map[string]func(*Config, *[]string, *[]Tensor, *[]Tensor, *[]Contribution){
		"skip":          func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) { (*u)[0].Skipped = 1 },
		"extra_attempt": func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) { (*u)[0].Attempted++ },
		"zero_tokens":   func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) { (*u)[0].EffectiveTokens = 0 },
		"overflow_counts": func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) {
			(*u)[0].EffectiveTokens = ^uint64(0)
		},
		"unknown_worker":   func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) { (*u)[0].WorkerID = "other" },
		"duplicate_worker": func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) { (*u)[0].WorkerID = "b" },
		"duplicate_cohort": func(_ *Config, o *[]string, _ *[]Tensor, _ *[]Tensor, _ *[]Contribution) { (*o)[1] = "a" },
		"zero_lr":          func(c *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, _ *[]Contribution) { c.LR = 0 },
		"nan_lr":           func(c *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, _ *[]Contribution) { c.LR = math.NaN() },
		"weight_overflow":  func(c *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, _ *[]Contribution) { c.LR = math.MaxFloat64 },
		"momentum_overflow": func(_ *Config, _ *[]string, _ *[]Tensor, m *[]Tensor, u *[]Contribution) {
			(*m)[0].Values[0] = math.MaxFloat32
			(*u)[0].Delta = tensor(math.MaxFloat32)
			(*u)[1].Delta = tensor(math.MaxFloat32)
		},
		"shape": func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) {
			(*u)[0].Delta[0].Shape = []int{1, 1}
		},
		"name": func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) {
			(*u)[0].Delta[0].Name = "other"
		},
		"nan": func(_ *Config, _ *[]string, _ *[]Tensor, _ *[]Tensor, u *[]Contribution) {
			(*u)[0].Delta[0].Values[0] = float32(math.NaN())
		},
	} {
		t.Run(name, func(t *testing.T) {
			c := cfg
			o := []string{"a", "b"}
			b := tensor(base[0].Values...)
			m := tensor(mom[0].Values...)
			u := []Contribution{contribution("a", tensor(1), 2), contribution("b", tensor(2), 2)}
			mutate(&c, &o, &b, &m, &u)
			beforeB := append([]float32(nil), b[0].Values...)
			beforeM := append([]float32(nil), m[0].Values...)
			r, e := Step(c, o, b, m, u)
			if e == nil || r.Model != nil {
				t.Fatal("invalid candidate accepted")
			}
			if !reflect.DeepEqual(b[0].Values, beforeB) || !reflect.DeepEqual(m[0].Values, beforeM) {
				t.Fatal("mutated state on failure")
			}
		})
	}
	if _, e := Delta(tensor(math.MaxFloat32), tensor(-math.MaxFloat32)); e == nil {
		t.Fatal("delta overflow accepted")
	}
	var negative struct {
		Tokens uint64 `json:"tokens"`
	}
	if json.Unmarshal([]byte(`{"tokens":-1}`), &negative) == nil {
		t.Fatal("negative count decoded")
	}
}

func TestTensorLayoutValidation(t *testing.T) {
	for _, bad := range [][]Tensor{nil, {{Name: "w", Shape: []int{MaxElements, 2}, Values: nil}}, {{Name: "w", Shape: []int{0}, Values: nil}}, {{Name: "w", Shape: []int{1}, Values: []float32{float32(math.Inf(1))}}}, {{Name: "w", Shape: []int{2}, Values: []float32{1}}}, {{Name: "w", Shape: []int{1}, Values: []float32{1}}, {Name: "w", Shape: []int{1}, Values: []float32{1}}}} {
		if Validate(bad) == nil {
			t.Fatal("accepted bad layout")
		}
	}
}

func TestCanonicalReductionOrderAndMultipleTensors(t *testing.T) {
	base := []Tensor{{Name: "matrix", Shape: []int{1, 2}, Values: []float32{4, -4}}, {Name: "bias", Shape: []int{1}, Values: []float32{1}}}
	mom, _ := ZerosLike(base)
	makeDelta := func(v float32) []Tensor {
		d, _ := ZerosLike(base)
		d[0].Values[0], d[0].Values[1], d[1].Values[0] = v, -v, v
		return d
	}
	updates := []Contribution{contribution("z", makeDelta(1e20), 1), contribution("a", makeDelta(-1e20), 1), contribution("m", makeDelta(3), 1)}
	cfg := Config{LR: 1, Aggregation: "uniform", InnerSteps: 2}
	r, e := Step(cfg, []string{"z", "a", "m"}, base, mom, updates)
	if e != nil {
		t.Fatal(e)
	}
	want := []Tensor{{Name: "matrix", Shape: []int{1, 2}, Values: []float32{3, -3}}, {Name: "bias", Shape: []int{1}, Values: []float32{0}}}
	near(t, r.Model, want)
	reverse, e := Step(cfg, []string{"z", "a", "m"}, base, mom, []Contribution{updates[2], updates[1], updates[0]})
	if e != nil || !reflect.DeepEqual(r, reverse) {
		t.Fatal("arrival order changed cancellation/rounding")
	}
	updates[0].Delta[0], updates[0].Delta[1] = updates[0].Delta[1], updates[0].Delta[0]
	if _, e = Step(cfg, []string{"z", "a", "m"}, base, mom, updates); e == nil {
		t.Fatal("reordered tensor layout accepted")
	}
}
