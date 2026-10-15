package optimizer

import "fmt"

type Config struct {
	LR          float64
	Momentum    float64
	Aggregation string
	InnerSteps  uint64
}

// Contribution deliberately excludes all provider/workload/private-state data.
type Contribution struct {
	WorkerID        string
	Delta           []Tensor
	Attempted       uint64
	Committed       uint64
	Skipped         uint64
	EffectiveTokens uint64
}

type Result struct {
	Model           []Tensor
	Momentum        []Tensor
	Gradient        []Tensor
	EffectiveTokens uint64
}

// Step returns fresh arrays and never mutates caller state, even on failure.
// Workers reduce in committed cohort order. Each weighted sum is accumulated in
// float64 then rounded once to float32; momentum and model are rounded at their
// respective state boundaries. There is no outer weight decay.
func Step(cfg Config, order []string, model, momentum []Tensor, updates []Contribution) (Result, error) {
	if !finite(cfg.LR) || cfg.LR <= 0 || !finite(cfg.Momentum) || cfg.Momentum < 0 || cfg.Momentum >= 1 || cfg.InnerSteps == 0 || (cfg.Aggregation != "uniform" && cfg.Aggregation != "effective_tokens") {
		return Result{}, fmt.Errorf("invalid outer optimizer configuration")
	}
	if e := ValidateMatch(model, momentum); e != nil {
		return Result{}, e
	}
	if len(order) == 0 || len(order) > 64 || len(order) != len(updates) {
		return Result{}, fmt.Errorf("exact committed cohort required")
	}
	byID := make(map[string]Contribution, len(updates))
	var total uint64
	for _, u := range updates {
		if u.WorkerID == "" || byID[u.WorkerID].WorkerID != "" || u.Attempted != cfg.InnerSteps || u.Committed != cfg.InnerSteps || u.Skipped != 0 || u.EffectiveTokens == 0 {
			return Result{}, fmt.Errorf("invalid worker/counts: exactly H attempts and commits, zero skips required")
		}
		if u.EffectiveTokens > ^uint64(0)-total {
			return Result{}, fmt.Errorf("effective token count overflow")
		}
		total += u.EffectiveTokens
		if e := ValidateMatch(model, u.Delta); e != nil {
			return Result{}, e
		}
		byID[u.WorkerID] = u
	}
	ordered := make([]Contribution, len(order))
	for i, id := range order {
		u, ok := byID[id]
		if !ok {
			return Result{}, fmt.Errorf("missing/duplicate cohort worker %q", id)
		}
		ordered[i] = u
		delete(byID, id)
	}
	out := Result{Model: allocate(model), Momentum: allocate(model), Gradient: allocate(model), EffectiveTokens: total}
	for i, t := range model {
		for j, w := range t.Values {
			var sum float64
			for _, u := range ordered {
				weight := 1 / float64(len(order))
				if cfg.Aggregation == "effective_tokens" {
					weight = float64(u.EffectiveTokens) / float64(total)
				}
				// Explicit conversions require rounding here rather than allowing
				// fused multiply-add to change cancellation across workers.
				sum += float64(weight * float64(u.Delta[i].Values[j]))
			}
			g, e := checked(sum)
			if e != nil {
				return Result{}, e
			}
			m, e := checked(float64(cfg.Momentum*float64(momentum[i].Values[j])) + float64(g))
			if e != nil {
				return Result{}, e
			}
			update := float64(g) + float64(cfg.Momentum*float64(m))
			next, e := checked(float64(w) - float64(cfg.LR*update))
			if e != nil {
				return Result{}, e
			}
			out.Gradient[i].Values[j] = g
			out.Momentum[i].Values[j] = m
			out.Model[i].Values[j] = next
		}
	}
	return out, nil
}
