package workload

import (
	"context"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/trust"
)

func TestLearnerWorkloadIssuanceAndProfileSeparation(t *testing.T) {
	f := newFixture(t)
	s := f.grant.Request.Request.Scope
	s.Profile = LearnerVersion
	s.Group = ""
	s.Member = ""
	s.MembershipHash = ""
	s.Rank = 0
	s.Worker = randomID(t)
	s.Coordinator = randomID(t)
	s.PlanHash = strings.Repeat("c", 64)
	f.grant = f.sign(t, s, newKey(t))
	out, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
	check(t, err)
	check(t, ValidateResult(f.a.Anchor, f.grant, out, f.now))
	if out.Binding.Version != trust.LearnerBindingVersion || out.Binding.Worker != s.Worker || out.Binding.PlanHash != s.PlanHash || out.Binding.Group != "" || out.Binding.Audience != "diloco/learner" {
		t.Fatal("wrong learner profile", out.Binding)
	}
	for name, change := range map[string]func(*trust.WorkloadBinding){
		"worker": func(b *trust.WorkloadBinding) { b.Worker = randomID(t) }, "plan": func(b *trust.WorkloadBinding) { b.PlanHash = strings.Repeat("f", 64) }, "attempt": func(b *trust.WorkloadBinding) { b.Attempt = randomID(t) }, "run": func(b *trust.WorkloadBinding) { b.Run = randomID(t) }, "audience": func(b *trust.WorkloadBinding) { b.Audience = Audience }, "role": func(b *trust.WorkloadBinding) { b.Role = trust.Controller }, "coordinator": func(b *trust.WorkloadBinding) { b.Coordinator = randomID(t) },
	} {
		t.Run(name, func(t *testing.T) {
			bad := out.Binding
			change(&bad)
			if trust.VerifyWorkload(f.a.Anchor, f.view, out.Chain, bad, f.now) == nil {
				t.Fatal("foreign binding accepted")
			}
		})
	}
	for _, change := range []func(*Scope){func(s *Scope) { s.Group = randomID(t) }, func(s *Scope) { s.Rank = 1 }, func(s *Scope) { s.Profile = "" }, func(s *Scope) { s.PlanHash = "" }, func(s *Scope) { s.Worker = "" }} {
		bad := s
		change(&bad)
		if bad.Validate() == nil {
			t.Fatal("mixed profile accepted")
		}
	}
	reopened, err := Open(f.path, f.a, f.view, f.now)
	check(t, err)
	again, err := reopened.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
	check(t, err)
	if again.Binding != out.Binding || f.issuer.calls != 1 {
		t.Fatal("retry minted another identity")
	}
}
