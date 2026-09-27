package main

import (
	"context"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/discovery"
)

func TestAgentNodeAdvertisementUsesOnlyNodeClaims(t *testing.T) {
	claims := nodeDiscoveryClaims(strings.Repeat("a", 32), strings.Repeat("b", 32))
	a := discovery.Advertisement{Service: discovery.Node, Instance: claims.Node, Port: 7445, Claims: claims}
	txt, err := a.TXT()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := discovery.DecodeTXT(discovery.Node, a.Port, txt); err != nil {
		t.Fatal(err)
	}
	ad, err := (discovery.Explicit{}).Advertise(context.Background(), a)
	if err != nil {
		t.Fatal("advertising off must still accept agent schema", err)
	}
	if err := ad.Close(); err != nil {
		t.Fatal(err)
	}
}

type hintProvider struct {
	discovery.Explicit
	hints []discovery.Hint
}

func (p hintProvider) Browse(context.Context, discovery.Service) ([]discovery.Hint, error) {
	return p.hints, nil
}

func TestEnrollmentDiscoveryIsAddressOnlyAndUnambiguous(t *testing.T) {
	hint := discovery.Hint{Service: discovery.Enrollment, Endpoint: "127.0.0.1:7444", Claims: discovery.Claims{RootFingerprint: "attacker-selected", Policies: []string{"trusted_lan"}}}
	for _, test := range []struct {
		name  string
		hints []discovery.Hint
		want  string
	}{
		{"none", nil, ""},
		{"sole", []discovery.Hint{hint}, "https://127.0.0.1:7444"},
		{"duplicate-address", []discovery.Hint{hint, hint}, "https://127.0.0.1:7444"},
		{"ambiguous", []discovery.Hint{hint, {Service: discovery.Enrollment, Endpoint: "127.0.0.2:7444"}}, ""},
		{"wrong-service", []discovery.Hint{{Service: discovery.Node, Endpoint: hint.Endpoint}}, ""},
		{"invalid-address", []discovery.Hint{{Service: discovery.Enrollment, Endpoint: "attacker@localhost:7444"}}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := discoverEnrollmentEndpoint(context.Background(), hintProvider{hints: test.hints})
			if got != test.want || (err == nil) != (test.want != "") {
				t.Fatal(got, err)
			}
		})
	}
}
