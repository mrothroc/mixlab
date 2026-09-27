package main

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/trust"
)

type revocationDelivery struct {
	Endpoint  string `json:"endpoint"`
	Node      string `json:"node,omitempty"`
	Delivered bool   `json:"delivered"`
}

func pushRevocation(ctx context.Context, a trust.Anchor, snapshot trust.SignedSnapshot, mode string, explicit []string) ([]revocationDelivery, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	hints, err := nodeHints(ctx, mode, explicit, discovery.MDNS{})
	// Discovery failure must not suppress explicitly requested destinations.
	if err != nil {
		hints = nil
		for _, endpoint := range explicit {
			hints = append(hints, discovery.Hint{Service: discovery.Node, Endpoint: endpoint})
		}
	}
	seen := map[string]bool{}
	var endpoints []string
	for _, h := range hints {
		if !seen[h.Endpoint] {
			seen[h.Endpoint] = true
			endpoints = append(endpoints, h.Endpoint)
		}
	}
	sort.Strings(endpoints)
	result := make([]revocationDelivery, 0, len(endpoints))
	for _, endpoint := range endpoints {
		r, pushErr := clusterapp.PushNodeSnapshot(ctx, a, snapshot, endpoint, time.Now)
		result = append(result, revocationDelivery{Endpoint: endpoint, Node: r.Node, Delivered: pushErr == nil})
		if pushErr != nil {
			err = errors.Join(err, fmt.Errorf("%s: %w", endpoint, pushErr))
		}
	}
	return result, err
}
