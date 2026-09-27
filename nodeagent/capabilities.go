package nodeagent

import (
	"fmt"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

// Validate checks the public response contract, not its authenticity. The
// caller must bind Node to the peer authenticated on the response channel.
func (c Capabilities) Validate() error {
	if c.TransportEndpoint != "" {
		if err := ValidateTransportEndpoint(c.TransportEndpoint); err != nil {
			return err
		}
	}
	if c.Version != CapabilityVersion || !identifier(c.Node) || c.Generation == 0 || c.ObservedAt <= 0 || c.ProbeObservedAt <= 0 || c.ProbeObservedAt > c.ObservedAt || c.DisplayName == "" || len(c.DisplayName) > 128 {
		return fmt.Errorf("invalid node capability identity or observation")
	}
	for _, r := range c.DisplayName {
		if r < 32 || r == 127 {
			return fmt.Errorf("invalid capability display name")
		}
	}
	if err := c.Probe.Validate(); err != nil {
		return err
	}
	if err := c.Limits.Validate(); err != nil {
		return err
	}
	if len(c.ManifestVersions) == 0 || len(c.ManifestVersions) > 16 || !slices.IsSorted(c.ManifestVersions) {
		return fmt.Errorf("invalid manifest capability versions")
	}
	for i, v := range c.ManifestVersions {
		if !selector(v) || i > 0 && v == c.ManifestVersions[i-1] {
			return fmt.Errorf("invalid manifest capability version")
		}
	}
	if c.Datasets == nil || len(c.Datasets) > 1024 {
		return fmt.Errorf("invalid capability dataset catalog")
	}
	seen := map[string]bool{}
	for _, d := range c.Datasets {
		if !selector(d.Selector) || !contract.Hash(d.ID) || seen[d.Selector] {
			return fmt.Errorf("invalid capability dataset identity")
		}
		seen[d.Selector] = true
	}
	a := c.Availability
	if a.NodeVersion == 0 || a.CapabilityGeneration != c.Generation || a.Available != (a.Lease == nil) {
		return fmt.Errorf("inconsistent node availability")
	}
	if a.Lease != nil {
		if err := a.Lease.validate(c.Node); err != nil {
			return err
		}
		if a.Lease.State == Released || a.Lease.CapabilityGeneration != c.Generation {
			return fmt.Errorf("inconsistent active lease")
		}
	}
	want := a.Available && c.Probe.Available && c.Probe.MLXSupported && slices.Contains(c.Probe.Backends, "ring")
	if c.Recruitable != want {
		return fmt.Errorf("inconsistent recruitability")
	}
	return nil
}

// Freshness is selection policy, separate from the versioned wire contract.
func (c Capabilities) Fresh(now time.Time, observationAge, probeAge time.Duration) bool {
	observed, probed := time.Unix(c.ObservedAt, 0), time.Unix(c.ProbeObservedAt, 0)
	return !observed.After(now) && !probed.After(now) && now.Sub(observed) <= observationAge && now.Sub(probed) <= probeAge
}
