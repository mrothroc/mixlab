package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerprobe"
)

const ProfileVersion = "mixlab_node_profile_v1"
const CapabilityVersion = "mixlab_node_capabilities_v1"
const profileFile = "node-profile.json"
const profileClaim = "node-profile.claim"
const profileReady = "node-profile.ready"

type profileIdentity struct{ Node, InitialHash string }

type Dataset struct {
	Selector string `json:"selector"`
	ID       string `json:"id"`
}
type LocalDataset struct {
	Dataset
	TrainPattern string `json:"train_pattern"`
}
type Profile struct {
	TransportEndpoint string             `json:"transport_endpoint,omitempty"`
	Version           string             `json:"version"`
	Node              string             `json:"node"`
	DisplayName       string             `json:"display_name"`
	Generation        uint64             `json:"generation"`
	Probe             workerprobe.Report `json:"probe"`
	ProbeObservedAt   int64              `json:"probe_observed_at"`
	Limits            nodejob.Limits     `json:"limits"`
	Datasets          []LocalDataset     `json:"datasets"`
}

func selector(s string) bool {
	if s == "" || s == "." || s == ".." || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if c != '_' && c != '-' && c != '.' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') {
			return false
		}
	}
	return true
}

func (p Profile) Validate() error {
	if p.TransportEndpoint != "" {
		if err := ValidateTransportEndpoint(p.TransportEndpoint); err != nil {
			return err
		}
	}
	if p.Version != ProfileVersion || !identifier(p.Node) || p.Generation == 0 || p.ProbeObservedAt <= 0 || p.DisplayName == "" || len(p.DisplayName) > 128 || len(p.Datasets) > 1024 {
		return fmt.Errorf("invalid local node profile")
	}
	for _, c := range p.DisplayName {
		if c < 32 || c == 127 {
			return fmt.Errorf("display name contains control characters")
		}
	}
	if err := p.Probe.Validate(); err != nil {
		return err
	}
	l := p.Limits
	if err := l.Validate(); err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, d := range p.Datasets {
		if !selector(d.Selector) || !contract.Hash(d.ID) || seen[d.Selector] || !filepath.IsAbs(d.TrainPattern) || filepath.Clean(d.TrainPattern) != d.TrainPattern || len(d.TrainPattern) > 4096 {
			return fmt.Errorf("invalid local dataset registration")
		}
		seen[d.Selector] = true
	}
	return nil
}

func (s *Store) loadProfile() ([]byte, Profile, error) {
	ready, err := s.path.ReadFileLimit(profileReady, 32)
	if err != nil {
		return nil, Profile{}, err
	}
	if string(ready) != s.node {
		return nil, Profile{}, fmt.Errorf("profile publication incomplete")
	}
	b, err := s.path.ReadFileLimit(profileFile, 1<<20)
	if err != nil {
		return nil, Profile{}, err
	}
	var p Profile
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, p, err
	}
	again, err := json.Marshal(p)
	if err != nil || !bytes.Equal(b, again) || p.Node != s.node {
		return nil, p, fmt.Errorf("invalid protected node profile")
	}
	if err := p.Validate(); err != nil {
		return nil, p, err
	}
	claim, err := s.path.ReadFileLimit(profileClaim, 256)
	if err != nil {
		return nil, p, err
	}
	var identity profileIdentity
	if json.Unmarshal(claim, &identity) != nil || identity.Node != s.node || !contract.Hash(identity.InitialHash) {
		return nil, p, fmt.Errorf("profile identity claim changed")
	}
	return b, p, nil
}

// InstallProfile is an administrator-local operation. Controller requests
// cannot select executable/dataset paths or replace capability observations.
// Profile-first publication is completed by exact retry after interruption.
func (s *Store) InstallProfile(ctx context.Context, p Profile) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if p.Node != s.node {
		return fmt.Errorf("profile belongs to another node")
	}
	encoded, err := json.Marshal(p)
	if err != nil {
		return err
	}
	var copy Profile
	if err := json.Unmarshal(encoded, &copy); err != nil {
		return err
	}
	p = copy
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		leaseBytes, r, err := s.load()
		if err != nil {
			return err
		}
		if r.Active != "" {
			return fmt.Errorf("cannot change a leased node profile")
		}
		old, existing, err := s.loadProfile()
		switch {
		case errors.Is(err, os.ErrNotExist):
			if p.Generation != r.CapabilityGeneration {
				return fmt.Errorf("missing initialized node profile")
			}
			if _, err := s.path.ReadFileLimit(profileReady, 32); err == nil {
				return fmt.Errorf("published profile history missing")
			} else if !errors.Is(err, os.ErrNotExist) {
				return err
			}
			claim, _ := json.Marshal(profileIdentity{s.node, digest(p)})
			if err := publishInitial(s.path, profileClaim, claim); err != nil {
				return err
			}
			if err := publishInitial(s.path, profileFile, encoded); err != nil {
				return err
			}
			if err := s.path.CompareAndSwap(profileReady, nil, []byte(s.node)); err != nil {
				return err
			}
			old = encoded
		case err != nil:
			return err
		case bytes.Equal(old, encoded):
			if p.Generation == r.CapabilityGeneration {
				return nil
			}
			if p.Generation != r.CapabilityGeneration+1 {
				return fmt.Errorf("profile generation diverged")
			}
		default:
			if existing.Generation != r.CapabilityGeneration || p.Generation != existing.Generation+1 {
				return fmt.Errorf("profile update must advance one generation")
			}
		}
		if !bytes.Equal(old, encoded) {
			if err := s.path.CompareAndSwap(profileFile, old, encoded); err != nil {
				return err
			}
		}
		if r.CapabilityGeneration != p.Generation {
			r.CapabilityGeneration = p.Generation
			r.NodeVersion++
			return s.save(leaseBytes, r)
		}
		return nil
	})
}

type Capabilities struct {
	TransportEndpoint string             `json:"transport_endpoint,omitempty"`
	Version           string             `json:"version"`
	Node              string             `json:"node"`
	DisplayName       string             `json:"display_name"`
	Generation        uint64             `json:"generation"`
	ObservedAt        int64              `json:"observed_at"`
	Probe             workerprobe.Report `json:"probe"`
	ProbeObservedAt   int64              `json:"probe_observed_at"`
	ManifestVersions  []string           `json:"manifest_versions"`
	Limits            nodejob.Limits     `json:"limits"`
	Datasets          []Dataset          `json:"datasets"`
	Availability      Availability       `json:"availability"`
	Recruitable       bool               `json:"recruitable"`
}

func (s *Store) Capabilities(ctx context.Context, actor trust.AuthenticatedPrincipal, now time.Time) (out Capabilities, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, p, err := s.loadProfile()
		if err != nil {
			return err
		}
		available, err := s.Availability(now)
		if err != nil {
			return err
		}
		if p.Generation != available.CapabilityGeneration {
			return fmt.Errorf("profile publication needs reconciliation")
		}
		out = Capabilities{Version: CapabilityVersion, Node: s.node, DisplayName: p.DisplayName, Generation: p.Generation, ObservedAt: now.Unix(), Probe: p.Probe, ProbeObservedAt: p.ProbeObservedAt, ManifestVersions: []string{nodejob.Version}, Limits: p.Limits, Availability: available, Datasets: []Dataset{}}
		out.TransportEndpoint = p.TransportEndpoint
		for _, d := range p.Datasets {
			out.Datasets = append(out.Datasets, d.Dataset)
		}
		out.Recruitable = available.Available && p.Probe.Available && p.Probe.MLXSupported && slices.Contains(p.Probe.Backends, "ring")
		return nil
	})
	return out, err
}

type PreparationPorts struct {
	Clock           func() time.Time
	Probe           func(context.Context) (workerprobe.Report, error)
	DatasetID       func(context.Context, string) (string, error)
	ArtifactPresent func(context.Context, nodejob.ArtifactRef) error
}

// PrepareChecked rechecks the installed executable/device and local dataset
// content after reservation. The active lease freezes profile replacement.
// The dataset path comes only from the local catalog, never from the manifest.
func (s *Store) PrepareChecked(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted nodejob.Accepted, version uint64, ports PreparationPorts, now time.Time) (Job, error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return Job{}, err
	}
	signed, _, err := accepted.Value()
	if err != nil {
		return Job{}, err
	}
	m := signed.Manifest
	if m.Controller != actor.Principal || m.Node != s.node {
		return Job{}, fmt.Errorf("wrong preparation owner")
	}
	if ports.Clock == nil || ports.Probe == nil || ports.DatasetID == nil || ports.ArtifactPresent == nil {
		return Job{}, fmt.Errorf("preparation verification ports required")
	}
	_, profile, err := s.loadProfile()
	if err != nil {
		return Job{}, err
	}
	av, err := s.Availability(now)
	if err != nil {
		return Job{}, err
	}
	if av.Lease == nil || av.Lease.ID != m.Lease || av.Lease.Controller != actor.Principal || av.Lease.CapabilityGeneration != profile.Generation || av.CapabilityGeneration != profile.Generation || now.Unix() >= av.Lease.Expires || (av.Lease.State != Reserved && av.Lease.State != Prepared) || now.Unix() >= m.Expires {
		return Job{}, fmt.Errorf("reservation/profile mismatch")
	}
	if !m.Limits.Within(profile.Limits) {
		return Job{}, fmt.Errorf("job exceeds node resource policy")
	}
	probe, err := ports.Probe(ctx)
	if err != nil {
		return Job{}, err
	}
	if err := probe.Validate(); err != nil {
		return Job{}, err
	}
	if !probe.Available || !probe.MLXSupported || probe.BuildID != m.BuildID || !sameProbeContract(probe, profile.Probe) || !slices.Contains(probe.Backends, "ring") {
		return Job{}, fmt.Errorf("installed worker capabilities changed or unavailable")
	}
	if probe.FreeMemoryBytes != nil && *probe.FreeMemoryBytes < m.Limits.MemoryBytes {
		return Job{}, fmt.Errorf("insufficient observed free device memory")
	}
	path := ""
	for _, d := range profile.Datasets {
		if d.Selector == m.DatasetSelector && d.ID == m.DatasetID {
			path = d.TrainPattern
			break
		}
	}
	if path == "" {
		return Job{}, fmt.Errorf("logical dataset not registered with the required identity")
	}
	got, err := ports.DatasetID(ctx, path)
	if err != nil {
		return Job{}, err
	}
	if got != m.DatasetID {
		return Job{}, fmt.Errorf("local dataset content changed")
	}
	for _, a := range m.Artifacts {
		if err := ports.ArtifactPresent(ctx, a); err != nil {
			return Job{}, err
		}
	}
	return s.PrepareJob(ctx, actor, accepted, version, ports.Clock())
}
func sameProbeContract(a, b workerprobe.Report) bool {
	a.FreeMemoryBytes = nil
	b.FreeMemoryBytes = nil
	return digest(a) == digest(b)
}
