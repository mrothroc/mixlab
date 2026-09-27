package recruitment

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/workload"
)

const launchVersion = "mixlab_recruitment_attempt_v1"
const launchFile = "launch.json"

// LaunchPlan fixes the selected cohort before the first remote reservation.
// Addresses are routes; Node identities, not routes, authorize every operation.
type LaunchPlan struct {
	Cluster    string    `json:"cluster"`
	Controller string    `json:"controller"`
	Run        string    `json:"run"`
	Attempt    string    `json:"attempt"`
	Selection  Selection `json:"selection"`
	TTLSeconds int       `json:"ttl_seconds"`
}

type launchNode struct {
	Candidate  Candidate                   `json:"candidate"`
	Reserve    nodeagent.Reserve           `json:"reserve"`
	Touched    bool                        `json:"touched"`
	Fenced     bool                        `json:"fenced"`
	Lease      *nodeagent.Lease            `json:"lease,omitempty"`
	Manifest   *nodejob.Signed             `json:"manifest,omitempty"`
	Prepared   *nodeagent.PreparedWorkload `json:"prepared,omitempty"`
	Grant      *workload.Grant             `json:"grant,omitempty"`
	Credential *workload.Result            `json:"credential,omitempty"`
	Activated  *nodeagent.Job              `json:"activated,omitempty"`
	Start      nodeagent.StartCommand      `json:"start"`
	Started    *nodeagent.Job              `json:"started,omitempty"`
	Released   bool                        `json:"released"`
	Renewal    *nodeagent.LeaseCommand     `json:"renewal,omitempty"`
}
type launchRecord struct {
	Version   string                 `json:"version"`
	Plan      LaunchPlan             `json:"plan"`
	Phase     string                 `json:"phase"`
	Nodes     []launchNode           `json:"nodes"`
	Transport *grouptransport.Signed `json:"transport,omitempty"`
}

// LaunchPorts keeps recruitment independent of HTTP, key storage and issuance.
// Implementations authenticate the exact candidate before transmitting bodies.
// Builders/signers return values; they cannot start workers or mutate leases.
type LaunchPorts struct {
	Reserve          func(context.Context, Candidate, nodeagent.Reserve) (nodeagent.Lease, error)
	LeaseStatus      func(context.Context, Candidate, string) (nodeagent.Lease, error)
	AbortReservation func(context.Context, Candidate, nodeagent.Reserve) (nodeagent.ReservationAbort, error)
	Renew            func(context.Context, Candidate, nodeagent.LeaseCommand) (nodeagent.Lease, error)
	BuildJobs        func(context.Context, LaunchPlan, []nodeagent.Lease) ([]nodejob.Signed, error)
	Prepare          func(context.Context, Candidate, uint64, nodejob.Signed) (nodeagent.PreparedWorkload, error)
	Grant            func(context.Context, workload.SignedRequest) (workload.Grant, error)
	Issue            func(context.Context, workload.Grant) (workload.Result, error)
	BuildTransport   func(context.Context, LaunchPlan, []nodejob.Signed, []workload.Result) (grouptransport.Signed, error)
	Activate         func(context.Context, Candidate, uint64, grouptransport.Signed, workload.Grant, workload.Result) (nodeagent.Job, error)
	Start            func(context.Context, Candidate, nodeagent.StartCommand) (nodeagent.Job, error)
	Status           func(context.Context, Candidate, string) (nodeagent.Job, error)
	Clock            func() time.Time
}

type Launch struct{ path statehome.Path }

func randomID() (string, error) {
	var b [16]byte
	_, err := rand.Read(b[:])
	return hex.EncodeToString(b[:]), err
}

func (p LaunchPlan) validate() error {
	for _, id := range []string{p.Cluster, p.Controller, p.Run, p.Attempt} {
		if !canonicalHex(id, 16) {
			return fmt.Errorf("canonical launch identities required")
		}
	}
	s := p.Selection
	if s.Policy != SelectionPolicy || s.Requirements.Cluster != p.Cluster || len(s.Selected) != s.Requirements.Count || p.TTLSeconds < 60 || p.TTLSeconds > 3600 {
		return fmt.Errorf("complete selected cohort and lease TTL [60,3600] required")
	}
	if err := s.Requirements.Validate(); err != nil {
		return err
	}
	previous := ""
	for _, c := range s.Selected {
		if c.Capabilities.Validate() != nil || !c.Capabilities.Recruitable || c.Capabilities.Node <= previous || nodeagent.ValidateTransportEndpoint(c.Capabilities.TransportEndpoint) != nil {
			return fmt.Errorf("invalid ordered selected node")
		}
		previous = c.Capabilities.Node
	}
	return nil
}

func BeginLaunch(path statehome.Path, plan LaunchPlan) (*Launch, error) {
	if path.Kind() != statehome.Principal {
		return nil, fmt.Errorf("controller-owned launch context required")
	}
	if err := plan.validate(); err != nil {
		return nil, err
	}
	r := launchRecord{Version: launchVersion, Plan: plan, Phase: "new", Nodes: make([]launchNode, len(plan.Selection.Selected))}
	for i, c := range plan.Selection.Selected {
		key, err := randomID()
		if err != nil {
			return nil, err
		}
		start, err := randomID()
		if err != nil {
			return nil, err
		}
		r.Nodes[i] = launchNode{Candidate: c, Reserve: nodeagent.Reserve{IdempotencyKey: key, ExpectedNodeVersion: c.Capabilities.Availability.NodeVersion, CapabilityGeneration: c.Capabilities.Generation, Run: plan.Run, TTLSeconds: plan.TTLSeconds}, Start: nodeagent.StartCommand{IdempotencyKey: start}}
	}
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if err := path.Publish(func(stage statehome.Path) error { return stage.CompareAndSwap(launchFile, nil, b) }); err != nil {
		return nil, err
	}
	return OpenLaunch(path)
}
func OpenLaunch(path statehome.Path) (*Launch, error) {
	if path.Kind() != statehome.Principal {
		return nil, fmt.Errorf("controller-owned launch context required")
	}
	s := &Launch{path: path}
	_, _, err := s.load()
	return s, err
}
func (s *Launch) load() ([]byte, launchRecord, error) {
	b, err := s.path.ReadFileLimit(launchFile, 64<<20)
	if err != nil {
		return nil, launchRecord{}, err
	}
	var r launchRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	again, _ := json.Marshal(r)
	if !bytes.Equal(b, again) || r.Version != launchVersion || len(r.Nodes) != len(r.Plan.Selection.Selected) {
		return nil, r, fmt.Errorf("invalid launch journal")
	}
	if err := r.Plan.validate(); err != nil {
		return nil, r, err
	}
	switch r.Phase {
	case "new", "reserving", "preparing", "issuing", "activating", "starting", "running", "aborting", "aborted", "succeeded":
	default:
		return nil, r, fmt.Errorf("invalid launch phase")
	}
	for i, n := range r.Nodes {
		want, _ := json.Marshal(r.Plan.Selection.Selected[i])
		got, _ := json.Marshal(n.Candidate)
		if !bytes.Equal(want, got) || n.Reserve.Run != r.Plan.Run || !canonicalHex(n.Reserve.IdempotencyKey, 16) || !canonicalHex(n.Start.IdempotencyKey, 16) {
			return nil, r, fmt.Errorf("launch journal changed reservation identity")
		}
		if n.Lease != nil && !leaseMatches(r.Plan, n, *n.Lease) {
			return nil, r, fmt.Errorf("launch journal lease mismatch")
		}
		if !n.Touched && (n.Lease != nil || n.Fenced || n.Released) || n.Fenced && n.Lease == nil && !n.Released || n.Released && !n.Fenced || n.Prepared != nil && n.Manifest == nil || n.Grant != nil && n.Prepared == nil || n.Credential != nil && n.Grant == nil || n.Activated != nil && (n.Credential == nil || r.Transport == nil) || n.Started != nil && n.Activated == nil {
			return nil, r, fmt.Errorf("inconsistent launch effect history")
		}
		if n.Manifest != nil {
			if err := checkLaunchManifest(r.Plan, n, i, *n.Manifest); err != nil {
				return nil, r, err
			}
		}
	}
	return b, r, nil
}
func (s *Launch) save(old *[]byte, r launchRecord) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > 64<<20 {
		return fmt.Errorf("launch journal capacity exceeded")
	}
	if err := s.path.CompareAndSwap(launchFile, *old, b); err != nil {
		return err
	}
	*old = b
	return nil
}

func leaseMatches(p LaunchPlan, n launchNode, l nodeagent.Lease) bool {
	return l.Format == nodeagent.LeaseVersion && canonicalHex(l.ID, 16) && l.Node == n.Candidate.Capabilities.Node && l.Controller == p.Controller && l.Run == p.Run && l.CapabilityGeneration == n.Reserve.CapabilityGeneration && l.Version > 0
}
func checkLaunchManifest(p LaunchPlan, n launchNode, rank int, s nodejob.Signed) error {
	m := s.Manifest
	q, err := m.SigningRequest()
	if err != nil {
		return err
	}
	r := p.Selection.Requirements
	if n.Lease == nil || m.Lease != n.Lease.ID || m.Controller != p.Controller || m.Node != n.Candidate.Capabilities.Node || m.Membership.RunID != p.Run || m.Attempt != p.Attempt || m.Rank != rank || len(m.Members) != len(p.Selection.Selected) || m.BuildID != r.BuildID || m.DatasetID != r.DatasetID || m.DatasetSelector != r.DatasetSelector || m.Limits != r.Limits || s.Proof.Request != q {
		return fmt.Errorf("signed job differs from fixed recruitment plan")
	}
	for i, c := range p.Selection.Selected {
		if m.Members[i].Node != c.Capabilities.Node {
			return fmt.Errorf("signed job changed node assignment")
		}
	}
	return nil
}

func (p LaunchPorts) validate() error {
	if p.Reserve == nil || p.LeaseStatus == nil || p.AbortReservation == nil || p.Renew == nil || p.BuildJobs == nil || p.Prepare == nil || p.Grant == nil || p.Issue == nil || p.BuildTransport == nil || p.Activate == nil || p.Start == nil || p.Status == nil || p.Clock == nil {
		return fmt.Errorf("complete bounded launch ports required")
	}
	return nil
}
