// Package admission owns the recoverable cross-context admission saga. It
// neither mutates node leases nor owns coordinator run decisions or keys.
package admission

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
)

const version = "mixlab_learner_admission_v1"
const file = "admission.json"

// Specification is durable before reservation. Restart cannot substitute a
// different program, initial model, outer recipe or designated evaluator.
type Specification struct {
	Contract        run.Contract                 `json:"contract"`
	BaseModel       run.Blob                     `json:"base_model"`
	Outer           run.Outer                    `json:"outer"`
	Coordinator     run.Coordinator              `json:"coordinator"`
	ArtifactStoreID string                       `json:"artifact_store_id"`
	Endpoint        string                       `json:"endpoint"`
	Config          json.RawMessage              `json:"config"`
	Policy          coordinator.EvaluationPolicy `json:"evaluation_policy"`
	CreatedAt       int64                        `json:"created_at"`
}

func (v Specification) validate() error {
	for _, err := range []error{v.Contract.Validate(), v.BaseModel.Validate(), v.Outer.Validate(), v.Coordinator.Validate()} {
		if err != nil {
			return err
		}
	}
	b, err := nodejob.CanonicalConfig(v.Config)
	if err != nil || !bytes.Equal(b, v.Config) || nodejob.Hash(b) != v.Contract.ConfigHash {
		return fmt.Errorf("admission config differs from specification")
	}
	if v.BaseModel.Kind != "model_f32" || v.CreatedAt <= 0 || v.Endpoint == "" || v.ArtifactStoreID == "" {
		return fmt.Errorf("invalid admission specification")
	}
	return nil
}

// Effects must be bounded and idempotent by the supplied immutable request.
// AbortReservation must fence even an ambiguously successful reservation, stop
// any child and confirm credential cleanup and release before returning nil.
type Ports struct {
	Reserve          func(context.Context, recruitment.Candidate, nodeagent.Reserve) (nodeagent.Lease, error)
	AbortReservation func(context.Context, recruitment.Candidate, nodeagent.Reserve) error
	BuildPlan        func(context.Context, recruitment.LaunchPlan, Specification, []nodeagent.Lease) (run.Plan, error)
	Stage            func(context.Context, run.Plan, coordinator.EvaluationPolicy) error
	Prepare          func(context.Context, run.Plan, Specification, int) (run.Registration, error)
	Start            func(context.Context, run.Plan, run.Registration) error
	Register         func(context.Context, run.Registration) error
	Commit           func(context.Context, run.Commit) error
	// Abort atomically fences admission or returns the winning commit. It must
	// also fence a plan whose Stage request may still be in flight.
	Abort    func(context.Context, run.Plan, coordinator.EvaluationPolicy) (*run.Commit, error)
	Finalize func(context.Context, run.Plan, run.Commit, int) error
	Clock    func() int64
}
type node struct {
	Reserve      nodeagent.Reserve `json:"reserve"`
	Touched      bool              `json:"touched"`
	Lease        *nodeagent.Lease  `json:"lease,omitempty"`
	Registration *run.Registration `json:"registration,omitempty"`
	Started      bool              `json:"started"`
	Registered   bool              `json:"registered"`
	Finalized    bool              `json:"finalized"`
	Released     bool              `json:"released"`
}
type record struct {
	Format        string                 `json:"format"`
	Request       recruitment.LaunchPlan `json:"request"`
	Specification Specification          `json:"specification"`
	Deadline      int64                  `json:"deadline"`
	Nodes         []node                 `json:"nodes"`
	Plan          *run.Plan              `json:"plan,omitempty"`
	Staged        bool                   `json:"staged"`
	Candidate     *run.Commit            `json:"candidate,omitempty"`
	Committed     bool                   `json:"committed"`
	Aborting      bool                   `json:"aborting"`
	Fenced        bool                   `json:"fenced"`
	Done          bool                   `json:"done"`
}
type Saga struct{ path statehome.Path }

func Begin(path statehome.Path, p recruitment.LaunchPlan, spec Specification, deadline int64) (*Saga, error) {
	r := record{Format: version, Request: p, Specification: spec, Deadline: deadline, Nodes: make([]node, len(p.Selection.Selected))}
	for i, c := range p.Selection.Selected {
		h, _ := run.Digest(struct{ Run, Attempt, Node string }{p.Run, p.Attempt, c.Capabilities.Node})
		r.Nodes[i].Reserve = nodeagent.Reserve{IdempotencyKey: h[:32], ExpectedNodeVersion: c.Capabilities.Availability.NodeVersion, CapabilityGeneration: c.Capabilities.Generation, Run: p.Run, TTLSeconds: p.TTLSeconds}
	}
	if err := r.validate(); err != nil {
		return nil, err
	}
	if path.Kind() != statehome.Principal {
		return nil, fmt.Errorf("admission-owned principal context required")
	}
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if err = path.Publish(func(stage statehome.Path) error { return stage.CompareAndSwap(file, nil, b) }); err != nil {
		return nil, err
	}
	return Open(path)
}
func Open(path statehome.Path) (*Saga, error) {
	if path.Kind() != statehome.Principal {
		return nil, fmt.Errorf("admission-owned principal context required")
	}
	s := &Saga{path}
	_, _, err := s.load()
	return s, err
}
func (s *Saga) load() ([]byte, record, error) {
	b, err := s.path.ReadFileLimit(file, 8<<20)
	if err != nil {
		return nil, record{}, err
	}
	var r record
	if err = json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	want, _ := json.Marshal(r)
	if !bytes.Equal(b, want) {
		return nil, r, fmt.Errorf("noncanonical admission journal")
	}
	return b, r, r.validate()
}
func (s *Saga) save(old []byte, r record) error {
	if err := r.validate(); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > 8<<20 {
		return fmt.Errorf("admission journal capacity exceeded")
	}
	return s.path.CompareAndSwap(file, old, b)
}
func (r record) validate() error {
	p := r.Request
	if err := r.Specification.validate(); err != nil {
		return err
	}
	if r.Format != version || r.Deadline <= 0 || p.Run == "" || p.Controller == "" || p.Attempt == "" || len(r.Nodes) == 0 || len(r.Nodes) != len(p.Selection.Selected) || p.TTLSeconds < 60 || p.TTLSeconds > 3600 {
		return fmt.Errorf("invalid admission identity or cohort")
	}
	if err := p.Selection.Requirements.Validate(); err != nil {
		return err
	}
	if p.Selection.Requirements.Cluster != p.Cluster || p.Selection.Requirements.Count != len(r.Nodes) {
		return fmt.Errorf("selection differs from admission")
	}
	v := r.Specification
	if v.Coordinator.ClusterID != p.Cluster || v.Contract.BuildHash != p.Selection.Requirements.BuildID || v.Contract.DatasetHash != p.Selection.Requirements.DatasetID || r.Deadline <= v.CreatedAt || r.Deadline-v.CreatedAt > 3600 {
		return fmt.Errorf("specification differs from recruitment or deadline")
	}
	if r.Plan != nil {
		if err := r.Plan.Validate(); err != nil {
			return err
		}
		if r.Plan.Cohort.RunID != p.Run || r.Plan.Coordinator.ClusterID != p.Cluster || len(r.Plan.Slots) != len(r.Nodes) || r.Plan.ExpiresAt != r.Deadline {
			return fmt.Errorf("plan differs from admission")
		}
		if r.Plan.Contract != v.Contract || r.Plan.BaseModel != v.BaseModel || r.Plan.Outer != v.Outer || r.Plan.Coordinator != v.Coordinator || r.Plan.Endpoint != v.Endpoint || r.Plan.ArtifactStoreID != v.ArtifactStoreID || r.Plan.CreatedAt != v.CreatedAt {
			return fmt.Errorf("built plan changed immutable specification")
		}
	}
	if r.Staged && r.Plan == nil || r.Candidate != nil && !r.Staged || r.Committed && r.Candidate == nil || r.Committed && r.Fenced {
		return fmt.Errorf("invalid admission transition")
	}
	if r.Candidate != nil {
		if err := r.Candidate.ValidateFor(*r.Plan); err != nil {
			return err
		}
		for i, n := range r.Nodes {
			if n.Registration == nil || !n.Registered {
				return fmt.Errorf("commit candidate lacks completed registration")
			}
			a, _ := run.Digest(*n.Registration)
			b, _ := run.Digest(r.Candidate.Registrations[i])
			if a != b {
				return fmt.Errorf("commit candidate changed prepared registration")
			}
		}
	}
	seen := map[string]bool{}
	for i, n := range r.Nodes {
		c := p.Selection.Selected[i]
		if err := c.Capabilities.Validate(); err != nil {
			return err
		}
		if !slices.Contains(c.Capabilities.ManifestVersions, nodejob.LearnerVersion) || c.Capabilities.Probe.DeviceKind != "metal" || !c.Capabilities.Availability.Available {
			return fmt.Errorf("selected node does not advertise an available learner")
		}
		if seen[c.Capabilities.Node] {
			return fmt.Errorf("duplicate selected node")
		}
		seen[c.Capabilities.Node] = true
		h, _ := run.Digest(struct{ Run, Attempt, Node string }{p.Run, p.Attempt, c.Capabilities.Node})
		if n.Reserve != (nodeagent.Reserve{IdempotencyKey: h[:32], ExpectedNodeVersion: c.Capabilities.Availability.NodeVersion, CapabilityGeneration: c.Capabilities.Generation, Run: p.Run, TTLSeconds: p.TTLSeconds}) {
			return fmt.Errorf("reservation intent changed")
		}
		if !n.Touched && (n.Lease != nil || n.Released) || n.Started && n.Registration == nil || n.Registered && !n.Started || n.Finalized && !r.Committed {
			return fmt.Errorf("invalid admission effect order")
		}
		if n.Released && !r.Fenced {
			return fmt.Errorf("allocation released before commit fence")
		}
		if r.Done && ((r.Committed && !n.Finalized) || (!r.Committed && (!r.Fenced || n.Touched && !n.Released))) {
			return fmt.Errorf("incomplete terminal admission")
		}
		if n.Lease != nil {
			l := n.Lease
			if l.ID == "" || l.Controller != p.Controller || l.Run != p.Run || l.Node != c.Capabilities.Node || l.CapabilityGeneration != n.Reserve.CapabilityGeneration {
				return fmt.Errorf("reservation response substituted owner/node")
			}
		}
		if r.Plan != nil {
			if n.Lease == nil || r.Plan.Slots[i].Allocation.LeaseID != n.Lease.ID || r.Plan.Slots[i].Allocation.NodeID != n.Lease.Node {
				return fmt.Errorf("plan substituted allocation")
			}
			capabilityHash, _ := run.Digest(c.Capabilities)
			if r.Plan.Slots[i].CapabilityHash != capabilityHash {
				return fmt.Errorf("plan changed recruited capability evidence")
			}
		}
		if n.Registration != nil {
			if !r.Staged {
				return fmt.Errorf("prepared before stage")
			}
			if err := n.Registration.ValidateFor(*r.Plan); err != nil {
				return err
			}
			if n.Registration.Slot != r.Plan.Slots[i] {
				return fmt.Errorf("registration slot changed")
			}
		}
	}
	return nil
}

// Step performs at most one external effect. Its prior intent is durable, and
// replay repeats the exact effect after a lost response or ambiguous local CAS.
// The service, not the submitting terminal, repeatedly calls Step after restart.
func (s *Saga) Step(ctx context.Context, p Ports) (done bool, err error) {
	if p.Reserve == nil || p.AbortReservation == nil || p.BuildPlan == nil || p.Stage == nil || p.Prepare == nil || p.Start == nil || p.Register == nil || p.Commit == nil || p.Abort == nil || p.Finalize == nil || p.Clock == nil {
		return false, fmt.Errorf("complete admission ports required")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	err = s.path.WithProcessLock(ctx, "admission.lock", func() error {
		old, r, e := s.load()
		if e != nil {
			return e
		}
		if r.Done {
			done = true
			return nil
		}
		if !r.Committed && p.Clock() >= r.Deadline && !r.Aborting {
			r.Aborting = true
			return s.save(old, r)
		}
		if r.Aborting && !r.Committed {
			return s.compensate(ctx, p, old, r)
		}
		for i, n := range r.Nodes {
			if !n.Touched {
				r.Nodes[i].Touched = true
				return s.save(old, r)
			}
			if n.Lease == nil {
				l, e := p.Reserve(ctx, r.Request.Selection.Selected[i], n.Reserve)
				if e != nil {
					return e
				}
				r.Nodes[i].Lease = &l
				return s.save(old, r)
			}
		}
		if r.Plan == nil {
			leases := make([]nodeagent.Lease, len(r.Nodes))
			for i, n := range r.Nodes {
				leases[i] = *n.Lease
			}
			plan, e := p.BuildPlan(ctx, r.Request, r.Specification, leases)
			if e != nil {
				return e
			}
			r.Plan = &plan
			return s.save(old, r)
		}
		if !r.Staged {
			if e := p.Stage(ctx, *r.Plan, r.Specification.Policy); e != nil {
				return e
			}
			r.Staged = true
			return s.save(old, r)
		}
		for i, n := range r.Nodes {
			if n.Registration == nil {
				reg, e := p.Prepare(ctx, *r.Plan, r.Specification, i)
				if e != nil {
					return e
				}
				r.Nodes[i].Registration = &reg
				return s.save(old, r)
			}
			if !n.Started {
				if e := p.Start(ctx, *r.Plan, *n.Registration); e != nil {
					return e
				}
				r.Nodes[i].Started = true
				return s.save(old, r)
			}
			if !n.Registered {
				if e := p.Register(ctx, *n.Registration); e != nil {
					return e
				}
				r.Nodes[i].Registered = true
				return s.save(old, r)
			}
		}
		if r.Candidate == nil {
			plan := r.Plan
			c := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: plan.Cohort.RunID, PlanHash: plan.SHA256, Generation: plan.Cohort.Generation, CohortHash: plan.Cohort.CohortHash}, Coordinator: plan.Coordinator, CommittedAt: p.Clock()}
			for _, n := range r.Nodes {
				c.Registrations = append(c.Registrations, *n.Registration)
			}
			c.Binding.CommitHash, e = c.Digest()
			if e != nil {
				return e
			}
			r.Candidate = &c
			return s.save(old, r)
		}
		if !r.Committed {
			if e := p.Commit(ctx, *r.Candidate); e != nil {
				return e
			}
			r.Committed = true
			return s.save(old, r)
		}
		for i, n := range r.Nodes {
			if !n.Finalized {
				if e := p.Finalize(ctx, *r.Plan, *r.Candidate, i); e != nil {
					return e
				}
				r.Nodes[i].Finalized = true
				return s.save(old, r)
			}
		}
		r.Done = true
		done = true
		return s.save(old, r)
	})
	return done, err
}
func (s *Saga) Cancel(ctx context.Context) error {
	return s.path.WithProcessLock(ctx, "admission.lock", func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		if r.Committed || r.Done {
			return fmt.Errorf("committed admission belongs to coordinator")
		}
		r.Aborting = true
		return s.save(old, r)
	})
}
func (s *Saga) compensate(ctx context.Context, p Ports, old []byte, r record) error {
	if !r.Fenced {
		if r.Plan != nil {
			c, err := p.Abort(ctx, *r.Plan, r.Specification.Policy)
			if err != nil {
				return err
			}
			if c != nil {
				if err = c.ValidateFor(*r.Plan); err != nil {
					return err
				}
				if r.Candidate == nil || r.Candidate.Binding != c.Binding {
					return fmt.Errorf("coordinator committed a conflicting admission")
				}
				r.Candidate = c
				r.Staged = true
				r.Committed = true
				r.Aborting = false
				return s.save(old, r)
			}
		}
		r.Fenced = true
		return s.save(old, r)
	}
	for i, n := range r.Nodes {
		if n.Touched && !n.Released {
			if err := p.AbortReservation(ctx, r.Request.Selection.Selected[i], n.Reserve); err != nil {
				return err
			}
			r.Nodes[i].Released = true
			return s.save(old, r)
		}
	}
	r.Done = true
	return s.save(old, r)
}
