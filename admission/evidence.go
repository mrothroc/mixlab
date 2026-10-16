package admission

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust/workload"
)

// Public signed envelopes are admission evidence, not private credentials.
// Persist them before sending so trust refresh/restart cannot change an
// ambiguously successful request's identity. Keys remain with their owners.
func (i *Inbox) Manifest(ctx context.Context, run, job string, build func() (nodejob.SignedLearner, error)) (out nodejob.SignedLearner, err error) {
	err = i.evidence(ctx, run, job, "manifest", &out, func() (any, error) {
		return build()
	}, func() error {
		if err := out.Manifest.Validate(); err != nil {
			return err
		}
		if out.Manifest.Plan.Cohort.RunID != run || out.Manifest.Slot.Allocation.JobID != job {
			return fmt.Errorf("manifest evidence binding mismatch")
		}
		q, err := out.Manifest.SigningRequest()
		if err != nil {
			return err
		}
		if out.Proof.Request != q {
			return fmt.Errorf("manifest proof differs from intent")
		}
		return nil
	})
	return
}

func (i *Inbox) WorkloadGrant(ctx context.Context, run, job string, request workload.SignedRequest, build func() (workload.Grant, error)) (out workload.Grant, err error) {
	return i.workloadGrant(ctx, run, job, "workload-grant", request, build)
}

// Retry grants have separate immutable evidence for each operational attempt.
func (i *Inbox) RecoveryWorkloadGrant(ctx context.Context, run, job string, request workload.SignedRequest, build func() (workload.Grant, error)) (workload.Grant, error) {
	s := request.Request.Scope
	if err := s.Validate(); err != nil {
		return workload.Grant{}, err
	}
	if s.Profile != workload.LearnerVersion || s.Controller != s.Coordinator {
		return workload.Grant{}, fmt.Errorf("coordinator recovery scope required")
	}
	return i.workloadGrant(ctx, run, job, "recovery-"+s.Attempt+"-grant", request, build)
}

func (i *Inbox) workloadGrant(ctx context.Context, run, job, kind string, request workload.SignedRequest, build func() (workload.Grant, error)) (out workload.Grant, err error) {
	err = i.evidence(ctx, run, job, kind, &out, func() (any, error) { return build() }, func() error {
		a, _ := json.Marshal(out.Request)
		b, _ := json.Marshal(request)
		if !bytes.Equal(a, b) || request.Request.Scope.Run != run || request.Request.Scope.Job != job {
			return fmt.Errorf("workload grant changed request")
		}
		q, err := request.GrantRequest()
		if err != nil {
			return err
		}
		if out.Proof.Request != q {
			return fmt.Errorf("workload grant proof differs from intent")
		}
		return nil
	})
	return
}

// Issuance itself is idempotent at the authority. Replaying the persisted exact
// grant recovers a lost issuance response; no private key enters this inbox.
func (i *Inbox) evidence(ctx context.Context, run, job, kind string, out any, build func() (any, error), validate func() error) error {
	if _, err := i.Read(ctx, run); err != nil {
		return err
	}
	if len(job) != 32 {
		return fmt.Errorf("canonical evidence job ID required")
	}
	for _, c := range job {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return fmt.Errorf("canonical evidence job ID required")
		}
	}
	name := run + "-" + job + "-" + kind
	return i.path.WithProcessLock(ctx, name+".lock", func() error {
		b, err := i.path.ReadFileLimit(name, 2<<20)
		missing := errors.Is(err, os.ErrNotExist)
		if err != nil && !missing {
			return err
		}
		if missing {
			v, err := build()
			if err != nil {
				return err
			}
			b, err = json.Marshal(v)
			if err != nil {
				return err
			}
		}
		if len(b) > 2<<20 {
			return fmt.Errorf("admission evidence exceeds bound")
		}
		if err := json.Unmarshal(b, out); err != nil {
			return err
		}
		canonical, err := json.Marshal(out)
		if err != nil || !bytes.Equal(b, canonical) {
			return fmt.Errorf("noncanonical admission evidence")
		}
		if err := validate(); err != nil {
			return err
		}
		if missing {
			return i.path.CompareAndSwap(name, nil, b)
		}
		return ctx.Err()
	})
}
