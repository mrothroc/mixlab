package admission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// Observation is bounded operational evidence, not admission or run authority.
// Replaying work always reads the saga/coordinator journals, never this record.
type Observation struct {
	At    int64  `json:"at"`
	Error string `json:"error,omitempty"`
	// Retry timing is scheduling evidence only, never proof of compensation.
	RetryAt       int64  `json:"retry_at,omitempty"`
	RetryFailures uint32 `json:"retry_failures,omitempty"`
	RetryPhase    string `json:"retry_phase,omitempty"`
	EventAt       int64  `json:"event_at,omitempty"`
	FailureNode   string `json:"failure_node,omitempty"`
	FailureClass  string `json:"failure_class,omitempty"`
}

func (v Observation) valid() bool {
	if v.At <= 0 || len(v.Error) > 4096 || v.RetryAt < 0 || v.RetryFailures > 16 ||
		v.EventAt < 0 || len(v.FailureNode) > 64 || len(v.FailureClass) > 64 {
		return false
	}
	if v.RetryAt == 0 {
		return v.RetryFailures == 0 && v.RetryPhase == ""
	}
	return v.Error != "" && v.RetryFailures > 0 && v.RetryAt > v.At && v.RetryAt-v.At <= 60 &&
		(v.RetryPhase == "admitting" || v.RetryPhase == "aborting" || v.RetryPhase == "committed")
}

func (i *Inbox) OwnService(ctx context.Context, serve func(context.Context) error) error {
	if serve == nil {
		return fmt.Errorf("service callback required")
	}
	return i.path.WithProcessLock(ctx, "service-owner.lock", func() error { return serve(ctx) })
}

func (i *Inbox) Observe(ctx context.Context, id string, at int64, failure error) error {
	v := Observation{At: at}
	if failure != nil {
		v.Error = failure.Error()
	}
	return i.ObserveState(ctx, id, v)
}

// ObserveState persists bounded scheduling and diagnostic data independently
// of the admission journal. A missing old record remains backward compatible.
func (i *Inbox) ObserveState(ctx context.Context, id string, v Observation) error {
	if _, err := i.Read(ctx, id); err != nil {
		return err
	}
	if !v.valid() {
		return fmt.Errorf("invalid service observation")
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return i.path.WithProcessLock(ctx, "inbox.lock", func() error {
		old, err := i.path.ReadFileLimit(id+".observation", 8192)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err == nil {
			var previous Observation
			if err := json.Unmarshal(old, &previous); err != nil {
				return err
			}
			if !previous.valid() {
				return fmt.Errorf("invalid persisted service observation")
			}
			// Keep live diagnostics at minute cadence unless the error changes.
			at := previous.At
			previous.At = v.At
			if previous == v && v.At >= at && v.At-at < 60 {
				return nil
			}
		}
		return i.path.CompareAndSwap(id+".observation", old, b)
	})
}

func (i *Inbox) Observation(ctx context.Context, id string) (Observation, error) {
	var v Observation
	if _, err := i.Read(ctx, id); err != nil {
		return v, err
	}
	b, err := i.path.ReadFileLimit(id+".observation", 8192)
	if errors.Is(err, os.ErrNotExist) {
		return v, nil
	}
	if err != nil {
		return v, err
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return v, err
	}
	if !v.valid() {
		return v, fmt.Errorf("invalid persisted service observation")
	}
	return v, nil
}
