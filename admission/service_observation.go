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
}

func (i *Inbox) OwnService(ctx context.Context, serve func(context.Context) error) error {
	if serve == nil {
		return fmt.Errorf("service callback required")
	}
	return i.path.WithProcessLock(ctx, "service-owner.lock", func() error { return serve(ctx) })
}

func (i *Inbox) Observe(ctx context.Context, id string, at int64, failure error) error {
	if _, err := i.Read(ctx, id); err != nil {
		return err
	}
	v := Observation{At: at}
	if failure != nil {
		v.Error = failure.Error()
	}
	if at <= 0 || len(v.Error) > 4096 {
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
			if previous.At <= 0 || len(previous.Error) > 4096 {
				return fmt.Errorf("invalid persisted service observation")
			}
			// Keep live diagnostics at minute cadence unless the error changes.
			if previous.Error == v.Error && at >= previous.At && at-previous.At < 60 {
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
	if v.At <= 0 || len(v.Error) > 4096 {
		return v, fmt.Errorf("invalid persisted service observation")
	}
	return v, nil
}
