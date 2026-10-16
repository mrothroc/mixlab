//go:build darwin || linux

package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerjob"
)

// Relay handles one opaque request and returns its reply. The owner validates
// payloads and represents authoritative application errors in the reply; a Go
// error fails the local session with an uncertain delivery outcome. Callbacks
// must honor ctx and retain no envelope storage after returning. Hosting invokes
// at most one callback at a time and never interprets round or tensor payloads.
// The caller scopes callback resources to one Run and closes them afterward.
type Relay func(context.Context, wc.Envelope) (wc.Envelope, error)

// Freeze owner-defined schemas without interpreting learner round/tensor data.
func (p *LaunchPlan) freezeAssignment() (wc.Binding, wc.Envelope, int, error) {
	var binding wc.Binding
	var envelope wc.Envelope
	if p.Learner == nil {
		if p.Relay != nil || p.RelayTimeout != 0 || p.LearnerPrivateDirectory.Dir() != "" {
			return binding, envelope, 0, fmt.Errorf("learner session/storage without learner assignment")
		}
		b, err := p.Assignment.Binding()
		if err != nil {
			return binding, envelope, 0, err
		}
		blob, err := json.Marshal(p.Assignment)
		if err != nil {
			return binding, envelope, 0, err
		}
		p.Assignment, err = workerjob.Decode(blob, b)
		if err != nil {
			return binding, envelope, 0, err
		}
		envelope, err = workerjob.Envelope(b, 1, wc.KindAssignment, p.Assignment)
		return b, envelope, p.Assignment.RuntimeSeconds, err
	}
	if !reflect.ValueOf(p.Assignment).IsZero() || p.Relay == nil || p.Learner.Cohort == nil {
		return binding, envelope, 0, fmt.Errorf("learner launch requires exclusive schema, cohort and session callback")
	}
	if p.RelayTimeout < 0 || p.RelayTimeout > 10*time.Minute {
		return binding, envelope, 0, fmt.Errorf("invalid relay timeout")
	}
	if p.RelayTimeout == 0 {
		p.RelayTimeout = time.Minute
	}
	b, err := p.Learner.Binding()
	if err != nil {
		return binding, envelope, 0, err
	}
	blob, err := json.Marshal(p.Learner)
	if err != nil {
		return binding, envelope, 0, err
	}
	a, err := workerjob.DecodeLearner(blob, b)
	if err != nil {
		return binding, envelope, 0, err
	}
	p.Learner = &a
	envelope = wc.Envelope{Version: wc.Version, JobID: b.JobID, AttemptID: b.AttemptID, Sequence: 1,
		CorrelationID: b.AttemptID, Kind: wc.KindAssignment, PayloadKind: workerjob.LearnerVersion, PayloadVersion: 1, Payload: blob}
	return b, envelope, a.RuntimeSeconds, envelope.Validate()
}

func (p LaunchPlan) prepareLearnerStorage() error {
	if p.Learner == nil {
		return nil
	}
	if int64(p.Learner.PrivateMaxBytes) > p.Limits.DiskBytes-p.Limits.LogBytes {
		return fmt.Errorf("learner private budget exceeds approved disk allowance")
	}
	private := p.LearnerPrivateDirectory
	if private.Kind() != statehome.Worker || private.Dir() == "" || pathsOverlap(private.Dir(), p.Directory.Dir()) {
		return fmt.Errorf("learner private storage requires a separate worker-owned directory")
	}
	// Allocation owns creation. Launch must never replace missing published
	// storage, even before the first private hash has been acknowledged.
	return private.Validate()
}

func pathsOverlap(a, b string) bool {
	within := func(parent, child string) bool {
		rel, err := filepath.Rel(parent, child)
		return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	}
	return within(a, b) || within(b, a)
}

// nextSessionEnvelope drains already-received events even after peer EOF. This
// preserves terminal outcomes written immediately before normal process exit.
func nextSessionEnvelope(ctx context.Context, session *local.Dispatcher) (wc.Envelope, error) {
	events, controls := session.Events(), session.Controls()
	for events != nil || controls != nil {
		select {
		case e, ok := <-controls:
			if ok {
				return e, nil
			}
			controls = nil
		default:
		}
		select {
		case e, ok := <-events:
			if ok {
				return e, nil
			}
			events = nil
		case e, ok := <-controls:
			if ok {
				return e, nil
			}
			controls = nil
		case <-ctx.Done():
			return wc.Envelope{}, ctx.Err()
		}
	}
	return wc.Envelope{}, session.Err()
}

type relayWork struct {
	envelope wc.Envelope
	deadline time.Time
}

func serveRelay(ctx context.Context, session *local.Dispatcher, callback Relay, work <-chan relayWork) error {
	for {
		var item relayWork
		select {
		case <-ctx.Done():
			return ctx.Err()
		case item = <-work:
		}
		requestCtx, cancel := context.WithDeadline(ctx, item.deadline)
		if err := requestCtx.Err(); err != nil {
			cancel()
			return err
		}
		// One bounded active callback. A callback that violates the cancellation
		// contract cannot prevent process cleanup or launch additional handlers.
		type response struct {
			envelope wc.Envelope
			err      error
		}
		result := make(chan response, 1)
		go func() { e, err := callback(requestCtx, item.envelope); result <- response{e, err} }()
		var err error
		select {
		case reply := <-result:
			err = reply.err
			if err == nil {
				err = session.Reply(requestCtx, item.envelope, reply.envelope)
			}
		case <-requestCtx.Done():
			err = requestCtx.Err()
		}
		cancel()
		if err != nil {
			return err
		}
	}
}

func monitorLearner(ctx, start context.Context, session *local.Dispatcher, r Result, p LaunchPlan) (Result, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	work := make(chan relayWork, 16)
	errs := make(chan error, 1)
	go func() { errs <- serveRelay(ctx, session, p.Relay, work) }()
	defer func() { cancel(); <-errs }()
	events, controls := session.Events(), session.Controls()
	for events != nil || controls != nil {
		var e wc.Envelope
		var ok bool
		var startup <-chan struct{}
		if !r.Ready {
			startup = start.Done()
		}
		select {
		case e, ok = <-controls:
			if !ok {
				controls = nil
				continue
			}
		case e, ok = <-events:
			if !ok {
				events = nil
				continue
			}
		case err := <-errs:
			// Put the single terminal result back for the deferred join.
			errs <- err
			return r, err
		case <-ctx.Done():
			return r, ctx.Err()
		case <-startup:
			return r, start.Err()
		}
		switch e.Kind {
		case wc.KindReadiness, wc.KindHeartbeat, wc.KindProgress, wc.KindTerminalOutcome, wc.KindCancellation:
			if err := workerjob.CheckPayload(e, e.Kind); err != nil {
				return r, err
			}
			var event workerjob.Event
			if err := workerjob.DecodePayload(e.Payload, &event); err != nil {
				return r, err
			}
			switch e.Kind {
			case wc.KindReadiness:
				if r.Ready {
					return r, fmt.Errorf("duplicate worker readiness")
				}
				r.Ready = true
			case wc.KindProgress:
				if !r.Ready {
					return r, fmt.Errorf("worker progress before readiness")
				}
				r.Progress = event
			case wc.KindCancellation:
				return r, context.Canceled
			case wc.KindTerminalOutcome:
				r.Terminal = event
				if event.Error != "" {
					return r, errors.New(event.Error)
				}
				if !r.Ready {
					return r, fmt.Errorf("worker completed before readiness")
				}
				return r, nil
			}
		case wc.KindAssignment:
			return r, fmt.Errorf("unexpected worker assignment")
		default:
			select {
			case work <- relayWork{e, time.Now().Add(p.RelayTimeout)}:
			default:
				return r, local.ErrBackpressure
			}
		}
	}
	return r, session.Err()
}
