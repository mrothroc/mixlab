package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

const DiLoCoSubmissionRoute = "/v1/diloco/submissions"
const DiLoCoInfoRoute = "/v1/diloco/info"

type DiLoCoInfo struct {
	Version     string `json:"version"`
	Cluster     string `json:"cluster"`
	Coordinator string `json:"coordinator"`
	Controller  string `json:"admission_controller"`
	Endpoint    string `json:"endpoint"`
}

type DiLoCoSubmissionStatus struct {
	RunID        string            `json:"run_id"`
	Admission    string            `json:"admission"`
	RunPhase     coordinator.Phase `json:"run_phase,omitempty"`
	Round        uint64            `json:"round"`
	Failure      string            `json:"failure,omitempty"`
	ServiceError bool              `json:"service_error"`
	ObservedAt   int64             `json:"observed_at,omitempty"`
}

// DiLoCoSubmissionAPI owns transport authentication only. Admission persists
// the immutable request; a separate service loop owns all remote side effects.
type DiLoCoSubmissionAPI struct {
	Inbox                                          *admission.Inbox
	Repository                                     coordinator.Repository
	Policy                                         *managedtls.Policy
	ControllerID, CoordinatorID, Cluster, Endpoint string
	Clock                                          func() time.Time
	// Verify is read-only preflight (e.g. initial artifact existence/layout).
	Verify func(context.Context, admission.Submission) error
}

func (a DiLoCoSubmissionAPI) Handler() (http.Handler, error) {
	if a.Inbox == nil || a.Repository == nil || a.Policy == nil || a.Clock == nil || a.Verify == nil || !nodeRouteID(a.ControllerID) || !nodeRouteID(a.CoordinatorID) || !nodeRouteID(a.Cluster) || a.Endpoint == "" {
		return nil, fmt.Errorf("complete durable submission service required")
	}
	return a.Policy.AuthenticateHTTP(http.HandlerFunc(a.serve)), nil
}

func (a DiLoCoSubmissionAPI) serve(w http.ResponseWriter, r *http.Request) {
	actor, ok := managedtls.Principal(r.Context())
	if !ok || actor.Role != trust.Controller || actor.Cluster != a.Cluster {
		http.Error(w, "controller required", http.StatusForbidden)
		return
	}
	if r.URL.RawQuery != "" || r.URL.RawPath != "" {
		http.NotFound(w, r)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	r = r.WithContext(ctx)
	current := func() bool {
		if r.TLS == nil || ctx.Err() != nil {
			return false
		}
		p, err := a.Policy.Authenticate(*r.TLS)
		return err == nil && p.Principal == actor.Principal && p.Role == actor.Role && p.Cluster == actor.Cluster
	}
	if r.Method == http.MethodGet && r.URL.Path == DiLoCoInfoRoute {
		if r.ContentLength != 0 || len(r.TransferEncoding) != 0 || !current() {
			http.Error(w, "empty authenticated request required", http.StatusBadRequest)
			return
		}
		writeJSON(w, DiLoCoInfo{"mixlab_diloco_service_v1", a.Cluster, a.CoordinatorID, a.ControllerID, a.Endpoint})
		return
	}
	if r.Method == http.MethodPost && r.URL.Path == DiLoCoSubmissionRoute {
		var q admission.Submission
		if err := readJSON(w, r, &q, 8<<20); err != nil {
			http.Error(w, "invalid bounded submission", http.StatusBadRequest)
			return
		}
		if !current() {
			http.Error(w, "current trust required", http.StatusForbidden)
			return
		}
		if q.Owner != actor.Principal || q.Launch.Controller != a.ControllerID || q.Launch.Cluster != a.Cluster || q.Specification.Coordinator.PrincipalID != a.CoordinatorID || q.Specification.Endpoint != a.Endpoint || q.Specification.CreatedAt > a.Clock().Unix() {
			http.Error(w, "submission owner, service or deadline mismatch", http.StatusForbidden)
			return
		}
		if err := q.Validate(); err != nil {
			http.Error(w, "invalid submission contract", http.StatusBadRequest)
			return
		}
		// An exact retry recovers a lost acceptance acknowledgement even after
		// admission expires. It cannot create a new run or change its owner.
		old, readErr := a.Inbox.Read(ctx, q.Launch.Run)
		if readErr == nil {
			before, _ := json.Marshal(old)
			after, _ := json.Marshal(q)
			if !bytes.Equal(before, after) || !current() {
				http.Error(w, "accepted run conflicts with submission", http.StatusConflict)
				return
			}
			writeJSON(w, DiLoCoSubmissionStatus{RunID: q.Launch.Run, Admission: "ACCEPTED"})
			return
		}
		if !errors.Is(readErr, os.ErrNotExist) || q.Deadline <= a.Clock().Unix() {
			http.Error(w, "new submission requires live deadline and intact admission state", http.StatusConflict)
			return
		}
		if err := a.Verify(ctx, q); err != nil {
			http.Error(w, "submission preflight failed; inspect service diagnostics", http.StatusConflict)
			return
		}
		if !current() {
			http.Error(w, "current trust required", http.StatusForbidden)
			return
		}
		if err := a.Inbox.Accept(ctx, q); err != nil {
			http.Error(w, "durable acceptance failed", http.StatusConflict)
			return
		}
		writeJSON(w, DiLoCoSubmissionStatus{RunID: q.Launch.Run, Admission: "ACCEPTED"})
		return
	}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, DiLoCoSubmissionRoute+"/"), "/")
	if !strings.HasPrefix(r.URL.Path, DiLoCoSubmissionRoute+"/") || len(parts) > 2 || !nodeRouteID(parts[0]) {
		http.NotFound(w, r)
		return
	}
	q, err := a.Inbox.Read(ctx, parts[0])
	if err != nil || q.Owner != actor.Principal {
		http.Error(w, "submission not available", http.StatusNotFound)
		return
	}
	if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
		http.Error(w, "body not allowed", http.StatusBadRequest)
		return
	}
	if !current() {
		http.Error(w, "current trust required", http.StatusForbidden)
		return
	}
	saga, err := a.Inbox.Resume(ctx, q.Launch.Run)
	if err != nil {
		http.Error(w, "admission journal unavailable", http.StatusServiceUnavailable)
		return
	}
	if len(parts) == 2 && parts[1] == "cancel" && r.Method == http.MethodPost {
		if err := saga.Cancel(ctx); err != nil {
			http.Error(w, "cancel rejected; committed runs belong to coordinator", http.StatusConflict)
			return
		}
	} else if len(parts) != 1 || r.Method != http.MethodGet {
		http.NotFound(w, r)
		return
	}
	v, err := saga.Snapshot(ctx)
	if err != nil {
		http.Error(w, "admission journal unavailable", http.StatusServiceUnavailable)
		return
	}
	out := DiLoCoSubmissionStatus{RunID: q.Launch.Run, Admission: "ADMITTING"}
	switch {
	case v.Committed:
		out.Admission = "COMMITTED"
	case v.Done:
		out.Admission = "ABORTED"
	case v.Aborting:
		out.Admission = "ABORTING"
	}
	if v.Committed {
		s, err := a.Repository.Load(ctx, q.Launch.Run)
		if err != nil {
			http.Error(w, "committed run journal unavailable", http.StatusServiceUnavailable)
			return
		}
		out.RunPhase, out.Round, out.Failure = s.Phase, s.Round, s.Failure
	}
	o, err := a.Inbox.Observation(ctx, q.Launch.Run)
	if err != nil {
		http.Error(w, "service observation unavailable", http.StatusServiceUnavailable)
		return
	}
	out.ServiceError, out.ObservedAt = o.Error != "", o.At
	if !current() {
		http.Error(w, "current trust required", http.StatusForbidden)
		return
	}
	writeJSON(w, out)
}
