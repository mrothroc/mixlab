package artifactapi

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

const ReadRoute = "/v1/diloco/artifacts/read"
const CreateRoute = "/v1/diloco/artifacts/create"
const MaxEnvelope = 1 << 20

type Request struct {
	Grant authenticated.Grant `json:"grant"`
	Use   run.GrantUse        `json:"use"`
}

func (q Request) Validate() error {
	if err := q.Grant.Validate(); err != nil {
		return err
	}
	return q.Use.Validate()
}

type CurrentTrust interface {
	Current(context.Context, time.Time) (trust.VerifiedSnapshot, error)
}
type Gateway struct {
	Policy      *managedtls.Policy
	Anchor      trust.Anchor
	Trust       CurrentTrust
	Coordinator run.Coordinator
	Repository  coordinator.Repository
	Store       *artifactlocal.Store
	Ledger      *Ledger
	Clock       func() time.Time
	// TransferTimeout bounds network/disk work; grants may expire sooner.
	TransferTimeout time.Duration
}

func (g Gateway) Handler() (http.Handler, error) {
	if g.Policy == nil || g.Trust == nil || g.Repository == nil || g.Store == nil || g.Ledger == nil || g.Clock == nil || g.TransferTimeout <= 0 || g.TransferTimeout > time.Minute {
		return nil, fmt.Errorf("complete bounded artifact gateway required")
	}
	if err := g.Coordinator.Validate(); err != nil {
		return nil, err
	}
	if g.Anchor.Cluster() != g.Coordinator.ClusterID {
		return nil, fmt.Errorf("coordinator trust anchor mismatch")
	}
	return g.Policy.AuthenticateHTTP(http.HandlerFunc(g.serve)), nil
}

// Wire framing is a 4-byte big-endian envelope length, canonical JSON envelope,
// then exactly the declared artifact bytes for create (nothing for read).
func WriteRequest(w io.Writer, q Request) error {
	if err := q.Validate(); err != nil {
		return err
	}
	b, err := json.Marshal(q)
	if err != nil {
		return err
	}
	if len(b) > MaxEnvelope {
		return fmt.Errorf("artifact envelope too large")
	}
	if err = binary.Write(w, binary.BigEndian, uint32(len(b))); err != nil {
		return err
	}
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		return io.ErrShortWrite
	}
	return err
}
func readRequest(r io.Reader) (Request, error) {
	var q Request
	var size uint32
	if err := binary.Read(r, binary.BigEndian, &size); err != nil {
		return q, err
	}
	if size == 0 || size > MaxEnvelope {
		return q, fmt.Errorf("invalid artifact envelope size")
	}
	b := make([]byte, size)
	if _, err := io.ReadFull(r, b); err != nil {
		return q, err
	}
	return q, authenticated.Decode(b, &q)
}
func (g Gateway) serve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.RawQuery != "" || (r.URL.Path != ReadRoute && r.URL.Path != CreateRoute) {
		http.NotFound(w, r)
		return
	}
	peer, ok := managedtls.Principal(r.Context())
	if !ok {
		http.Error(w, "managed principal required", http.StatusUnauthorized)
		return
	}
	control := http.NewResponseController(w)
	deadline := time.Now().Add(g.TransferTimeout)
	if control.SetReadDeadline(deadline) != nil || control.SetWriteDeadline(deadline) != nil {
		http.Error(w, "bounded transport required", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), g.TransferTimeout)
	defer cancel()
	q, err := readRequest(r.Body)
	if err != nil {
		http.Error(w, "invalid artifact request", http.StatusBadRequest)
		return
	}
	proof, err := g.authorize(ctx, peer, q, r.URL.Path)
	if err != nil {
		http.Error(w, "artifact access rejected", http.StatusForbidden)
		return
	}
	remaining := time.Unix(q.Grant.Body.ExpiresAt, 0).Sub(g.Clock())
	if remaining <= 0 {
		http.Error(w, "artifact grant expired", http.StatusForbidden)
		return
	}
	deadline = time.Now().Add(min(remaining, g.TransferTimeout))
	_ = control.SetReadDeadline(deadline)
	_ = control.SetWriteDeadline(deadline)
	ctx, stop := context.WithTimeout(ctx, remaining)
	defer stop()
	wrote := false
	err = g.Ledger.WithUse(ctx, q.Grant.Body, q.Use, proof, func() error {
		// Repeat authorization after waiting for another retry/use to complete.
		peer, err := g.Policy.Authenticate(*r.TLS)
		if err != nil {
			return err
		}
		if _, err := g.authorize(ctx, peer, q, r.URL.Path); err != nil {
			return err
		}
		if r.URL.Path == CreateRoute {
			return g.Store.PutVerified(ctx, q.Grant.Body.Artifact.Ref, r.Body)
		}
		var extra [1]byte
		if _, err := io.ReadFull(r.Body, extra[:]); err != io.EOF {
			return fmt.Errorf("unexpected read payload")
		}
		// Verify before writing response headers, then reverify while streaming.
		f, err := g.Store.OpenVerified(ctx, q.Grant.Body.Artifact.Ref)
		if err != nil {
			return err
		}
		_ = f.Close()
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", fmt.Sprint(q.Grant.Body.Artifact.Ref.Bytes))
		wrote = true
		return g.Store.Copy(ctx, q.Grant.Body.Artifact.Ref, w)
	})
	if err != nil {
		if !wrote {
			http.Error(w, "artifact transfer failed", http.StatusConflict)
		}
		return
	}
	if !wrote {
		w.WriteHeader(http.StatusNoContent)
	}
}
func (g Gateway) authorize(ctx context.Context, peer trust.AuthenticatedPrincipal, q Request, route string) (trust.AcceptedProof, error) {
	var zero trust.AcceptedProof
	if err := q.Validate(); err != nil {
		return zero, err
	}
	b := q.Grant.Body
	now := g.Clock()
	if peer.Role != trust.Node || peer.Cluster != b.Coordinator.ClusterID || peer.Principal != b.PrincipalID || b.Coordinator != g.Coordinator || now.Unix() < b.IssuedAt || now.Unix() >= b.ExpiresAt || q.Use.GrantHash != b.SHA256 || q.Use.GrantID != b.ID || q.Use.Operation != b.Operation || q.Use.UploadID != b.UploadID {
		return zero, fmt.Errorf("grant principal/operation binding mismatch")
	}
	// Resumable chunk uploads are not advertised by this full-body endpoint.
	if (route == ReadRoute && b.Operation != "read") || (route == CreateRoute && b.Operation != "create") {
		return zero, fmt.Errorf("unsupported artifact operation")
	}
	v, err := g.Trust.Current(ctx, now)
	if err != nil {
		return zero, err
	}
	proof, err := trust.VerifyProof(g.Anchor, v, q.Grant.Proof, q.Grant.Proof.Request, now)
	if err != nil {
		return zero, err
	}
	s, err := g.Repository.Load(ctx, b.Binding.RunID)
	if err != nil {
		return zero, err
	}
	if err = s.Validate(); err != nil {
		return zero, err
	}
	if s.Commit == nil || b.Binding != s.Commit.Binding || s.Plan.Coordinator != g.Coordinator {
		return zero, fmt.Errorf("grant not committed to this run")
	}
	found := false
	for _, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == b.WorkerID && slot.Member.ParticipantID == b.ParticipantID && slot.Allocation.NodeID == peer.Principal {
			found = true
		}
	}
	if !found {
		return zero, fmt.Errorf("grant worker/participant mismatch")
	}
	if s.Phase == coordinator.Evaluating && b.Operation == "read" && s.Evaluation != nil && b.AssignmentID == s.Evaluation.ID && b.WorkerID == s.Evaluation.WorkerID && b.Artifact == s.Evaluation.Model && now.Unix() >= s.Evaluation.IssuedAt && now.Unix() < s.Evaluation.ExpiresAt && b.ExpiresAt <= s.Evaluation.ExpiresAt {
		return proof, nil
	}
	if s.Phase != coordinator.Open {
		return zero, fmt.Errorf("no open artifact assignment")
	}
	for _, a := range s.Assignments {
		if a.ID == b.AssignmentID && a.WorkerID == b.WorkerID && now.Unix() >= a.IssuedAt && now.Unix() < a.ExpiresAt && b.ExpiresAt <= a.ExpiresAt {
			if b.Operation == "read" && b.Artifact != a.BaseModel && b.Artifact != a.OuterState {
				return zero, fmt.Errorf("unassigned read artifact")
			}
			return proof, nil
		}
	}
	return zero, fmt.Errorf("stale artifact assignment")
}
