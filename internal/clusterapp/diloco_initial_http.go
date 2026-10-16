package clusterapp

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

const DiLoCoInitialModelRoute = "/v1/diloco/initial-model"

// Initial model uploads are controller-only, bounded, checksum-verified F32
// tensors. They convey no run authority and cannot modify an existing blob.
func diLoCoInitialHandler(policy *managedtls.Policy, store coordinator.Artifacts) http.Handler {
	return policy.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		actor, ok := managedtls.Principal(r.Context())
		if !ok || actor.Role != trust.Controller {
			http.Error(w, "controller required", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodPut || r.URL.Path != DiLoCoInitialModelRoute || r.URL.RawQuery != "" || r.URL.RawPath != "" {
			http.NotFound(w, r)
			return
		}
		if r.ContentLength <= 0 || uint64(r.ContentLength) > artifact.MaxBytes || len(r.TransferEncoding) != 0 {
			http.Error(w, "bounded model content length required", http.StatusBadRequest)
			return
		}
		expected := r.Header.Get("X-Mixlab-SHA256")
		ref := artifact.Ref{SHA256: expected, Bytes: uint64(r.ContentLength)}
		if ref.Validate() != nil {
			http.Error(w, "model digest required", http.StatusBadRequest)
			return
		}
		deadline := time.Now().Add(time.Minute)
		control := http.NewResponseController(w)
		if control.SetReadDeadline(deadline) != nil || control.SetWriteDeadline(deadline) != nil {
			http.Error(w, "bounded transfer required", http.StatusServiceUnavailable)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), time.Minute)
		defer cancel()
		f, err := os.CreateTemp("", "mixlab-initial-model-*")
		if err != nil {
			http.Error(w, "model staging unavailable", http.StatusServiceUnavailable)
			return
		}
		defer func() { _ = f.Close() }()
		if err := os.Remove(f.Name()); err != nil {
			http.Error(w, "private staging unavailable", http.StatusServiceUnavailable)
			return
		}
		if err := artifact.Copy(ctx, f, r.Body, ref); err != nil {
			http.Error(w, "model checksum/size mismatch", http.StatusBadRequest)
			return
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			http.Error(w, "model staging failed", http.StatusServiceUnavailable)
			return
		}
		values, err := tensors.Read(f, ref.Bytes)
		if err != nil {
			http.Error(w, "ordered finite F32 model tensors required", http.StatusBadRequest)
			return
		}
		if r.TLS == nil {
			http.Error(w, "TLS required", http.StatusForbidden)
			return
		}
		current, err := policy.Authenticate(*r.TLS)
		if err != nil || current.Principal != actor.Principal || current.Role != actor.Role {
			http.Error(w, "current controller required", http.StatusForbidden)
			return
		}
		blob, err := store.Put(ctx, "model_f32", values)
		if err != nil {
			http.Error(w, "model persistence failed", http.StatusConflict)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		writeJSON(w, blob)
	}))
}

func validateDiLoCoInfo(i DiLoCoInfo, coordinator string) error {
	if i.Version != "mixlab_diloco_service_v1" || i.Coordinator != coordinator || !nodeRouteID(i.Cluster) || !nodeRouteID(i.Controller) {
		return fmt.Errorf("coordinator service identity mismatch")
	}
	return nil
}
