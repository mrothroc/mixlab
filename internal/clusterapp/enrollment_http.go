package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
)

type CurrentTrust func(context.Context, time.Time) (trust.VerifiedSnapshot, error)

// EnrollmentHandler exposes bootstrap operations only. Administrator decisions,
// private handles, revocation, renewal, and workload APIs are deliberately absent.
func EnrollmentHandler(service *enrollment.Service, window enrollment.Window, channel *enrollmenttls.Channel, current CurrentTrust, clock func() time.Time) (http.Handler, error) {
	if service == nil || channel == nil || current == nil || clock == nil {
		return nil, fmt.Errorf("enrollment dependencies required")
	}
	requests := 0
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if requests > 1024 {
			w.Header().Set("Connection", "close")
			http.Error(w, "request budget exhausted", http.StatusTooManyRequests)
			return
		}
		if r.TLS == nil || r.TLS.NegotiatedProtocol != enrollmenttls.Protocol {
			http.Error(w, "dedicated enrollment channel required", http.StatusUnauthorized)
			return
		}
		if _, err := channel.Network(); err != nil {
			http.Error(w, "enrollment connection expired", http.StatusUnauthorized)
			return
		}
		at := clock()
		if at.Unix() < window.Created || at.Unix() >= window.Expires {
			w.Header().Set("Connection", "close")
			http.Error(w, "enrollment window expired", http.StatusGone)
			return
		}
		if r.URL.RawQuery != "" || r.URL.Fragment != "" {
			http.Error(w, "unexpected query", http.StatusBadRequest)
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/trust/enrollment-window" {
			writeJSON(w, window)
			return
		}
		v, err := current(r.Context(), at)
		if err != nil {
			http.Error(w, "trust authority unavailable", http.StatusServiceUnavailable)
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/trust/snapshots/latest" {
			b, err := v.Bytes()
			if err != nil {
				http.Error(w, "trust unavailable", http.StatusServiceUnavailable)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			_, _ = w.Write(b)
			return
		}
		var result enrollment.Progress
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/trust/enrollment-requests":
			var q enrollment.SignedInteractiveRequest
			if err = readJSON(w, r, &q, 16<<10); err == nil {
				if q.Request.Window != window.ID {
					err = fmt.Errorf("wrong enrollment window")
				} else {
					result, err = service.Begin(r.Context(), q, channel, v, at)
				}
			}
		case strings.HasPrefix(r.URL.Path, "/v1/trust/enrollment-requests/"):
			rest := strings.TrimPrefix(r.URL.Path, "/v1/trust/enrollment-requests/")
			parts := strings.Split(rest, "/")
			switch {
			case len(parts) == 1 && r.Method == http.MethodGet:
				result, err = service.Poll(r.Context(), parts[0], channel, v, at)
			case len(parts) == 2 && parts[1] == "client-confirmations" && r.Method == http.MethodPost:
				var q struct {
					Digest string `json:"digest"`
				}
				if err = readJSON(w, r, &q, 1024); err == nil {
					result, err = service.ConfirmClient(r.Context(), parts[0], q.Digest, channel, v, at)
				}
			default:
				http.NotFound(w, r)
				return
			}
		default:
			http.NotFound(w, r)
			return
		}
		if err != nil {
			http.Error(w, "enrollment request rejected", http.StatusBadRequest)
			return
		}
		writeJSON(w, result)
	}), nil
}

func readJSON(w http.ResponseWriter, r *http.Request, dst any, limit int64) error {
	if r.Header.Get("Content-Type") != "application/json" {
		return fmt.Errorf("JSON content type required")
	}
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	b, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}
	defer clear(b)
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	canonical, err := json.Marshal(dst)
	if err != nil {
		return err
	}
	defer clear(canonical)
	if !bytes.Equal(bytes.TrimSpace(b), canonical) {
		return fmt.Errorf("canonical JSON object required")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(v)
}
