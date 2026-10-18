package clusterapp

import (
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/workload"
)

const workloadIssueRoute = "/v1/trust/workloads/issue"
const workloadIssueAuthorizedRoute = "/v1/trust/workloads/issue-authorized"

func authorityManagedHandler(a *Authority, clock func() time.Time) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.RawQuery != "" || (r.URL.Path != "/v1/trust/principals/renew" && r.URL.Path != workloadIssueRoute && r.URL.Path != workloadIssueAuthorizedRoute) {
			http.NotFound(w, r)
			return
		}
		peer, ok := managedtls.Principal(r.Context())
		if !ok || r.TLS == nil {
			http.Error(w, "managed principal required", http.StatusUnauthorized)
			return
		}
		// Body decoding can take long enough for trust to change. Capture the
		// authority's current view only after decoding, before each mutation.
		current := func() (trust.VerifiedSnapshot, bool) {
			v, err := a.Current(r.Context(), clock())
			if err != nil {
				http.Error(w, "authority unavailable", http.StatusServiceUnavailable)
			}
			return v, err == nil
		}
		var chain [][]byte
		for _, c := range r.TLS.PeerCertificates {
			chain = append(chain, c.Raw)
		}
		switch r.URL.Path {
		case workloadIssueAuthorizedRoute:
			if (peer.Role != trust.Controller && peer.Role != trust.Coordinator) || a.Workloads == nil {
				http.Error(w, "workload issuance unavailable to this principal", http.StatusForbidden)
				return
			}
			var q workload.AuthorizedGrant
			if err := readJSON(w, r, &q, 4*trust.MaxTrustBytes+16384); err != nil {
				http.Error(w, "invalid authorized workload request", http.StatusBadRequest)
				return
			}
			v, ok := current()
			if !ok {
				return
			}
			out, err := a.Workloads.IssueAuthorized(r.Context(), chain, q, v, clock())
			if err != nil {
				http.Error(w, "workload issuance rejected", http.StatusForbidden)
				return
			}
			writeJSON(w, out)
		case workloadIssueRoute:
			if (peer.Role != trust.Controller && peer.Role != trust.Coordinator) || a.Workloads == nil {
				http.Error(w, "workload issuance unavailable to this principal", http.StatusForbidden)
				return
			}
			var q workload.Grant
			if err := readJSON(w, r, &q, 2*trust.MaxTrustBytes+8192); err != nil {
				http.Error(w, "invalid workload request", http.StatusBadRequest)
				return
			}
			v, ok := current()
			if !ok {
				return
			}
			out, err := a.Workloads.Issue(r.Context(), chain, q, v, clock())
			if err != nil {
				http.Error(w, "workload issuance rejected", http.StatusForbidden)
				return
			}
			writeJSON(w, out)
		default:
			var q enrollment.SignedRenewalRequest
			if err := readJSON(w, r, &q, 4096); err != nil {
				http.Error(w, "invalid renewal request", http.StatusBadRequest)
				return
			}
			v, ok := current()
			if !ok {
				return
			}
			out, err := a.Enrollment.Renew(r.Context(), chain, q, v, clock())
			if err != nil {
				http.Error(w, "renewal rejected", http.StatusForbidden)
				return
			}
			writeJSON(w, out)
		}
	})
}
