package main

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

func doctorFailure(err error) string {
	if errors.Is(err, securekeys.ErrUnavailable) {
		return "keychain_or_backend_unavailable: use the enrolled backend in its authorized session; unlock the login Keychain at the console. No automatic file-backend fallback."
	}
	d := clusterdiagnostic.Classify(err)
	return d.Reason + ": " + d.Hint + " Observed failure only; this does not establish macOS Local Network Privacy denial."
}

func doctorCoordinatorTarget(endpoint, id string) (string, error) {
	u, err := url.Parse(endpoint)
	b, idErr := hex.DecodeString(id)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Opaque != "" || idErr != nil || len(b) != 16 || id != strings.ToLower(id) {
		return "", fmt.Errorf("doctor requires an HTTPS coordinator origin and exact lowercase coordinator-id")
	}
	port := u.Port()
	if port == "" {
		port = "443"
	}
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return "", fmt.Errorf("doctor coordinator endpoint requires a valid port")
	}
	return net.JoinHostPort(u.Hostname(), port), nil
}

func doctorCoordinator(ctx context.Context, home, identity, endpoint, id string, add func(string, string, string), failed func(string, error)) {
	target, err := doctorCoordinatorTarget(endpoint, id)
	if err != nil {
		failed("coordinator_endpoint", err)
		return
	}
	// TCP and authenticated service observations are independent. A reachable
	// listener is not evidence that TLS, trust or the application is ready.
	doctorDial(ctx, target, add, failed)
	if identity == "" {
		add("coordinator_tls_trust", "unknown", "Pass -principal-state-dir for an enrolled controller to authenticate the exact coordinator. TCP alone proves no identity or service readiness.")
		return
	}
	opts, err := stateOptions(home, identity)
	if err != nil {
		failed("coordinator_local_identity", err)
		return
	}
	path, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		failed("coordinator_local_identity", err)
		return
	}
	p, err := principal.Open(path, time.Now())
	if err != nil {
		failed("coordinator_local_identity", err)
		return
	}
	defer func() { _ = p.Close() }()
	s, _, err := p.Active(time.Now())
	if err != nil {
		failed("coordinator_local_identity", errors.Join(clusterdiagnostic.ErrLocalIdentity, err))
		return
	}
	if s.Role != trust.Controller {
		add("coordinator_local_identity", "failed", "The coordinator Info route requires an enrolled controller, not an agent or coordinator identity. No credentials changed.")
		return
	}
	c, err := clusterapp.NewDiLoCoClient(p, endpoint, id, time.Now)
	if err != nil {
		failed("coordinator_tls_trust", err)
		return
	}
	doctorCoordinatorInfo(ctx, c.Info, add, failed)
}

func doctorCoordinatorInfo(ctx context.Context, info func(context.Context) (clusterapp.DiLoCoInfo, error), add func(string, string, string), failed func(string, error)) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if _, err := info(ctx); err != nil {
		failed("coordinator_tls_trust", err)
		return
	}
	add("coordinator_tls_trust", "ok", "Read-only Info request authenticated the exact coordinator with current local trust and matched its installed HTTPS origin. This does not prove learner availability, run progress, or the service process's outbound network permission.")
}
