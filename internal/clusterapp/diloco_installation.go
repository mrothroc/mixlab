package clusterapp

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

const coordinatorOwnerFile = "coordinator-owner.json"

func InitializeDiLoCoCoordinator(o DiLoCoCoordinatorOptions) error {
	if o.Clock == nil {
		return fmt.Errorf("clock required")
	}
	p, err := principal.Open(o.Principal, o.Clock())
	if err != nil {
		return err
	}
	defer func() { _ = p.Close() }()
	a, err := principal.Open(o.AdmissionPrincipal, o.Clock())
	if err != nil {
		return err
	}
	defer func() { _ = a.Close() }()
	s, _, err := p.Active(o.Clock())
	if err != nil {
		return err
	}
	c, _, err := a.Active(o.Clock())
	if err != nil {
		return err
	}
	if s.Role != trust.Coordinator || c.Role != trust.Controller || s.Cluster != c.Cluster || s.Fingerprint != c.Fingerprint {
		return fmt.Errorf("matching enrolled coordinator and controller required")
	}
	return validateCoordinatorRoots(o, run.Coordinator{ClusterID: s.Cluster, PrincipalID: s.Principal, Role: "coordinator", Audience: "learner-run"}, c.Principal, true)
}

func validateCoordinatorRoots(o DiLoCoCoordinatorOptions, owner run.Coordinator, controller string, initialize bool) error {
	u, err := url.Parse(o.Endpoint)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.User != nil || u.Opaque != "" {
		return fmt.Errorf("coordinator requires an HTTPS origin")
	}
	paths := []statehome.Path{o.Principal, o.AdmissionPrincipal, o.State, o.Admission, o.Checkpoints, o.Artifacts}
	for i, p := range paths {
		if p.Dir() == "" {
			return fmt.Errorf("explicit coordinator context path required")
		}
		for j, q := range paths {
			if i != j && (p.Dir() == q.Dir() || strings.HasPrefix(p.Dir(), q.Dir()+string(filepath.Separator))) {
				return fmt.Errorf("coordinator context paths must not overlap")
			}
		}
	}
	for _, root := range []struct {
		name string
		path statehome.Path
	}{{"coordinator", o.State}, {"admission", o.Admission}, {"checkpoint", o.Checkpoints}, {"artifact", o.Artifacts}} {
		b, err := json.Marshal(struct{ Format, Role, Cluster, Coordinator, Admission, Endpoint string }{"mixlab_coordinator_owner_v1", root.name, owner.ClusterID, owner.PrincipalID, controller, o.Endpoint})
		if err != nil {
			return err
		}
		if initialize {
			err = root.path.Publish(func(stage statehome.Path) error { return stage.CompareAndSwap(coordinatorOwnerFile, nil, b) })
			if err != nil && !errors.Is(err, statehome.ErrExists) {
				return err
			}
		}
		got, err := root.path.ReadFileLimit(coordinatorOwnerFile, 4096)
		if err != nil {
			return fmt.Errorf("%s state not initialized: %w", root.name, err)
		}
		if !bytes.Equal(got, b) {
			return fmt.Errorf("%s state belongs to another service configuration", root.name)
		}
	}
	return nil
}
