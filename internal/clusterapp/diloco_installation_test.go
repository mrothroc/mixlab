package clusterapp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
)

func TestDiLoCoIndependentRootOwnership(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	path := func(name string) statehome.Path {
		p, e := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(base, name)}, statehome.Context{Kind: statehome.Principal})
		if e != nil {
			t.Fatal(e)
		}
		return p
	}
	o := DiLoCoCoordinatorOptions{Principal: path("identity"), AdmissionPrincipal: path("controller"), State: path("state"), Admission: path("admission"), Artifacts: path("artifacts"), Checkpoints: path("checkpoints"), Endpoint: "https://127.0.0.1:7446"}
	owner := run.Coordinator{ClusterID: strings.Repeat("a", 32), PrincipalID: strings.Repeat("b", 32), Role: "coordinator", Audience: "learner-run"}
	controller := strings.Repeat("c", 32)
	if err = validateCoordinatorRoots(o, owner, controller, false); err == nil {
		t.Fatal("accepted uninitialized contexts")
	}
	for range 2 {
		if err = validateCoordinatorRoots(o, owner, controller, true); err != nil {
			t.Fatal(err)
		}
	}
	if err = validateCoordinatorRoots(o, owner, controller, false); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*DiLoCoCoordinatorOptions){func(v *DiLoCoCoordinatorOptions) { v.Admission = v.State }, func(v *DiLoCoCoordinatorOptions) { v.State, v.Artifacts = v.Artifacts, v.State }, func(v *DiLoCoCoordinatorOptions) { v.Endpoint = "https://other.local:7446" }} {
		bad := o
		mutate(&bad)
		if err = validateCoordinatorRoots(bad, owner, controller, true); err == nil {
			t.Fatal("accepted path/config substitution")
		}
	}
	if err = os.Remove(filepath.Join(o.State.Dir(), coordinatorOwnerFile)); err != nil {
		t.Fatal(err)
	}
	if err = validateCoordinatorRoots(o, owner, controller, true); err == nil {
		t.Fatal("reinitialized missing published ownership")
	}
}
