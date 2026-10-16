package workerhost

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRuntimeLearnerPublicationRecovery(t *testing.T) {
	for _, ready := range []bool{false, true} {
		for _, damage := range []string{"none", "private", "parent", "identity", "changed-identity", "symlink", "permissions"} {
			t.Run(fmtReady(ready)+"/"+damage, func(t *testing.T) {
				s := runtimeStoreFixture(t)
				run, worker := strings.Repeat("a", 32), strings.Repeat("b", 32)
				p, err := s.LearnerPrivate(context.Background(), run, worker)
				if err != nil {
					t.Fatal(err)
				}
				parent := filepath.Dir(p.Dir())
				if filepath.Base(p.Dir()) != "private" {
					t.Fatal("learner did not receive child context")
				}
				identity := filepath.Join(parent, "runtime-identity")
				b, err := os.ReadFile(identity)
				if err != nil || string(b) != "learner/"+run+"/"+worker {
					t.Fatalf("identity: %q %v", b, err)
				}
				entries, err := os.ReadDir(p.Dir())
				if err != nil || len(entries) != 0 {
					t.Fatalf("learner context not empty: %v %v", entries, err)
				}
				if !ready {
					// Publication completed, but the root ready CAS was interrupted.
					b, err := json.Marshal(runtimeIndex{runtimeIndexVersion, []runtimeEntry{{Job: run, Attempt: worker, Kind: "learner"}}})
					if err != nil {
						t.Fatal(err)
					}
					if err := s.root.WriteFile(runtimeIndexFile, b); err != nil {
						t.Fatal(err)
					}
				}
				switch damage {
				case "private":
					err = os.Remove(p.Dir())
				case "parent":
					err = os.RemoveAll(parent)
				case "identity":
					err = os.Remove(identity)
				case "changed-identity":
					err = os.WriteFile(identity, []byte("other"), 0600)
				case "symlink":
					if err = os.Remove(p.Dir()); err == nil {
						err = os.Symlink(t.TempDir(), p.Dir())
					}
				case "permissions":
					err = os.Chmod(p.Dir(), 0755)
				}
				if err != nil {
					t.Fatal(err)
				}
				reopened, err := OpenRuntimeStore(s.root)
				if err != nil {
					t.Fatal(err)
				}
				again, err := reopened.LearnerPrivate(context.Background(), run, worker)
				// Pending allocations have never been handed to a launcher, so a
				// wholly absent parent is retryable. A surviving incomplete parent
				// or any missing ready allocation fails closed.
				wantSuccess := damage == "none" || (!ready && damage == "parent")
				if wantSuccess {
					if err != nil || again.Dir() != p.Dir() {
						t.Fatalf("recovery: %v", err)
					}
				} else if err == nil {
					t.Fatal("accepted damaged publication")
				}
				_, index, err := reopened.load()
				if err != nil || index.Entries[0].Ready != (ready || wantSuccess) {
					t.Fatal("incorrect ready publication after recovery", err)
				}
				if damage == "private" {
					if _, err := os.Lstat(p.Dir()); !os.IsNotExist(err) {
						t.Fatal("recreated missing learner child", err)
					}
				}
			})
		}
	}
}

func fmtReady(ready bool) string {
	if ready {
		return "ready"
	}
	return "pending"
}
