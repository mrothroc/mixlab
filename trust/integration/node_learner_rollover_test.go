package integration

import (
	"context"
	"crypto"
	"errors"
	"io"
	"net/http"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

func TestNodeLearnerAdmissionSnapshotRollover(t *testing.T) {
	for _, stage := range []string{"prepare", "credential", "finalize"} {
		for _, mode := range []string{"refresh", "unavailable", "malformed", "revoked", "caller_revoked", "caller_certificate_revoked", "caller_role_removed", "wrong_job", "wrong_scope", "request_deadline"} {
			if stage == "prepare" && (mode == "caller_revoked" || mode == "caller_certificate_revoked" || mode == "caller_role_removed" || mode == "wrong_job" || mode == "wrong_scope" || mode == "request_deadline") || stage == "finalize" && mode == "request_deadline" {
				continue
			}
			t.Run(stage+"/"+mode, func(t *testing.T) {
				x := newLearnerHTTP(t)
				f := x.f
				n := &recoveryNodeTrust{}
				refreshes := 0
				var beforeOperation *nodeagent.LocalExecution
				revokedPrincipal := x.prepare.Signed.Manifest.Controller
				if stage == "finalize" {
					revokedPrincipal = x.prepare.Signed.Manifest.Plan.Coordinator.PrincipalID
				}
				n.refresh = func(ctx context.Context) error {
					if _, ok := ctx.Deadline(); !ok {
						t.Fatal("node refresh has no deadline")
					}
					refreshes++
					if mode == "unavailable" {
						return io.ErrUnexpectedEOF
					}
					if mode == "request_deadline" {
						<-ctx.Done()
						return ctx.Err()
					}
					if mode == "caller_revoked" || mode == "caller_certificate_revoked" || mode == "caller_role_removed" {
						caller, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
						check(t, err)
						f.snapshot.Payload.Generation++
						if mode == "caller_role_removed" {
							roles := f.snapshot.Payload.EligibleRoles[:0]
							for _, role := range f.snapshot.Payload.EligibleRoles {
								if role != trust.Controller {
									roles = append(roles, role)
								}
							}
							f.snapshot.Payload.EligibleRoles = roles
						} else {
							kind, id := "principal", caller.Principal
							if mode == "caller_certificate_revoked" {
								kind, id = "certificate", caller.Serial
							}
							f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: kind, ID: id, Mode: "compromise", Reason: "test", FirstGeneration: f.snapshot.Payload.Generation}}
						}
						f.signSnapshot(t)
					}
					if mode == "revoked" {
						f.snapshot.Payload.Generation++
						f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: revokedPrincipal, Mode: "compromise", Reason: "test", FirstGeneration: f.snapshot.Payload.Generation}}
						f.signSnapshot(t)
					}
					n.pin(f.view)
					return nil
				}
				installRecoveryNodeTrust(t, x, n)
				controller, _ := x.client(t, f.client)
				coordinator, _ := x.client(t, x.coordinator)
				advance := func(at string) {
					if stage == at {
						n.pin(f.view)
						f.snapshot.Payload.Generation++
						f.signSnapshot(t)
					}
				}
				assertResult := func(err error) {
					if mode == "refresh" {
						check(t, err)
					} else {
						want := http.StatusConflict
						if mode == "unavailable" || mode == "request_deadline" {
							want = http.StatusServiceUnavailable
						}
						var remote *clusterapp.NodeHTTPError
						if !errors.As(err, &remote) || remote.Status != want {
							t.Fatalf("rollover status: %v, want %d", err, want)
						}
					}
					want := 1
					if mode == "malformed" || mode == "wrong_job" || mode == "wrong_scope" {
						want = 0
					}
					if refreshes != want {
						t.Fatalf("refreshes=%d, want %d", refreshes, want)
					}
					if mode != "refresh" && beforeOperation != nil {
						after, err := x.store.ActiveExecution(context.Background())
						check(t, err)
						if !reflect.DeepEqual(beforeOperation, after) {
							t.Fatal("rejected evidence changed local learner state")
						}
					}
				}
				ctx := context.Background()
				advance("prepare")
				request, err := x.prepare.Signed.Manifest.SigningRequest()
				check(t, err)
				x.prepare.Signed.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, request, f.now)
				check(t, err)
				if stage == "prepare" && mode == "malformed" {
					x.prepare.Signed.Proof.Signature[0] ^= 1
				}
				prepared, err := controller.PrepareLearner(ctx, x.prepare)
				if stage == "prepare" {
					assertResult(err)
					return
				}
				check(t, err)
				advance("credential")
				if stage == "credential" && mode == "wrong_scope" {
					scope := prepared.Request.Request.Scope
					scope.Job = id(t)
					prepared.Request.Request, err = workload.NewRequest(scope, key(t))
					check(t, err)
					request, err = prepared.Request.Request.SigningRequest()
					check(t, err)
					cert := x.policy.ServerConfig().Certificates[0]
					prepared.Request.Proof, err = trust.SignPrincipalProof(f.a, cert.Certificate, cert.PrivateKey.(crypto.Signer), f.view, request, f.now)
					check(t, err)
				}
				request, err = prepared.Request.GrantRequest()
				check(t, err)
				proof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, request, f.now)
				check(t, err)
				grant := workload.Grant{Request: prepared.Request, Proof: proof}
				issuer, err := workload.Initialize(ctx, learnerHTTPPath(t, statehome.Authority), workload.Authority{Anchor: f.a, Issuer: f.issuer, Key: f.issuerKey}, f.view, f.now)
				check(t, err)
				credential, err := issuer.Issue(ctx, f.client.Chain, grant, f.view, f.now)
				check(t, err)
				if stage == "credential" {
					revokedPrincipal = credential.Binding.Principal
				}
				if stage == "credential" && mode == "malformed" {
					credential.Binding.Job = id(t)
				}
				job := x.prepare.Signed.Manifest.Slot.Allocation.JobID
				target := job
				if stage == "credential" && mode == "wrong_job" {
					target = id(t)
				}
				beforeOperation, err = x.store.ActiveExecution(ctx)
				check(t, err)
				err = controller.CredentialLearner(ctx, target, clusterapp.NodeLearnerCredentialRequest{Grant: grant, Credential: credential})
				if stage == "credential" {
					assertResult(err)
					return
				}
				check(t, err)
				check(t, controller.StartLearner(ctx, job, clusterapp.NodeLearnerStartRequest{Key: id(t)}))
				advance("finalize")
				plan := x.prepare.Signed.Manifest.Plan
				commit := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: plan.Cohort.RunID, PlanHash: plan.SHA256, Generation: plan.Cohort.Generation, CohortHash: plan.Cohort.CohortHash}, Coordinator: plan.Coordinator, Registrations: []run.Registration{prepared.Registration}, CommittedAt: f.now.Unix()}
				if mode == "wrong_scope" {
					allocation := &commit.Registrations[0].Slot.Allocation
					allocation.JobID, allocation.RefHash = id(t), ""
					allocation.RefHash, err = run.Digest(*allocation)
					check(t, err)
				}
				commit.Binding.CommitHash, err = commit.Digest()
				check(t, err)
				check(t, commit.Validate())
				proof, err = trust.SignPrincipalProof(f.a, x.coordinator.Chain, x.coordinator.Key, f.view, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: commit.Binding.CommitHash, Context: plan.Cohort.RunID, Audience: plan.Coordinator.Audience}, f.now)
				check(t, err)
				if mode == "malformed" {
					proof.Signature[0] ^= 1
				}
				if mode == "wrong_job" {
					target = id(t)
				}
				caller := coordinator
				if mode == "caller_revoked" || mode == "caller_certificate_revoked" || mode == "caller_role_removed" {
					caller = controller
				}
				beforeOperation, err = x.store.ActiveExecution(ctx)
				check(t, err)
				assertResult(caller.FinalizeLearner(ctx, target, authenticated.Commit{Body: commit, Proof: proof}))
			})
		}
	}
}
