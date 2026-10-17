package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust"
)

type delayedAdmissionLeaseClient struct {
	AdmissionLeaseClient
	wait func(context.Context) error
}

func (c delayedAdmissionLeaseClient) Reserve(ctx context.Context, q nodeagent.Reserve) (nodeagent.Lease, error) {
	if err := c.wait(ctx); err != nil {
		return nodeagent.Lease{}, err
	}
	return c.AdmissionLeaseClient.Reserve(ctx, q)
}
func (c delayedAdmissionLeaseClient) LeaseStatus(ctx context.Context, id string) (nodeagent.Lease, error) {
	if err := c.wait(ctx); err != nil {
		return nodeagent.Lease{}, err
	}
	return c.AdmissionLeaseClient.LeaseStatus(ctx, id)
}
func (c delayedAdmissionLeaseClient) Renew(ctx context.Context, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	if err := c.wait(ctx); err != nil {
		return nodeagent.Lease{}, err
	}
	return c.AdmissionLeaseClient.Renew(ctx, q)
}

func delayAdmissionLeases(k *DiLoCoAdmissionLeases, f *admissionLeaseFixture, delay time.Duration) {
	start, unix := time.Now(), f.now.Load()
	k.Clock = func() time.Time { return time.Unix(unix, 0).Add(time.Since(start)) }
	client := k.Client
	k.Client = func(role trust.Role, candidate recruitment.Candidate) (AdmissionLeaseClient, error) {
		c, err := client(role, candidate)
		return delayedAdmissionLeaseClient{c, func(ctx context.Context) error {
			active := f.active.Add(1)
			defer f.active.Add(-1)
			for old := f.maximum.Load(); active > old; old = f.maximum.Load() {
				if f.maximum.CompareAndSwap(old, active) {
					break
				}
			}
			timer := time.NewTimer(delay)
			defer timer.Stop()
			select {
			case <-ctx.Done():
			case <-timer.C:
			}
			f.now.Store(k.Clock().Unix())
			return ctx.Err()
		}}, err
	}
}

func TestAdmissionLeasesLoadedCallsAndLostACK(t *testing.T) {
	for _, mode := range []string{"precommit", "committed", "handoff_ack_lost", "partial_handoff", "reserve_ack_lost"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, 2)
				f.now.Store(1150)
				if mode == "precommit" || mode == "reserve_ack_lost" {
					f.prepared(0)
					f.prepared(1)
				} else {
					f.committed(t)
					for i := range f.v.Nodes {
						if mode != "partial_handoff" || i != 0 {
							f.handedOff(i, mode != "handoff_ack_lost")
						}
					}
				}
				for _, n := range f.nodes {
					// Three delayed calls cover status, accepted renewal with lost
					// ACK, and authoritative readback within one maintenance turn.
					n.lost = mode != "partial_handoff" && mode != "reserve_ack_lost"
				}
				if mode == "reserve_ack_lost" {
					f.v.Nodes[0].Lease = nil
				}
				before, err := json.Marshal(f.v)
				check(t, err)
				delayAdmissionLeases(k, f, 40*time.Second)
				start := time.Now()
				check(t, k.Maintain(context.Background(), f.q, f.v))
				if elapsed := time.Since(start); elapsed != 120*time.Second || elapsed >= diLoCoServiceBudget {
					t.Fatalf("maintenance truncated loaded calls: elapsed=%s", elapsed)
				}
				for i, candidate := range f.q.Launch.Selection.Selected {
					n := f.nodes[candidate.Capabilities.Node]
					commands := f.commands(n.lease.ID)
					if len(commands) != 1 || n.lease.Expires <= k.Clock().Unix() || n.lease.ID != f.plan.Slots[i].Allocation.LeaseID {
						t.Fatal("renewal lost original allocation or duplicated mutation", commands, n.lease)
					}
				}
				after, err := json.Marshal(f.v)
				check(t, err)
				if string(before) != string(after) {
					t.Fatal("maintenance changed durable admission intent")
				}
			})
		})
	}
}

func TestAdmissionLeasesRenewBeforeNextLoadedServiceTurn(t *testing.T) {
	k, f := admissionLeaseSetup(t, 2)
	f.now.Store(1100)
	f.q.Launch.TTLSeconds = 900
	f.v.Request = f.q.Launch
	for _, n := range f.nodes {
		n.lease.Expires = 1400
		n.lease.RenewBy = 1300
	}
	check(t, k.Maintain(context.Background(), f.q, f.v))
	if len(f.renewals) != 2 {
		t.Fatal("waited for RenewBy despite insufficient loaded-turn headroom")
	}
	for _, n := range f.nodes {
		if n.lease.Expires != 2000 {
			t.Fatal("renewal changed accepted TTL", n.lease.Expires)
		}
	}
}

func TestAdmissionLeasesSlowCallsRetainAuthorityCaps(t *testing.T) {
	for _, mode := range []string{"admission", "partial_handoff", "workload"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, 2)
				f.now.Store(2750)
				if mode != "admission" {
					f.committed(t)
				}
				for i, candidate := range f.q.Launch.Selection.Selected {
					f.prepared(i)
					if mode == "workload" {
						f.handedOff(i, true)
					}
					n := f.nodes[candidate.Capabilities.Node]
					n.lease.Expires, n.lease.RenewBy = 2780, 2750
					if mode == "workload" {
						n.lease.WorkloadDeadline = 2800
					}
				}
				delayAdmissionLeases(k, f, 4*time.Second)
				check(t, k.Maintain(context.Background(), f.q, f.v))
				if len(f.renewals) != 2 {
					t.Fatal("slow live leases not renewed")
				}
				for _, n := range f.nodes {
					if n.lease.Expires != 2800 {
						t.Fatal("renewal escaped authority cap", n.lease)
					}
				}
				f.now.Store(2800)
				k.Clock = func() time.Time { return time.Unix(f.now.Load(), 0) }
				if err := k.Maintain(context.Background(), f.q, f.v); !errors.Is(err, ErrAdmissionLeaseLost) {
					t.Fatal("expired authority revived", err)
				}
				if len(f.renewals) != 2 {
					t.Fatal("expired authority sent another renewal")
				}
			})
		})
	}
}

func TestAdmissionLeasesSlowCallsStopAtCallerAndAdmissionDeadlines(t *testing.T) {
	for _, admissionDeadline := range []bool{false, true} {
		t.Run(fmt.Sprintf("admission=%v", admissionDeadline), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, 16)
				f.now.Store(2780)
				delayAdmissionLeases(k, f, time.Minute)
				budget := 10 * time.Second
				if admissionDeadline {
					budget = time.Minute
				}
				ctx, cancel := context.WithTimeout(context.Background(), budget)
				defer cancel()
				start := time.Now()
				err := k.Maintain(ctx, f.q, f.v)
				want := 10 * time.Second
				if admissionDeadline {
					want = 20 * time.Second
					if !errors.Is(err, ErrAdmissionLeaseLost) {
						t.Fatal("admission expiration not fenced", err)
					}
				}
				if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) != want || len(f.renewals) != 0 || f.active.Load() != 0 || f.maximum.Load() > admissionLeaseConcurrency {
					t.Fatalf("deadline changed authority: elapsed=%s renewals=%d err=%v", time.Since(start), len(f.renewals), err)
				}
			})
		})
	}
}

func TestAdmissionLeasesBudgetExhaustionResumesUnvisitedTail(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		k, f := admissionLeaseSetup(t, 64)
		f.now.Store(1150)
		client := k.Client
		blocked := map[string]bool{}
		for _, candidate := range f.q.Launch.Selection.Selected[:24] {
			blocked[candidate.Capabilities.Node] = true
		}
		k.Client = func(role trust.Role, candidate recruitment.Candidate) (AdmissionLeaseClient, error) {
			c, err := client(role, candidate)
			if !blocked[candidate.Capabilities.Node] {
				return c, err
			}
			return delayedAdmissionLeaseClient{c, func(ctx context.Context) error {
				f.active.Add(1)
				defer f.active.Add(-1)
				<-ctx.Done()
				return ctx.Err()
			}}, err
		}
		start := time.Now()
		for turn := range 2 {
			if err := k.Maintain(context.Background(), f.q, f.v); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal("missing bounded unavailable result", err)
			}
			if f.active.Load() != 0 || time.Since(start) != time.Duration(turn+1)*admissionLeaseBudget {
				t.Fatal("maintenance escaped its budget or left an RPC running")
			}
		}
		for _, candidate := range f.q.Launch.Selection.Selected[24:] {
			n := f.nodes[candidate.Capabilities.Node]
			if len(f.commands(n.lease.ID)) != 1 {
				t.Fatal("unvisited healthy peer starved behind unavailable prefix", candidate.Capabilities.Node)
			}
		}
	})
}

func TestAdmissionLeasesBudgetExhaustionResumesInterruptedRenewals(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		k, f := admissionLeaseSetup(t, 16)
		delayAdmissionLeases(k, f, 40*time.Second)
		start := time.Now()
		for turn := range 4 {
			err := k.Maintain(context.Background(), f.q, f.v)
			if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrAdmissionLeaseLost) {
				t.Fatal("interrupted healthy renewal became allocation loss", err)
			}
			if f.active.Load() != 0 || f.maximum.Load() > admissionLeaseConcurrency || time.Since(start) != time.Duration(turn+1)*admissionLeaseBudget {
				t.Fatal("interrupted scan escaped bounds or failed to join")
			}
			if turn == 0 {
				for i, candidate := range f.q.Launch.Selection.Selected {
					n := f.nodes[candidate.Capabilities.Node]
					want := 1
					if i >= admissionLeaseConcurrency {
						want = 0
					}
					if n.reads == 0 || len(f.commands(n.lease.ID)) != want {
						t.Fatal("fixture did not interrupt already-started renewals", i)
					}
				}
			}
		}
		for _, n := range f.nodes {
			if len(f.commands(n.lease.ID)) != 2 || n.lease.Expires <= k.Clock().Unix() {
				t.Fatal("started-but-interrupted peer starved across scans", n.lease)
			}
		}
	})
}

func TestAdmissionLeasesFullTurnInterruptedPrefixRotates(t *testing.T) {
	for _, count := range []int{16, 64} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, count)
				f.q.Launch.TTLSeconds = 900
				f.v.Request = f.q.Launch
				f.committed(t)
				f.now.Store(1100)
				slow := map[string]bool{}
				for i, candidate := range f.q.Launch.Selection.Selected {
					n := f.nodes[candidate.Capabilities.Node]
					n.request.TTLSeconds = 900
					f.v.Nodes[i].Reserve = n.request
					n.lease.Expires, n.lease.RenewBy = 2000, 1100
					if i < admissionLeaseConcurrency {
						slow[candidate.Capabilities.Node] = true
					} else {
						f.handedOff(i, true)
					}
				}
				before, err := json.Marshal(f.v)
				check(t, err)
				direct := k.Client
				delayAdmissionLeases(k, f, 49*time.Second)
				delayed := k.Client
				k.Client = func(role trust.Role, candidate recruitment.Candidate) (AdmissionLeaseClient, error) {
					if slow[candidate.Capabilities.Node] {
						return delayed(role, candidate)
					}
					return direct(role, candidate)
				}
				start := time.Now()
				for turn := range 4 {
					// Each prefix slot needs coordinator status (49s), controller
					// status (49s), then renewal (49s). Even a full 130s turn
					// interrupts renewal; healthy tail slots must still be serviced.
					err := k.Maintain(context.Background(), f.q, f.v)
					if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrAdmissionLeaseLost) {
						t.Fatal("interrupted prefix changed authority", err)
					}
					if f.active.Load() != 0 || f.maximum.Load() > admissionLeaseConcurrency || time.Since(start) != time.Duration(turn+1)*admissionLeaseBudget {
						t.Fatal("rotation escaped scan budget or concurrency bounds")
					}
					if turn == 0 {
						next, ok := k.scans.Load(f.q.Launch.Run)
						if !ok || next.(int) != admissionLeaseConcurrency {
							t.Fatal("full-turn interruption pinned the starting cohort", next)
						}
					}
				}
				for _, candidate := range f.q.Launch.Selection.Selected {
					n := f.nodes[candidate.Capabilities.Node]
					if slow[candidate.Capabilities.Node] {
						if len(f.commands(n.lease.ID)) != 0 || n.reads != 8 {
							t.Fatal("rotation assumed renewal success or abandoned interrupted peers")
						}
					} else if len(f.commands(n.lease.ID)) != 1 || n.reads != 2 || n.lease.Expires <= k.Clock().Unix() {
						t.Fatal("healthy tail starved or duplicated renewal", n.lease)
					}
				}
				after, err := json.Marshal(f.v)
				check(t, err)
				if string(before) != string(after) {
					t.Fatal("scan rotation changed durable admission intent")
				}
			})
		})
	}
}

type statusAdmissionLeaseClient struct {
	AdmissionLeaseClient
	status func(context.Context, string) (nodeagent.Lease, error)
}

func (c statusAdmissionLeaseClient) LeaseStatus(ctx context.Context, id string) (nodeagent.Lease, error) {
	return c.status(ctx, id)
}

func TestAdmissionLeasesExpiredControllerWindowDoesNotProveHandoffLoss(t *testing.T) {
	for _, crossing := range []bool{false, true} {
		t.Run(fmt.Sprintf("crossing=%v", crossing), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, 2)
				f.committed(t)
				for i, candidate := range f.q.Launch.Selection.Selected {
					f.handedOff(i, false)
					n := f.nodes[candidate.Capabilities.Node]
					n.lease.Expires, n.lease.RenewBy = 3000, 2800
				}
				unix := f.v.Deadline + 1
				if crossing {
					unix = f.v.Deadline - 10
				}
				start := time.Now()
				k.Clock = func() time.Time { return time.Unix(unix, 0).Add(time.Since(start)) }
				f.now.Store(unix)
				unavailable := true
				client := k.Client
				k.Client = func(role trust.Role, candidate recruitment.Candidate) (AdmissionLeaseClient, error) {
					c, err := client(role, candidate)
					return statusAdmissionLeaseClient{c, func(ctx context.Context, id string) (nodeagent.Lease, error) {
						if unavailable && role == trust.Coordinator {
							if crossing {
								time.Sleep(40 * time.Second)
								f.now.Store(k.Clock().Unix())
							}
							return nodeagent.Lease{}, context.DeadlineExceeded
						}
						return c.LeaseStatus(ctx, id)
					}}, err
				}
				before, err := json.Marshal(f.v)
				check(t, err)
				err = k.Maintain(context.Background(), f.q, f.v)
				if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrAdmissionLeaseLost) || len(f.renewals) != 0 {
					t.Fatal("unobserved owner became proven loss or assumed renewal", err)
				}
				unavailable = false
				check(t, k.Maintain(context.Background(), f.q, f.v))
				if len(f.renewals) != 2 {
					t.Fatal("fresh coordinator observation did not resolve handoff uncertainty")
				}
				for _, role := range f.roles {
					if role != trust.Coordinator {
						t.Fatal("expired controller authority renewed a lease")
					}
				}
				after, err := json.Marshal(f.v)
				check(t, err)
				if string(before) != string(after) {
					t.Fatal("observation invented a handoff acknowledgement")
				}
			})
		})
	}
}

func TestAdmissionLeasesObservedExpiredAuthorityStillFailsClosed(t *testing.T) {
	for _, owner := range []trust.Role{trust.Controller, trust.Coordinator} {
		t.Run(string(owner), func(t *testing.T) {
			k, f := admissionLeaseSetup(t, 2)
			f.committed(t)
			f.now.Store(f.v.Deadline + 1)
			for i, candidate := range f.q.Launch.Selection.Selected {
				if owner == trust.Coordinator {
					f.handedOff(i, false)
				}
				n := f.nodes[candidate.Capabilities.Node]
				// A future lease expiry cannot override observed expired authority,
				// even when the normal renewal/headroom check would skip renewal.
				n.lease.Expires, n.lease.RenewBy = 4000, 3500
				if owner == trust.Coordinator {
					n.lease.WorkloadDeadline = f.v.Deadline
				}
			}
			if err := k.Maintain(context.Background(), f.q, f.v); !errors.Is(err, ErrAdmissionLeaseLost) {
				t.Fatal("observed expired authority was treated as uncertain", err)
			}
			if len(f.renewals) != 0 {
				t.Fatal("expired authority was extended")
			}
		})
	}
}

func TestAdmissionLeasesDelayedObservationStillRejectsLoss(t *testing.T) {
	for _, mode := range []string{"expired", "scope", "cleanup"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, 2)
				f.now.Store(1150)
				n := f.nodes[f.q.Launch.Selection.Selected[0].Capabilities.Node]
				switch mode {
				case "expired":
					n.lease.Expires = 1180
				case "scope":
					n.lease.Run = "foreign"
				case "cleanup":
					n.lease.CleanupReason = "failed"
				}
				delayAdmissionLeases(k, f, 40*time.Second)
				if err := k.Maintain(context.Background(), f.q, f.v); !errors.Is(err, ErrAdmissionLeaseLost) {
					t.Fatal("slow response hid lost allocation", err)
				}
				if len(f.commands(n.lease.ID)) != 0 {
					t.Fatal("invalid allocation renewed")
				}
			})
		})
	}
}

type ambiguousAdmissionLeaseClient struct {
	AdmissionLeaseClient
	applied     bool
	readbackErr error
	observe     func()
}

func (c *ambiguousAdmissionLeaseClient) Renew(ctx context.Context, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	lease, err := c.AdmissionLeaseClient.Renew(ctx, q)
	if err != nil {
		return lease, err
	}
	c.applied = true
	<-ctx.Done()
	return nodeagent.Lease{}, ctx.Err()
}

func (c *ambiguousAdmissionLeaseClient) LeaseStatus(ctx context.Context, id string) (nodeagent.Lease, error) {
	c.observe()
	if c.applied && c.readbackErr != nil {
		return nodeagent.Lease{}, c.readbackErr
	}
	return c.AdmissionLeaseClient.LeaseStatus(ctx, id)
}

func TestAdmissionLeasesLostACKAcrossObservedExpiry(t *testing.T) {
	for _, readbackErr := range []error{nil, io.ErrUnexpectedEOF} {
		t.Run(fmt.Sprint(readbackErr), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				k, f := admissionLeaseSetup(t, 2)
				f.now.Store(1150)
				for _, n := range f.nodes {
					n.lease.Expires = 1160
				}
				start := time.Now()
				k.Clock = func() time.Time { return time.Unix(1150, 0).Add(time.Since(start)) }
				client := k.Client
				k.Client = func(role trust.Role, candidate recruitment.Candidate) (AdmissionLeaseClient, error) {
					c, err := client(role, candidate)
					return &ambiguousAdmissionLeaseClient{AdmissionLeaseClient: c, readbackErr: readbackErr, observe: func() { f.now.Store(k.Clock().Unix()) }}, err
				}
				err := k.Maintain(context.Background(), f.q, f.v)
				if readbackErr == nil {
					check(t, err)
				} else if !errors.Is(err, readbackErr) || errors.Is(err, ErrAdmissionLeaseLost) {
					t.Fatal("ambiguous renewal was mistaken for proven allocation loss", err)
				}
				if len(f.renewals) != 2 || time.Since(start) != 10*time.Second {
					t.Fatal("lost ACK escaped old expiry bound or duplicated renewal")
				}
			})
		})
	}
}

func TestAdmissionLeaseCallRejectsSuccessAfterCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, err := admissionLeaseCall(context.Background(), func(ctx context.Context) (nodeagent.Lease, error) {
			<-ctx.Done()
			return nodeagent.Lease{ID: "late"}, nil
		})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal("accepted response after call deadline", err)
		}
	})
}
