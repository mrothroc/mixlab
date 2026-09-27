package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/statehome"
)

const leaseFile = "node-leases.json"
const nodeIdentityFile = "node-management.identity"
const nodeReadyFile = "node-management.ready"
const nodeLock = "node-management.lock"
const maxLeases = 4096
const maxReceipts = 32768

type receipt struct {
	Key, Controller, Action, Digest string
	Result                          Lease
}
type leaseJournal struct {
	Version              string             `json:"version"`
	Cluster              string             `json:"cluster"`
	Node                 string             `json:"node"`
	NodeVersion          uint64             `json:"node_version"`
	CapabilityGeneration uint64             `json:"capability_generation"`
	Active               string             `json:"active"`
	Leases               []Lease            `json:"leases"`
	Receipts             []receipt          `json:"receipts"`
	ReservationFences    []reservationFence `json:"reservation_fences,omitempty"`
}
type Store struct {
	path          statehome.Path
	cluster, node string
}

// Initialize is explicit local node setup. Runtime Open never recreates a
// missing journal and therefore cannot forget an uncertain running child.
func Initialize(ctx context.Context, path statehome.Path, cluster, node string, generation uint64) (*Store, error) {
	if path.Kind() != statehome.Agent || !identifier(cluster) || !identifier(node) || generation == 0 {
		return nil, fmt.Errorf("agent context and enrolled node identity required")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	s := &Store{path, cluster, node}
	r := leaseJournal{Version: LeaseVersion, Cluster: cluster, Node: node, NodeVersion: 1, CapabilityGeneration: generation, Leases: []Lease{}, Receipts: []receipt{}}
	err := path.WithProcessLock(ctx, nodeLock, func() error {
		if _, err := path.ReadFileLimit(nodeReadyFile, 256); err == nil {
			return fmt.Errorf("node already initialized")
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err := publishInitial(path, nodeIdentityFile, s.identity()); err != nil {
			return err
		}
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		if err := publishInitial(path, leaseFile, b); err != nil {
			return err
		}
		return path.CompareAndSwap(nodeReadyFile, nil, s.identity())
	})
	return s, err
}

// An incomplete initializer can finish only the exact never-exposed bytes.
// All runtime readers require the separate ready marker before opening state.
func publishInitial(path statehome.Path, name string, b []byte) error {
	old, err := path.ReadFileLimit(name, int64(len(b)+1))
	if err == nil {
		if !bytes.Equal(old, b) {
			return fmt.Errorf("initialization differs from durable %s", name)
		}
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return path.CompareAndSwap(name, nil, b)
}
func Open(path statehome.Path, cluster, node string) (*Store, error) {
	if path.Kind() != statehome.Agent || !identifier(cluster) || !identifier(node) {
		return nil, fmt.Errorf("agent context and enrolled node identity required")
	}
	s := &Store{path, cluster, node}
	_, _, err := s.load()
	if err != nil {
		return nil, err
	}
	return s, nil
}
func (s *Store) load() ([]byte, leaseJournal, error) {
	ready, err := s.path.ReadFileLimit(nodeReadyFile, 256)
	if err != nil {
		return nil, leaseJournal{}, err
	}
	if !bytes.Equal(ready, s.identity()) {
		return nil, leaseJournal{}, fmt.Errorf("node management publication incomplete")
	}
	identity, err := s.path.ReadFileLimit(nodeIdentityFile, 256)
	if err != nil {
		return nil, leaseJournal{}, err
	}
	if !bytes.Equal(identity, s.identity()) {
		return nil, leaseJournal{}, fmt.Errorf("node management identity changed")
	}
	b, err := s.path.ReadFileLimit(leaseFile, 32<<20)
	if err != nil {
		return nil, leaseJournal{}, err
	}
	var r leaseJournal
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) {
		return nil, r, fmt.Errorf("noncanonical lease journal")
	}
	if err := s.validate(r); err != nil {
		return nil, r, err
	}
	return b, r, nil
}

func (s *Store) identity() []byte {
	b, _ := json.Marshal(struct{ Version, Cluster, Node string }{LeaseVersion, s.cluster, s.node})
	return b
}
func (s *Store) validate(r leaseJournal) error {
	if r.Version != LeaseVersion || r.Cluster != s.cluster || r.Node != s.node || r.NodeVersion == 0 || r.CapabilityGeneration == 0 || len(r.Leases) > maxLeases || len(r.Receipts) > maxReceipts || len(r.ReservationFences) > maxReceipts {
		return fmt.Errorf("invalid node lease journal")
	}
	seen := map[string]Lease{}
	active := 0
	for _, l := range r.Leases {
		if err := l.validate(s.node); err != nil {
			return err
		}
		if _, ok := seen[l.ID]; ok {
			return fmt.Errorf("duplicate lease")
		}
		seen[l.ID] = l
		if l.State != Released {
			active++
			if r.Active != l.ID {
				return fmt.Errorf("active lease mismatch")
			}
		}
	}
	if (active == 0 && r.Active != "") || active > 1 {
		return fmt.Errorf("exclusive lease invariant violated")
	}
	commands := map[string]bool{}
	fences := map[string]bool{}
	for _, f := range r.ReservationFences {
		key := f.Controller + ":" + f.Key
		if !identifier(f.Controller) || !identifier(f.Key) || len(f.Digest) != 64 || fences[key] {
			return fmt.Errorf("invalid reservation fence history")
		}
		fences[key] = true
	}
	for _, c := range r.Receipts {
		l, ok := seen[c.Result.ID]
		if !ok || !identifier(c.Key) || c.Controller != l.Controller || c.Result.Controller != l.Controller || c.Result.Version > l.Version || len(c.Digest) != 64 || commands[c.Controller+":"+c.Key] || c.Result.validate(s.node) != nil {
			return fmt.Errorf("invalid lease command history")
		}
		switch c.Action {
		case "reserve", "renew", "release":
		default:
			return fmt.Errorf("invalid lease command action")
		}
		commands[c.Controller+":"+c.Key] = true
	}
	return nil
}
func (s *Store) save(old []byte, r leaseJournal) error {
	if err := s.validate(r); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > 32<<20 {
		return fmt.Errorf("lease journal capacity reached")
	}
	return s.path.CompareAndSwap(leaseFile, old, b)
}

type Availability struct {
	NodeVersion, CapabilityGeneration uint64
	Available                         bool
	Lease                             *Lease
}

func (s *Store) Availability(now time.Time) (Availability, error) {
	_, r, err := s.load()
	if err != nil {
		return Availability{}, err
	}
	v := Availability{NodeVersion: r.NodeVersion, CapabilityGeneration: r.CapabilityGeneration, Available: r.Active == ""}
	for _, l := range r.Leases {
		if l.ID == r.Active {
			copy := l
			if now.Unix() >= l.Expires && l.State != Releasing {
				copy.CleanupReason = "expired"
			}
			v.Lease = &copy
		}
	}
	return v, nil
}
func replay(r leaseJournal, controller, key, action, hash string) (Lease, bool, error) {
	for _, c := range r.Receipts {
		if c.Controller == controller && c.Key == key {
			if c.Action != action || c.Digest != hash {
				return Lease{}, false, fmt.Errorf("idempotency key reused with different command")
			}
			return c.Result, true, nil
		}
	}
	if len(r.Receipts) >= maxReceipts {
		return Lease{}, false, fmt.Errorf("lease command history capacity reached")
	}
	return Lease{}, false, nil
}
func record(r *leaseJournal, controller, key, action, hash string, l Lease) {
	r.Receipts = append(r.Receipts, receipt{key, controller, action, hash, l})
	r.NodeVersion++
}
