// Package bootstrap owns explicit local cluster initialization. It never
// enrolls remote nodes, listens on a socket, or changes system trust stores.
package bootstrap

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
)

const version = "mixlab_cluster_bootstrap_v1"
const stateFile = "cluster-bootstrap.json"
const contextFile = "bootstrap-context.json"
const lockFile = "cluster-bootstrap.lock"
const credentialFile = "principal.json"
const maxBytes = 1 << 20

type IDs struct{ Cluster, Authority, Controller, Coordinator string }

func NewIDs() (IDs, error) {
	var ids IDs
	for _, dst := range []*string{&ids.Cluster, &ids.Authority, &ids.Controller, &ids.Coordinator} {
		s, err := certificates.NewID()
		if err != nil {
			return IDs{}, err
		}
		*dst = s
	}
	return ids, nil
}

type Target struct {
	Role           trust.Role
	Principal      string
	Final, Staging statehome.Path
}
type Config struct {
	Cluster                     string
	Authority                   statehome.Path
	Principals                  []Target
	Backend, Endpoint, Audience string
}
type Report struct {
	Cluster      string                `json:"cluster"`
	Identity     identity.Presentation `json:"identity"`
	AuthorityDir string                `json:"authority_dir"`
	Principals   []PrincipalLocation   `json:"principals"`
}
type PrincipalLocation struct {
	Role      trust.Role `json:"role"`
	Principal string     `json:"principal"`
	Directory string     `json:"directory"`
}

type keyRecord struct {
	Slot        keylifecycle.Slot  `json:"slot"`
	Started     bool               `json:"started"`
	Handle      *securekeys.Handle `json:"handle"`
	Certificate []byte             `json:"certificate"`
}
type contextRecord struct {
	Owner     string      `json:"owner"`
	Principal string      `json:"principal"`
	Final     string      `json:"final"`
	Staging   string      `json:"staging"`
	Scope     string      `json:"scope"`
	Published bool        `json:"published"`
	Keys      []keyRecord `json:"keys"`
}
type record struct {
	Version   string                 `json:"version"`
	Cluster   string                 `json:"cluster"`
	Created   int64                  `json:"created"`
	Backend   string                 `json:"backend"`
	Endpoint  string                 `json:"endpoint"`
	Audience  string                 `json:"audience"`
	Ready     bool                   `json:"ready"`
	Contexts  []contextRecord        `json:"contexts"`
	Endpoints *trust.SignedEndpoints `json:"endpoints"`
	Snapshot  *trust.SignedSnapshot  `json:"snapshot"`
}
type contextMarker struct {
	Version   string `json:"version"`
	Cluster   string `json:"cluster"`
	Owner     string `json:"owner"`
	Principal string `json:"principal"`
	Scope     string `json:"scope"`
	Backend   string `json:"backend"`
}

// PrincipalState contains only this principal's handle and public trust. No
// root/issuer/snapshot private handle appears in a principal context.
type PrincipalState struct {
	Version     string               `json:"version"`
	Cluster     string               `json:"cluster"`
	Role        trust.Role           `json:"role"`
	Principal   string               `json:"principal"`
	Key         securekeys.Handle    `json:"key"`
	Root        []byte               `json:"root"`
	Fingerprint string               `json:"fingerprint"`
	Chain       [][]byte             `json:"chain"`
	Snapshot    trust.SignedSnapshot `json:"snapshot"`
}

func marker(r record, c contextRecord) contextMarker {
	return contextMarker{version, r.Cluster, c.Owner, c.Principal, c.Scope, r.Backend}
}
func encode(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if len(b) > maxBytes {
		return nil, fmt.Errorf("bootstrap record too large")
	}
	return b, err
}
func exact(raw []byte, v any) error {
	if len(raw) > maxBytes {
		return fmt.Errorf("bootstrap record too large")
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("invalid bootstrap record")
	}
	b, err := encode(v)
	if err != nil || !bytes.Equal(raw, b) {
		return fmt.Errorf("noncanonical bootstrap record")
	}
	return nil
}
func scope(cluster, owner, principal string) string {
	d := sha256.Sum256([]byte(version + "\x00" + cluster + "\x00" + owner + "\x00" + principal))
	return hex.EncodeToString(d[:])
}
func resolved(dir string, kind statehome.Kind) (statehome.Path, error) {
	return statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: kind})
}
func selectedPath(c contextRecord) (statehome.Path, error) {
	kind := statehome.Enrollment
	dir := c.Staging
	if c.Owner == "cluster-authority" {
		kind = statehome.Authority
	} else if c.Published {
		kind = statehome.Principal
		dir = c.Final
	}
	return resolved(dir, kind)
}
func (r record) validate(authority string) error {
	u, err := url.Parse(r.Endpoint)
	if err != nil || len(r.Endpoint) > 2048 || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") || u.String() != r.Endpoint {
		return fmt.Errorf("invalid bootstrap endpoint")
	}
	if r.Version != version || !certificates.ValidID(r.Cluster) || r.Created <= 0 || (r.Backend != "file" && r.Backend != "keychain") || r.Audience == "" || len(r.Audience) > 256 || len(r.Contexts) != 4 {
		return fmt.Errorf("invalid bootstrap plan")
	}
	for _, c := range r.Audience {
		if c < 33 || c > 126 {
			return fmt.Errorf("invalid bootstrap audience")
		}
	}
	roles := []string{"cluster-authority", "authority", "controller", "coordinator"}
	paths := []string{}
	ids := map[string]bool{}
	keys := map[string]bool{}
	for n, c := range r.Contexts {
		if c.Owner != roles[n] || c.Scope != scope(r.Cluster, c.Owner, c.Principal) {
			return fmt.Errorf("bootstrap owner/scope mismatch")
		}
		if n == 0 {
			if c.Principal != "" || c.Final != authority || c.Staging != authority || len(c.Keys) != 3 {
				return fmt.Errorf("invalid CA context")
			}
			paths = append(paths, c.Final)
		} else {
			if !certificates.ValidID(c.Principal) || ids[c.Principal] || len(c.Keys) != 1 {
				return fmt.Errorf("invalid initial principal")
			}
			ids[c.Principal] = true
			paths = append(paths, c.Final, c.Staging)
		}
		for k, entry := range c.Keys {
			want := keylifecycle.Principal
			if n == 0 {
				want = []keylifecycle.Slot{keylifecycle.Root, keylifecycle.Issuer, keylifecycle.SnapshotSigner}[k]
			}
			if entry.Slot != want {
				return fmt.Errorf("invalid bootstrap key slot")
			}
			if entry.Handle != nil {
				h := entry.Handle
				if !entry.Started || h.Version != securekeys.Version || h.Scope != c.Scope || h.Backend != r.Backend || !certificates.ValidID(h.ID) || len(h.PublicKey) != 32 || keys[string(h.PublicKey)] {
					return fmt.Errorf("invalid or reused bootstrap key")
				}
				keys[string(h.PublicKey)] = true
			}
			if len(entry.Certificate) > 0 && entry.Handle == nil {
				return fmt.Errorf("certificate without key")
			}
			if r.Ready && (entry.Handle == nil || len(entry.Certificate) == 0 || !c.Published) {
				return fmt.Errorf("incomplete ready bootstrap")
			}
		}
	}
	for i, p := range paths {
		if !filepath.IsAbs(p) || filepath.Clean(p) != p {
			return fmt.Errorf("noncanonical bootstrap path")
		}
		if _, err := resolved(p, statehome.Principal); err != nil {
			return err
		}
		for _, q := range paths[:i] {
			if p == q || strings.HasPrefix(p, q+string(filepath.Separator)) || strings.HasPrefix(q, p+string(filepath.Separator)) {
				return fmt.Errorf("overlapping bootstrap paths")
			}
		}
	}
	if r.Ready && (r.Snapshot == nil || r.Endpoints == nil) {
		return fmt.Errorf("missing ready trust")
	}
	return nil
}
