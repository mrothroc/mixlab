package statehome

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

var (
	ErrUnsafe    = errors.New("unsafe state path")
	ErrNotFound  = errors.New("no matching state directory")
	ErrAmbiguous = errors.New("multiple matching state directories; specify an exact directory")
	ErrExists    = errors.New("state directory already exists")
)

// Kind identifies a storage context, not a domain role or permission.
type Kind string

const (
	Authority  Kind = "authority"
	Principal  Kind = "principal"
	Agent      Kind = "agent"
	Worker     Kind = "worker"
	Enrollment Kind = "enrollment"
)

// Options are explicit composition inputs. Env is the value of
// MIXLAB_STATE_HOME. WorkingDir is required only for relative selected paths.
// No shell expansion, environment lookup, or working-directory lookup occurs.
type Options struct {
	ExactDir   string
	Flag       string
	Env        string
	UserHome   string
	WorkingDir string
}

// Context contains opaque path segments. Role is used only by Principal; ID is
// a principal, node, worker, or staging ID. Authority uses only ClusterID and
// Enrollment uses only ID. Discovery treats empty applicable fields as wildcards.
type Context struct {
	Kind      Kind
	ClusterID string
	Role      string
	ID        string
}

// Path is a resolved context directory. Its zero value is invalid. Resolution
// does not imply that a directory exists, is safe, or contains valid domain state.
type Path struct {
	dir  string
	root string
	kind Kind
}

func (p Path) Dir() string { return p.dir }
func (p Path) Kind() Kind  { return p.kind }

// Resolve is pure. Precedence is ExactDir > Flag > Env > UserHome/.mixlab.
// Without ExactDir all applicable context segments must be supplied.
func Resolve(o Options, c Context) (Path, error) {
	parts, err := c.parts()
	if err != nil {
		return Path{}, err
	}
	root, err := o.base()
	if err != nil {
		return Path{}, err
	}
	if o.ExactDir != "" {
		return Path{root, root, c.Kind}, nil
	}
	for _, part := range parts {
		if part == "" {
			return Path{}, fmt.Errorf("%w: incomplete context", ErrUnsafe)
		}
	}
	return Path{filepath.Join(append([]string{root}, parts...)...), root, c.Kind}, nil
}

func (o Options) base() (string, error) {
	selected := o.ExactDir
	if selected == "" {
		selected = o.Flag
	}
	if selected == "" {
		selected = o.Env
	}
	if selected == "" {
		if o.UserHome == "" {
			return "", fmt.Errorf("%w: user home is required", ErrUnsafe)
		}
		// Validate before Join can erase traversal segments.
		home, err := absolute(o.UserHome, o.WorkingDir)
		if err != nil {
			return "", err
		}
		return filepath.Join(home, ".mixlab"), nil
	}
	return absolute(selected, o.WorkingDir)
}

func absolute(path, cwd string) (string, error) {
	if path == "" || strings.ContainsAny(path, "\\\x00") {
		return "", fmt.Errorf("%w: invalid path", ErrUnsafe)
	}
	for _, s := range strings.Split(path, "/") {
		if s == ".." {
			return "", fmt.Errorf("%w: parent traversal", ErrUnsafe)
		}
	}
	if !filepath.IsAbs(path) {
		if !filepath.IsAbs(cwd) {
			return "", fmt.Errorf("%w: absolute working directory is required", ErrUnsafe)
		}
		base, err := absolute(cwd, "")
		if err != nil {
			return "", err
		}
		path = filepath.Join(base, path)
	}
	return filepath.Clean(path), nil
}

const temporaryPrefix = ".statehome-"

func segment(s string) error {
	if s == "" || s == "." || s == ".." || strings.ContainsAny(s, "/\\\x00:") || strings.HasPrefix(s, temporaryPrefix) {
		return fmt.Errorf("%w: invalid path segment %q", ErrUnsafe, s)
	}
	return nil
}

func (c Context) parts() ([]string, error) {
	for _, s := range []string{c.ClusterID, c.Role, c.ID} {
		if s != "" {
			if err := segment(s); err != nil {
				return nil, err
			}
		}
	}
	switch c.Kind {
	case Authority:
		if c.Role == "" && c.ID == "" {
			return []string{"clusters", c.ClusterID, "authority"}, nil
		}
	case Principal:
		return []string{"principals", c.ClusterID, c.Role, c.ID}, nil
	case Agent, Worker:
		if c.Role == "" {
			return []string{string(c.Kind) + "s", c.ClusterID, c.ID}, nil
		}
	case Enrollment:
		if c.ClusterID == "" && c.Role == "" {
			return []string{"staging", "enrollment", c.ID}, nil
		}
	}
	return nil, fmt.Errorf("%w: invalid context or extraneous fields", ErrUnsafe)
}
