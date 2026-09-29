//go:build darwin || linux

package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerjob"
)

const nodeInstallationFile = "agent-installation.json"
const nodeInstallationVersion = "mixlab_agent_installation_v1"

// NodeInstallation is administrator-local configuration, never a job payload.
// Executable hashes are pinned at explicit setup, not learned on each start.
type NodeInstallation struct {
	Version            string `json:"version"`
	Cluster            string `json:"cluster"`
	Node               string `json:"node"`
	PrincipalDirectory string `json:"principal_directory"`
	WorkerBinary       string `json:"worker_binary"`
	WorkerBuild        string `json:"worker_build"`
	GuardianBinary     string `json:"guardian_binary"`
	GuardianBuild      string `json:"guardian_build"`
	RelayAddress       string `json:"relay_address"`
}

func (i NodeInstallation) validate() error {
	if err := i.validateMetadata(); err != nil {
		return err
	}
	for _, exe := range []struct{ path, hash string }{{i.WorkerBinary, i.WorkerBuild}, {i.GuardianBinary, i.GuardianBuild}} {
		got, err := workerjob.FileDigest(exe.path)
		if err != nil || got != exe.hash {
			return fmt.Errorf("approved executable changed; explicit local reapproval required")
		}
	}
	return nil
}

func (i NodeInstallation) validateMetadata() error {
	if i.Version != nodeInstallationVersion || !nodeRouteID(i.Cluster) || !nodeRouteID(i.Node) {
		return fmt.Errorf("invalid node installation identity")
	}
	for _, p := range []string{i.PrincipalDirectory, i.WorkerBinary, i.GuardianBinary} {
		if !filepath.IsAbs(p) || filepath.Clean(p) != p {
			return fmt.Errorf("installation requires canonical absolute local paths")
		}
	}
	return nodeagent.ValidateTransportEndpoint(i.RelayAddress)
}

func nodeSubdirectory(path statehome.Path, name string, kind statehome.Kind) (statehome.Path, error) {
	return statehome.Resolve(statehome.Options{ExactDir: filepath.Join(path.Dir(), name)}, statehome.Context{Kind: kind})
}

func (i NodeInstallation) validateLocation(path statehome.Path) error {
	for _, p := range []string{i.PrincipalDirectory, i.WorkerBinary, i.GuardianBinary} {
		if p == path.Dir() || strings.HasPrefix(p, path.Dir()+string(filepath.Separator)) || strings.HasPrefix(path.Dir(), p+string(filepath.Separator)) {
			return fmt.Errorf("installation and approved identity/executables must be separate")
		}
	}
	return nil
}

// InitializeNodeInstallation publishes node, runtime-allocation and empty
// credential contexts together. Agent startup never calls this initializer.
func InitializeNodeInstallation(ctx context.Context, path statehome.Path, i NodeInstallation, profile nodeagent.Profile) error {
	if path.Kind() != statehome.Agent || profile.Node != i.Node || profile.Probe.BuildID != i.WorkerBuild || profile.Generation != 1 {
		return fmt.Errorf("matching initial local node profile required")
	}
	i.Version = nodeInstallationVersion
	if profile.TransportEndpoint != i.RelayAddress {
		return fmt.Errorf("profile transport endpoint differs from installation")
	}
	if err := i.validate(); err != nil {
		return err
	}
	if err := profile.Validate(); err != nil {
		return err
	}
	if err := i.validateLocation(path); err != nil {
		return err
	}
	b, err := json.Marshal(i)
	if err != nil {
		return err
	}
	return path.Publish(func(stage statehome.Path) error {
		store, err := nodeagent.Initialize(ctx, stage, i.Cluster, i.Node, 1)
		if err != nil {
			return err
		}
		if err := store.InstallProfile(ctx, profile); err != nil {
			return err
		}
		root, err := nodeSubdirectory(stage, "runtime", statehome.Worker)
		if err != nil {
			return err
		}
		if _, err := workerhost.InitializeRuntimeStore(root); err != nil {
			return err
		}
		credentials, err := nodeSubdirectory(stage, "credentials", statehome.Agent)
		if err != nil {
			return err
		}
		if err := credentials.Ensure(); err != nil {
			return err
		}
		probe, err := nodeSubdirectory(stage, "probe", statehome.Worker)
		if err != nil {
			return err
		}
		if err := probe.Ensure(); err != nil {
			return err
		}
		return stage.CompareAndSwap(nodeInstallationFile, nil, b)
	})
}

func OpenNodeInstallation(path statehome.Path) (NodeInstallation, error) {
	i, _, err := readNodeInstallation(path)
	if err != nil {
		return i, err
	}
	return i, i.validate()
}

// InspectNodeInstallation reads local metadata for diagnostics only. It does not
// approve changed binaries or establish that the node is ready to run.
func InspectNodeInstallation(path statehome.Path) (NodeInstallation, error) {
	i, _, err := readNodeInstallation(path)
	return i, err
}

func readNodeInstallation(path statehome.Path) (NodeInstallation, []byte, error) {
	if path.Kind() != statehome.Agent {
		return NodeInstallation{}, nil, fmt.Errorf("agent state required")
	}
	b, err := path.ReadFileLimit(nodeInstallationFile, 16384)
	if err != nil {
		return NodeInstallation{}, nil, err
	}
	var i NodeInstallation
	if err := json.Unmarshal(b, &i); err != nil {
		return i, nil, err
	}
	canonical, _ := json.Marshal(i)
	if !bytes.Equal(b, canonical) {
		return i, nil, fmt.Errorf("noncanonical node installation")
	}
	return i, b, i.validateMetadata()
}

// OpenNodeRuntime validates all durable contexts; missing state cannot turn a
// previously initialized node into an empty, apparently available accelerator.
func OpenNodeRuntime(path statehome.Path, i NodeInstallation, anchor trust.Anchor) (*nodeagent.Store, *workerhost.GuardianRuntime, statehome.Path, error) {
	if i.Cluster != anchor.Cluster() {
		return nil, nil, statehome.Path{}, fmt.Errorf("installation differs from enrolled cluster")
	}
	store, err := nodeagent.Open(path, i.Cluster, i.Node)
	if err != nil {
		return nil, nil, statehome.Path{}, err
	}
	if err := store.CheckApprovedWorker(i.WorkerBuild); err != nil {
		return nil, nil, statehome.Path{}, err
	}
	root, err := nodeSubdirectory(path, "runtime", statehome.Worker)
	if err != nil {
		return nil, nil, statehome.Path{}, err
	}
	runtimes, err := workerhost.OpenRuntimeStore(root)
	if err != nil {
		return nil, nil, statehome.Path{}, err
	}
	runtime, err := workerhost.NewGuardianRuntime(runtimes, i.WorkerBinary, i.WorkerBuild, i.GuardianBinary, i.GuardianBuild, 2*time.Minute, 5*time.Second)
	if err != nil {
		return nil, nil, statehome.Path{}, err
	}
	credentials, err := nodeSubdirectory(path, "credentials", statehome.Agent)
	if err != nil {
		return nil, nil, statehome.Path{}, err
	}
	if err := credentials.Validate(); err != nil {
		return nil, nil, statehome.Path{}, err
	}
	return store, runtime, credentials, nil
}
