//go:build darwin || linux

package clusterapp

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

// NodeServiceLock is shared by foreground/service startup and offline upgrades.
const NodeServiceLock = "node-service.lock"

func ReapproveNodeInstallation(ctx context.Context, path statehome.Path, worker, guardian string, probe func(context.Context, string, string) (workerprobe.Report, error)) (out NodeInstallation, err error) {
	if probe == nil || !filepath.IsAbs(worker) || !filepath.IsAbs(guardian) {
		return out, fmt.Errorf("absolute executables and local probe required")
	}
	err = path.WithProcessLock(ctx, NodeServiceLock, func() error {
		i, old, err := readNodeInstallation(path)
		if err != nil {
			return err
		}
		i.WorkerBinary, i.GuardianBinary = filepath.Clean(worker), filepath.Clean(guardian)
		if err := i.validateLocation(path); err != nil {
			return err
		}
		i.WorkerBuild, err = workerjob.FileDigest(worker)
		if err != nil {
			return err
		}
		i.GuardianBuild, err = workerjob.FileDigest(guardian)
		if err != nil {
			return err
		}
		if err := i.validate(); err != nil {
			return err
		}
		s, err := nodeagent.Open(path, i.Cluster, i.Node)
		if err != nil {
			return err
		}
		err = s.ReapproveWorker(ctx, func(ctx context.Context) (workerprobe.Report, error) {
			r, err := probe(ctx, i.WorkerBinary, i.WorkerBuild)
			if err == nil && r.BuildID != i.WorkerBuild {
				err = fmt.Errorf("probe differs from approved worker")
			}
			return r, err
		}, func() error {
			if err := i.validate(); err != nil {
				return err
			}
			b, err := json.Marshal(i)
			if err != nil {
				return err
			}
			return path.CompareAndSwap(nodeInstallationFile, old, b)
		}, time.Now())
		out = i
		return err
	})
	return out, err
}
