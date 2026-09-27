package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/principal"
)

func fetchSubmissionOutput(ctx context.Context, p *principal.Store, path statehome.Path, stdout io.Writer) error {
	launch, err := recruitment.OpenLaunch(path)
	if err != nil {
		return err
	}
	identity, _, err := p.Active(time.Now())
	if err != nil {
		return err
	}
	node, job, err := launch.OutputOwner(identity.Principal)
	if err != nil {
		return err
	}
	client, err := clusterapp.NewNodeClient(p, node.Endpoint, node.Capabilities.Node, time.Now)
	if err != nil {
		return err
	}
	name, err := launch.OutputFilename(identity.Principal)
	if err != nil {
		return err
	}
	ref, err := client.FetchOutput(ctx, job, path, name)
	if err != nil {
		return fmt.Errorf("training completed but output retrieval failed; retry submit -fetch with retained attempt %s: %w", path.Dir(), err)
	}
	b, err := json.Marshal(submissionReceipt{Name: name, Ref: ref})
	if err != nil {
		return err
	}
	if err := path.CompareAndSwap(submissionOutputReceipt, nil, b); err != nil {
		old, readErr := path.ReadFileLimit(submissionOutputReceipt, 4096)
		if readErr != nil || string(old) != string(b) {
			return fmt.Errorf("output receipt conflict: %w", err)
		}
	}
	return json.NewEncoder(stdout).Encode(struct {
		Status   string       `json:"status"`
		Path     string       `json:"path"`
		Artifact artifact.Ref `json:"artifact"`
	}{"succeeded", filepath.Join(path.Dir(), name), ref})
}
