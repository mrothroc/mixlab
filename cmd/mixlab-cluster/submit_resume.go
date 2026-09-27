package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/checkpoint"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
)

const submissionOutputReceipt = "download.json"

type submissionReceipt struct {
	Name string       `json:"name"`
	Ref  artifact.Ref `json:"ref"`
}

func loadSubmissionResume(ctx context.Context, dir, controller string) (nodejob.Manifest, nodejob.ArtifactRef, func() (io.ReadSeekCloser, error), error) {
	var zero nodejob.Manifest
	var absent nodejob.ArtifactRef
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		return zero, absent, nil, err
	}
	s, err := recruitment.OpenLaunch(p)
	if err != nil {
		return zero, absent, nil, err
	}
	m, err := s.CheckpointSource(controller)
	if err != nil {
		return zero, absent, nil, err
	}
	b, err := p.ReadFileLimit(submissionOutputReceipt, 4096)
	if err != nil {
		return zero, absent, nil, fmt.Errorf("fetch checkpoint output before resume: %w", err)
	}
	var r submissionReceipt
	if err := json.Unmarshal(b, &r); err != nil {
		return zero, absent, nil, err
	}
	if r.Name != checkpoint.File {
		return zero, absent, nil, fmt.Errorf("download is not a checkpoint")
	}
	if err := r.Ref.Validate(); err != nil {
		return zero, absent, nil, err
	}
	f, err := p.OpenRead(r.Name)
	if err != nil {
		return zero, absent, nil, err
	}
	err = artifact.Copy(ctx, io.Discard, f, r.Ref)
	closeErr := f.Close()
	if err != nil {
		return zero, absent, nil, err
	}
	if closeErr != nil {
		return zero, absent, nil, closeErr
	}
	return m, nodejob.ArtifactRef{SHA256: r.Ref.SHA256, Bytes: r.Ref.Bytes, Kind: "checkpoint"}, func() (io.ReadSeekCloser, error) { return p.OpenRead(r.Name) }, nil
}

func validateResumeSelection(source nodejob.Manifest, selection recruitment.Selection) error {
	if source.DatasetID != selection.Requirements.DatasetID || source.DatasetSelector != selection.Requirements.DatasetSelector {
		return fmt.Errorf("resume requires original dataset identity and selector")
	}
	if len(source.Members) != len(selection.Selected) {
		return fmt.Errorf("resume requires original world size")
	}
	for i, n := range selection.Selected {
		if source.Members[i].Node != n.Capabilities.Node {
			return fmt.Errorf("resume requires original ordered nodes; no replacement workers")
		}
	}
	return nil
}

func applySubmissionResume(ctx context.Context, dir, controller string, o *clusterapp.RecruitmentOptions) (string, error) {
	m, ref, open, err := loadSubmissionResume(ctx, dir, controller)
	if err != nil {
		return "", err
	}
	if m.BuildID != o.NumericalPlan.BuildID || m.ConfigHash != o.NumericalPlan.ConfigHash || m.ProgramHash != o.NumericalPlan.ProgramHash || m.WeightLayoutHash != o.NumericalPlan.WeightLayoutHash || m.OptimizerHash != o.NumericalPlan.OptimizerHash {
		return "", fmt.Errorf("exact resume requires the same build/config/program/layout/optimizer")
	}
	if o.CheckpointAt > 0 && o.CheckpointAt <= m.CheckpointAt {
		return "", fmt.Errorf("checkpoint-at must advance the original attempt")
	}
	o.ResumeSource, o.ResumeArtifact, o.OpenResume = &m, &ref, open
	return m.Membership.RunID, nil
}
