package recruitment

import (
	"fmt"

	"github.com/mrothroc/mixlab/nodejob"
)

// CheckpointSource is durable successful-job evidence, not resume semantics.
// The trainer still validates the opaque checkpoint against this topology.
func (s *Launch) CheckpointSource(controller string) (nodejob.Manifest, error) {
	_, r, err := s.load()
	if err != nil {
		return nodejob.Manifest{}, err
	}
	if r.Plan.Controller != controller || r.Phase != "succeeded" || len(r.Nodes) == 0 || r.Nodes[0].Manifest == nil {
		return nodejob.Manifest{}, fmt.Errorf("owned successful checkpoint attempt required")
	}
	m := r.Nodes[0].Manifest.Manifest
	if m.CheckpointAt == 0 {
		return nodejob.Manifest{}, fmt.Errorf("source attempt exported weights, not an exact-resume checkpoint")
	}
	return m, nil
}

func (s *Launch) OutputFilename(controller string) (string, error) {
	_, r, err := s.load()
	if err != nil {
		return "", err
	}
	if r.Plan.Controller != controller || r.Phase != "succeeded" || len(r.Nodes) == 0 || r.Nodes[0].Manifest == nil {
		return "", fmt.Errorf("owned successful attempt required")
	}
	if r.Nodes[0].Manifest.Manifest.CheckpointAt > 0 {
		return "checkpoint.mixlab", nil
	}
	return "model.safetensors", nil
}
