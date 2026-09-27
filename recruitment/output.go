package recruitment

import "fmt"

// OutputOwner resolves rank zero from completed durable evidence, not discovery
// or caller-supplied IDs. It does not restart or change the attempt.
func (s *Launch) OutputOwner(controller string) (Candidate, string, error) {
	_, r, err := s.load()
	if err != nil {
		return Candidate{}, "", err
	}
	if r.Plan.Controller != controller || r.Phase != "succeeded" || len(r.Nodes) == 0 || r.Nodes[0].Manifest == nil {
		return Candidate{}, "", fmt.Errorf("owned successful launch required for output retrieval")
	}
	n := r.Nodes[0]
	return n.Candidate, n.Manifest.Manifest.Job, nil
}
