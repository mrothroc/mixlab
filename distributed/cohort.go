package distributed

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

const MaxLearners = 64

// LearnerMember is stable training identity, not a hosting attempt or MLX rank.
type LearnerMember struct {
	WorkerID           string `json:"worker_id"`
	ParticipantID      string `json:"participant_id"`
	IslandID           string `json:"island_id"`
	DatasetPartitionID string `json:"dataset_partition_id"`
	InnerGroupHash     string `json:"inner_group_hash"`
}

type LearnerCohortMembership struct {
	RunID           string          `json:"run_id"`
	Generation      uint64          `json:"generation"`
	OrderedLearners []LearnerMember `json:"ordered_learners"`
	CohortHash      string          `json:"cohort_hash"`
}

func identityText(s string) bool {
	return len(s) > 0 && len(s) <= 128 && strings.IndexFunc(s, func(r rune) bool { return r < 33 || r > 126 }) < 0
}

func NewLearnerCohort(runID string, generation uint64, members []LearnerMember) (LearnerCohortMembership, error) {
	m := LearnerCohortMembership{RunID: runID, Generation: generation, OrderedLearners: append([]LearnerMember(nil), members...)}
	if !identityText(runID) || len(members) < 1 || len(members) > MaxLearners {
		return m, fmt.Errorf("invalid learner cohort")
	}
	workers, participants, partitions := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, v := range members {
		if !identityText(v.WorkerID) || !identityText(v.ParticipantID) || !identityText(v.IslandID) || !identityText(v.DatasetPartitionID) || v.InnerGroupHash != "" || workers[v.WorkerID] || participants[v.ParticipantID] || partitions[v.DatasetPartitionID] {
			return m, fmt.Errorf("R2 requires unique workers, participants and partitions with no inner group")
		}
		workers[v.WorkerID] = true
		participants[v.ParticipantID] = true
		partitions[v.DatasetPartitionID] = true
	}
	b, err := json.Marshal(struct {
		Version    string          `json:"version"`
		RunID      string          `json:"run_id"`
		Generation uint64          `json:"generation"`
		Members    []LearnerMember `json:"members"`
	}{"mixlab-learner-cohort-v1", runID, generation, m.OrderedLearners})
	if err != nil {
		return m, err
	}
	h := sha256.Sum256(b)
	m.CohortHash = hex.EncodeToString(h[:])
	return m, nil
}

func (m LearnerCohortMembership) Canonical() (LearnerCohortMembership, error) {
	v, err := NewLearnerCohort(m.RunID, m.Generation, m.OrderedLearners)
	if err != nil {
		return v, err
	}
	if v.CohortHash != m.CohortHash {
		return v, fmt.Errorf("learner cohort hash mismatch")
	}
	return v, nil
}
