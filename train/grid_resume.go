package train

import (
	"fmt"
	"math"
	"slices"

	"github.com/mrothroc/mixlab/data"
)

// Grid state extends the common resume bundle; model, optimizer and scheduler
// tensors retain their existing format and restore path.
type gridResumeState struct {
	Version     int                   `json:"version"`
	Sampler     data.GridSamplerState `json:"sampler"`
	Trainable   []string              `json:"trainable"`
	BestRMSE    *float64              `json:"best_rmse,omitempty"`
	LastValLoss *float64              `json:"last_val_loss,omitempty"`
	FirstLoss   float64               `json:"first_loss"`
	LastLoss    float64               `json:"last_loss"`
	ValEvery    int                   `json:"val_every"`
	Stopped     bool                  `json:"stopped"`
}

func resumeTrainingDatasetHash(cfg *ArchConfig, path string) (string, error) {
	if cfg.GridEnabled() {
		return data.GridDatasetIdentity(path)
	}
	return trainingDatasetHash(path)
}

func restoreGridRunState(saved resumeManifest, sampler *data.GridSampler, shapes []WeightShape, batchSize, valEvery int, result *TrainResult) (float64, bool, error) {
	s := saved.Grid
	if s == nil || s.Version != 1 {
		return 0, false, fmt.Errorf("checkpoint has no supported grid replay state")
	}
	if !slices.Equal(s.Trainable, gridTrainableNames(shapes)) {
		return 0, false, fmt.Errorf("grid checkpoint trainable weights do not match")
	}
	if s.ValEvery != valEvery {
		return 0, false, fmt.Errorf("grid checkpoint validation cadence does not match")
	}
	if err := sampler.Restore(s.Sampler, saved.GlobalStep, batchSize); err != nil {
		return 0, false, err
	}
	best := math.Inf(1)
	for _, metric := range []*float64{s.BestRMSE, s.LastValLoss, &s.FirstLoss, &s.LastLoss} {
		if metric != nil && (math.IsNaN(*metric) || math.IsInf(*metric, 0) || *metric < 0) {
			return 0, false, fmt.Errorf("invalid grid checkpoint metric")
		}
	}
	if (s.BestRMSE == nil) != (s.LastValLoss == nil) {
		return 0, false, fmt.Errorf("incomplete grid checkpoint validation state")
	}
	if s.BestRMSE != nil {
		best = *s.BestRMSE
		result.HasValLoss = true
		result.LastValLoss = *s.LastValLoss
	}
	result.FirstLoss, result.LastLoss = s.FirstLoss, s.LastLoss
	return best, s.Stopped, nil
}

func snapshotGridRun(sampler *data.GridSampler, shapes []WeightShape, best float64, result TrainResult, valEvery int, stopped bool) *gridResumeState {
	s := &gridResumeState{Version: 1, Sampler: sampler.Snapshot(), Trainable: gridTrainableNames(shapes), FirstLoss: result.FirstLoss, LastLoss: result.LastLoss, ValEvery: valEvery, Stopped: stopped}
	if result.HasValLoss {
		s.BestRMSE = &best
		val := result.LastValLoss
		s.LastValLoss = &val
	}
	return s
}
