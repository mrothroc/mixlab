package train

import (
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/data"
)

type gridMetrics struct {
	SSE, UnmaskedSSE     float64
	Count, UnmaskedCount int64
}

func (m gridMetrics) MaskedMSE() float64  { return m.SSE / float64(m.Count) }
func (m gridMetrics) MaskedRMSE() float64 { return math.Sqrt(m.MaskedMSE()) }
func (m gridMetrics) String(scale float64) string {
	u := math.Sqrt(m.UnmaskedSSE / float64(m.UnmaskedCount))
	return fmt.Sprintf("masked_rmse=%g unmasked_rmse=%g scaled_masked_rmse=%g scaled_unmasked_rmse=%g count=%d unmasked_count=%d", m.MaskedRMSE(), u, scale*m.MaskedRMSE(), scale*u, m.Count, m.UnmaskedCount)
}
func (m *gridMetrics) add(b data.GridBatch, pred []float32) error {
	if len(pred) != len(b.Targets) {
		return fmt.Errorf("grid prediction size mismatch")
	}
	n := b.Count * b.Geometry.Height * b.Geometry.Width * b.Geometry.TargetChannels
	for j := 0; j < n; j++ {
		d := float64(pred[j]) - float64(b.Targets[j])
		s := d * d
		if math.IsNaN(s) || math.IsInf(s, 0) {
			return fmt.Errorf("non-finite grid prediction")
		}
		m.UnmaskedSSE += s
		m.UnmaskedCount++
		if b.LossMask[j] > 0 {
			m.SSE += s
			m.Count++
		}
	}
	return nil
}
func evaluateGridDataset(cfg *ArchConfig, t GPUTrainer, ds *data.GridDataset) (gridMetrics, error) {
	m := gridMetrics{}
	e, ok := t.(interface {
		EvaluateObjectiveGPUWithOutputs(objectiveBatch, int, int, []string) (float32, error)
	})
	if !ok {
		return m, fmt.Errorf("trainer cannot evaluate grid predictions")
	}
	bs := cfg.Training.BatchSize
	ids := make([]int, bs)
	g := ds.Geometry
	for start := 0; start < ds.Len(); start += bs {
		n := min(bs, ds.Len()-start)
		for j := 0; j < n; j++ {
			ids[j] = start + j
		}
		b, err := ds.ReadBatch(ids[:n], bs)
		if err != nil {
			return m, err
		}
		if _, err = e.EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, bs, 0, []string{"predictions"}); err != nil {
			return m, err
		}
		p, err := readTrainerOutput(t, "predictions", []int{bs, g.Height, g.Width, g.TargetChannels})
		if err != nil {
			return m, err
		}
		if err = m.add(b, p); err != nil {
			return m, err
		}
	}
	if m.Count == 0 {
		return m, fmt.Errorf("grid validation split has zero valid target values")
	}
	return m, nil
}
