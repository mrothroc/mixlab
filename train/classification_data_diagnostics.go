package train

import "fmt"

const defaultClassificationLabelDiversityBatches = 10

// classificationLabelDiversityMonitor checks the batches the trainer actually
// consumes. Global label counts do not catch class-ordered shards: every class
// can be present while each individual optimizer batch is still class-pure.
type classificationLabelDiversityMonitor struct {
	numLabels        int
	sampleLimit      int
	sampled          int
	allSingleLabel   bool
	diagnosticClosed bool
}

func newClassificationLabelDiversityMonitor(numLabels, availableBatches int) *classificationLabelDiversityMonitor {
	if numLabels < 2 || availableBatches <= 0 {
		return nil
	}
	sampleLimit := defaultClassificationLabelDiversityBatches
	if availableBatches < sampleLimit {
		sampleLimit = availableBatches
	}
	return &classificationLabelDiversityMonitor{
		numLabels:      numLabels,
		sampleLimit:    sampleLimit,
		allSingleLabel: true,
	}
}

// observe returns a warning exactly once when every eligible sampled batch is
// single-label. Batches with fewer than two real rows cannot demonstrate low
// diversity and are not counted toward the sample.
func (m *classificationLabelDiversityMonitor) observe(batch trainBatch) string {
	if m == nil || m.diagnosticClosed {
		return ""
	}
	distinct, realRows := distinctTrainingBatchLabels(batch)
	possibleDiversity := realRows
	if m.numLabels < possibleDiversity {
		possibleDiversity = m.numLabels
	}
	if possibleDiversity < 2 {
		return ""
	}
	m.sampled++
	if distinct != 1 {
		m.allSingleLabel = false
	}
	if m.sampled < m.sampleLimit {
		return ""
	}
	m.diagnosticClosed = true
	if !m.allSingleLabel {
		return ""
	}
	return fmt.Sprintf(
		"sampled %d training batches each contained 1 distinct label (of %d); "+
			"records may be class-ordered within shards -- shuffle records and labels together before prepare",
		m.sampled, m.numLabels,
	)
}

func distinctTrainingBatchLabels(batch trainBatch) (distinct, realRows int) {
	seen := make(map[int32]struct{})
	rowLimit := len(batch.labels)
	if batch.batchSize > 0 && batch.batchSize < rowLimit {
		rowLimit = batch.batchSize
	}
	if len(batch.exampleMask) == 0 && batch.exampleCount > 0 && batch.exampleCount < rowLimit {
		rowLimit = batch.exampleCount
	}
	for row := 0; row < rowLimit; row++ {
		if len(batch.exampleMask) > 0 {
			if row >= len(batch.exampleMask) || batch.exampleMask[row] == 0 {
				continue
			}
		}
		seen[batch.labels[row]] = struct{}{}
		realRows++
	}
	return len(seen), realRows
}
