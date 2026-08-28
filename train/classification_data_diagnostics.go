package train

import (
	"fmt"
	"math"
	"strconv"
)

const defaultClassificationLabelDiversityBatches = 10

// classificationLabelDiversityFraction is the share of *attainable* diversity a
// healthy sample is expected to reach. Diversity is measured as the excess over
// one, because a batch always contains at least one label: a threshold of
// "25% of achievable" would demand 2 of 2 labels in every binary batch, which a
// correctly shuffled but imbalanced corpus routinely misses.
const classificationLabelDiversityFraction = 0.25

// classificationLabelDiversityMonitor checks the batches the trainer actually
// consumes. Global label counts do not catch class-ordered shards: every class
// can be present while most optimizer batches still have near-zero diversity.
type classificationLabelDiversityMonitor struct {
	numLabels        int
	sampleLimit      int
	sampled          int
	distinctTotal    int
	achievableTotal  int
	minDistinct      int
	maxDistinct      int
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
		numLabels:   numLabels,
		sampleLimit: sampleLimit,
	}
}

// observe returns a warning exactly once when mean label diversity reaches less
// than a quarter of the diversity the sampled batches could have shown. Batches
// with fewer than two real rows cannot demonstrate low diversity and are not
// counted toward the sample.
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
	m.distinctTotal += distinct
	m.achievableTotal += possibleDiversity
	if m.sampled == 1 || distinct < m.minDistinct {
		m.minDistinct = distinct
	}
	if distinct > m.maxDistinct {
		m.maxDistinct = distinct
	}
	if m.sampled < m.sampleLimit {
		return ""
	}
	m.diagnosticClosed = true
	meanDistinct := float64(m.distinctTotal) / float64(m.sampled)
	meanAchievable := float64(m.achievableTotal) / float64(m.sampled)
	warningThreshold := 1 + classificationLabelDiversityFraction*(meanAchievable-1)
	if meanDistinct >= warningThreshold {
		return ""
	}
	return fmt.Sprintf(
		"sampled %d training batches averaged %.1f distinct labels "+
			"(of %s possible per batch, %d classes; min=%d max=%d); "+
			"records may be class-ordered within shards -- shuffle records and labels "+
			"together before prepare -- or one class may dominate the corpus",
		m.sampled, meanDistinct, formatLabelDiversity(meanAchievable), m.numLabels,
		m.minDistinct, m.maxDistinct,
	)
}

func formatLabelDiversity(value float64) string {
	if value == math.Trunc(value) {
		return strconv.FormatInt(int64(value), 10)
	}
	return strconv.FormatFloat(value, 'f', 1, 64)
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
