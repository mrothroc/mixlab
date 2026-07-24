package train

import (
	"fmt"
	"math"
	"sort"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// ClassificationMetrics contains native single-label sequence metrics.
type ClassificationMetrics struct {
	Loss     float64
	Accuracy float64
	MCC      float64
	MacroF1  float64
	AUROC    float64
	HasAUROC bool
	Examples int
}

func (m ClassificationMetrics) summary() string {
	s := fmt.Sprintf("loss=%.6f acc=%.4f mcc=%.4f macro_f1=%.4f", m.Loss, m.Accuracy, m.MCC, m.MacroF1)
	if m.HasAUROC {
		s += fmt.Sprintf(" auroc=%.4f", m.AUROC)
	}
	return s
}

type classificationOutputEvaluator interface {
	EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, outputNames []string) (float32, error)
	ReadOutput(name string, shape []int) ([]float32, error)
}

func evaluateClassificationValidation(cfg *ArchConfig, valSet *data.ValSet, trainer GPUTrainer, step, batchSize, seqLen int) (ClassificationMetrics, error) {
	if cfg == nil || !cfg.ClassificationEnabled() || cfg.Training.Classification == nil {
		return ClassificationMetrics{}, fmt.Errorf("classification validation requires an active classification config")
	}
	evaluator, ok := trainer.(classificationOutputEvaluator)
	if !ok {
		return ClassificationMetrics{}, fmt.Errorf("trainer does not support classification logits readback")
	}
	if valSet == nil || len(valSet.Batches) == 0 {
		return ClassificationMetrics{}, fmt.Errorf("no classification validation batches")
	}
	numLabels := cfg.Training.Classification.NumLabels
	labels := make([]int, 0, len(valSet.Batches)*batchSize)
	predictions := make([]int, 0, cap(labels))
	positiveScores := make([]float64, 0, cap(labels))
	lossSum := 0.0
	for batchIndex, vb := range valSet.Batches {
		prepared, err := prepareObjectiveBatchWithSeqLen(cfg, trainBatchFromValBatch(vb), step, arch.ObjectiveClassification, seqLen)
		if err != nil {
			return ClassificationMetrics{}, fmt.Errorf("prepare classification validation batch %d: %w", batchIndex, err)
		}
		loss, err := evaluator.EvaluateObjectiveGPUWithOutputs(prepared, batchSize, seqLen, []string{"classification_logits"})
		if err != nil {
			return ClassificationMetrics{}, fmt.Errorf("evaluate classification validation batch %d: %w", batchIndex, err)
		}
		logits, err := evaluator.ReadOutput("classification_logits", []int{batchSize, numLabels})
		if err != nil {
			return ClassificationMetrics{}, fmt.Errorf("read classification validation logits for batch %d: %w", batchIndex, err)
		}
		if len(logits) != batchSize*numLabels {
			return ClassificationMetrics{}, fmt.Errorf("classification logits size=%d, want %d", len(logits), batchSize*numLabels)
		}
		lossSum += float64(loss) * float64(batchSize)
		for row := 0; row < batchSize; row++ {
			label := int(prepared.classificationLabels[row])
			if label < 0 || label >= numLabels {
				return ClassificationMetrics{}, fmt.Errorf("classification label %d outside [0,%d)", label, numLabels)
			}
			rowLogits := logits[row*numLabels : (row+1)*numLabels]
			pred := 0
			for class := 1; class < numLabels; class++ {
				if rowLogits[class] > rowLogits[pred] {
					pred = class
				}
			}
			labels = append(labels, label)
			predictions = append(predictions, pred)
			if numLabels == 2 {
				positiveScores = append(positiveScores, binarySoftmaxProbability(rowLogits[0], rowLogits[1]))
			}
		}
	}
	metrics := classificationMetricsFromPredictions(labels, predictions, positiveScores, numLabels)
	metrics.Loss = lossSum / float64(len(labels))
	return metrics, nil
}

func binarySoftmaxProbability(negative, positive float32) float64 {
	delta := float64(negative) - float64(positive)
	if delta >= 0 {
		e := math.Exp(-delta)
		return e / (1 + e)
	}
	return 1 / (1 + math.Exp(delta))
}

func classificationMetricsFromPredictions(labels, predictions []int, positiveScores []float64, numLabels int) ClassificationMetrics {
	n := len(labels)
	result := ClassificationMetrics{Examples: n}
	if n == 0 || len(predictions) != n || numLabels < 2 {
		return result
	}
	confusion := make([][]float64, numLabels)
	for i := range confusion {
		confusion[i] = make([]float64, numLabels)
	}
	correct := 0
	for i, label := range labels {
		pred := predictions[i]
		if label < 0 || label >= numLabels || pred < 0 || pred >= numLabels {
			continue
		}
		confusion[label][pred]++
		if label == pred {
			correct++
		}
	}
	result.Accuracy = float64(correct) / float64(n)

	total := float64(n)
	trace := 0.0
	actual := make([]float64, numLabels)
	predicted := make([]float64, numLabels)
	f1Sum := 0.0
	for class := 0; class < numLabels; class++ {
		trace += confusion[class][class]
		for j := 0; j < numLabels; j++ {
			actual[class] += confusion[class][j]
			predicted[class] += confusion[j][class]
		}
		denom := actual[class] + predicted[class]
		if denom > 0 {
			f1Sum += 2 * confusion[class][class] / denom
		}
	}
	result.MacroF1 = f1Sum / float64(numLabels)
	dot, actualSq, predictedSq := 0.0, 0.0, 0.0
	for class := 0; class < numLabels; class++ {
		dot += actual[class] * predicted[class]
		actualSq += actual[class] * actual[class]
		predictedSq += predicted[class] * predicted[class]
	}
	denom := math.Sqrt((total*total - predictedSq) * (total*total - actualSq))
	if denom > 0 {
		result.MCC = (trace*total - dot) / denom
	}
	if numLabels == 2 && len(positiveScores) == n {
		result.AUROC, result.HasAUROC = binaryAUROC(labels, positiveScores)
	}
	return result
}

func binaryAUROC(labels []int, scores []float64) (float64, bool) {
	if len(labels) == 0 || len(labels) != len(scores) {
		return 0, false
	}
	type scoredLabel struct {
		score float64
		label int
	}
	ranked := make([]scoredLabel, len(labels))
	positives := 0
	for i := range labels {
		ranked[i] = scoredLabel{score: scores[i], label: labels[i]}
		if labels[i] == 1 {
			positives++
		}
	}
	negatives := len(labels) - positives
	if positives == 0 || negatives == 0 {
		return 0, false
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].score < ranked[j].score })
	positiveRankSum := 0.0
	for start := 0; start < len(ranked); {
		end := start + 1
		for end < len(ranked) && ranked[end].score == ranked[start].score {
			end++
		}
		averageRank := (float64(start+1) + float64(end)) / 2
		for i := start; i < end; i++ {
			if ranked[i].label == 1 {
				positiveRankSum += averageRank
			}
		}
		start = end
	}
	p := float64(positives)
	n := float64(negatives)
	return (positiveRankSum - p*(p+1)/2) / (p * n), true
}
