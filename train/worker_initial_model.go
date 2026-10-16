package train

import (
	"fmt"
	"io"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/artifact/tensors"
)

// RunWorkerInitialModel materializes configured float32 initialization once.
// It uses no GPU, dataset, network, or credential state. The resulting ordered
// tensor artifact is the common starting point for all admitted learners.
func RunWorkerInitialModel(in io.Reader, out io.Writer) error {
	b, err := io.ReadAll(io.LimitReader(in, (256<<10)+1))
	if err != nil {
		return err
	}
	p, err := inspectWorkerConfig(b, strings.Repeat("0", 64))
	if err != nil {
		return err
	}
	if p.Learner == nil {
		return fmt.Errorf("initial learner model requires DiLoCo")
	}
	cfg, err := arch.ParseArchConfig(b, "initial learner model")
	if err != nil {
		return err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return err
	}
	if err := validateWorkerInitialModelShapes(shapes); err != nil {
		return err
	}
	w := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	model := make([]LocalLearnerTensor, len(shapes))
	for i, s := range shapes {
		model[i] = LocalLearnerTensor{Name: s.Name, Shape: s.Shape, Data: w[i]}
	}
	return tensors.Write(out, LocalLearnerWireTensors(model))
}

func validateWorkerInitialModelShapes(shapes []WeightShape) error {
	const maxElements = 1 << 27
	var elements uint64
	for _, s := range shapes {
		n := uint64(1)
		if len(s.Shape) == 0 || len(s.Shape) > 8 {
			return fmt.Errorf("invalid initial model shape for %s", s.Name)
		}
		// Check before multiplication: shapeProduct can wrap before the cap check.
		for _, d := range s.Shape {
			if d <= 0 || uint64(d) > maxElements/n {
				return fmt.Errorf("initial model exceeds bounded artifact generation limit")
			}
			n *= uint64(d)
		}
		if elements > maxElements-n {
			return fmt.Errorf("initial model exceeds bounded artifact generation limit")
		}
		elements += n
	}
	return nil
}
