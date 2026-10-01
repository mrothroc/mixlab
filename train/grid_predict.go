package train

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

type PredictGridOptions struct{ ConfigPath, SafetensorsLoad, Input, Output, Split string }

// visitGridPredictions uses a prediction-only program: no optimizer, target
// input, loss computation or mutation of checkpoint weights.
func visitGridPredictions(cfg *ArchConfig, checkpoint string, ds *data.GridDataset, visit func(data.GridBatch, []float32) error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if checkpoint == "" {
		return fmt.Errorf("-safetensors-load is required")
	}
	if !cfg.GridEnabled() {
		return fmt.Errorf("grid prediction requires input_adapter.kind=grid")
	}
	if err := checkGridGeometry(cfg, ds.Geometry, false); err != nil {
		return err
	}
	if _, err := configureMLXMemoryLimits(cfg.Name); err != nil {
		return err
	}
	p, err := arch.BuildGridIRProgram(cfg, false)
	if err != nil {
		return err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return err
	}
	w, _, err := loadGridWeights(checkpoint, cfg, shapes, nil)
	if err != nil {
		return err
	}
	handles, err := uploadWeightHandles(shapes, w)
	if err != nil {
		return err
	}
	defer gpu.FreeHandles(handles)
	program, err := gpu.LowerIRProgram(p)
	if err != nil {
		return err
	}
	defer program.Destroy()
	bs := cfg.Training.BatchSize
	ids := make([]int, bs)
	for start := 0; start < ds.Len(); start += bs {
		n := min(bs, ds.Len()-start)
		for j := 0; j < n; j++ {
			ids[j] = start + j
		}
		b, err := ds.ReadBatch(ids[:n], bs)
		if err != nil {
			return err
		}
		inputs, err := makeGridInputs(p.Inputs, &b)
		if err != nil {
			return err
		}
		out, err := gpu.EvalProgramOutput(program, handles, inputs, "predictions")
		if err != nil {
			return err
		}
		for _, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("non-finite grid prediction")
			}
		}
		if err = visit(b, out); err != nil {
			return err
		}
	}
	return nil
}

func RunPredictGrid(opts PredictGridOptions) error {
	if opts.ConfigPath == "" || opts.SafetensorsLoad == "" || opts.Input == "" || opts.Output == "" {
		return fmt.Errorf("predict-grid requires -config, -safetensors-load, -grid-in and -grid-out")
	}
	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	if !cfg.GridEnabled() {
		return fmt.Errorf("predict-grid requires a grid model")
	}
	split := opts.Split
	if split == "" {
		split = "predict"
	}
	ds, err := data.OpenGridDataset(opts.Input, split)
	if err != nil {
		return err
	}
	defer func() { _ = ds.Close() }()
	// Publish a complete directory atomically, never overwrite prior predictions.
	if err = os.MkdirAll(filepath.Dir(opts.Output), 0755); err != nil {
		return err
	}
	if _, err = os.Lstat(opts.Output); !os.IsNotExist(err) {
		return fmt.Errorf("grid output must not already exist: %s", opts.Output)
	}
	tmp, err := os.MkdirTemp(filepath.Dir(opts.Output), ".grid-predict-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(tmp) }()
	type record struct {
		ID   string `json:"id"`
		File string `json:"file"`
	}
	index := []record{}
	g := cfg.InputAdapter
	ct := cfg.DenseRegression.TargetChannels
	p := g.Height * g.Width
	err = visitGridPredictions(cfg, opts.SafetensorsLoad, ds, func(b data.GridBatch, values []float32) error {
		if len(values) != b.BatchSize*p*ct {
			return fmt.Errorf("grid output size mismatch")
		}
		chw := make([]float32, p*ct)
		for row := 0; row < b.Count; row++ {
			for c := 0; c < ct; c++ {
				for j := 0; j < p; j++ {
					chw[c*p+j] = values[(row*p+j)*ct+c]
				}
			}
			name := fmt.Sprintf("%08d.npy", len(index))
			if err := writeGridNPY(filepath.Join(tmp, name), []int{ct, g.Height, g.Width}, chw); err != nil {
				return err
			}
			index = append(index, record{b.IDs[row], name})
		}
		return nil
	})
	if err != nil {
		return err
	}
	meta, err := json.MarshalIndent(struct {
		Format  string   `json:"format"`
		Units   string   `json:"units"`
		Records []record `json:"records"`
	}{"mixlab.grid_predictions.v1", "model", index}, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(tmp, "predictions.json"), append(meta, '\n'), 0644); err != nil {
		return err
	}
	return publishGridDirectory(tmp, opts.Output)
}

func writeGridNPY(path string, shape []int, values []float32) error {
	if shapeProduct(shape) != len(values) {
		return fmt.Errorf("npy shape mismatch")
	}
	parts := make([]string, len(shape))
	for j, n := range shape {
		parts[j] = fmt.Sprint(n)
	}
	header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%s,), }", strings.Join(parts, ", "))
	header += strings.Repeat(" ", (64-(10+len(header)+1)%64)%64) + "\n"
	buf := make([]byte, 10+len(header)+4*len(values))
	copy(buf, "\x93NUMPY\x01\x00")
	binary.LittleEndian.PutUint16(buf[8:10], uint16(len(header)))
	copy(buf[10:], header)
	for j, v := range values {
		binary.LittleEndian.PutUint32(buf[10+len(header)+4*j:], math.Float32bits(v))
	}
	return os.WriteFile(path, buf, 0644)
}

func runGridEval(cfg *ArchConfig, manifest, checkpoint string) error {
	ds, err := data.OpenGridDataset(manifest, "val")
	if err != nil {
		return err
	}
	defer func() { _ = ds.Close() }()
	if err = checkGridGeometry(cfg, ds.Geometry, true); err != nil {
		return err
	}
	m := gridMetrics{}
	if err = visitGridPredictions(cfg, checkpoint, ds, m.add); err != nil {
		return err
	}
	if m.Count == 0 {
		return fmt.Errorf("grid validation split has zero valid target values")
	}
	fmt.Println(m.String(cfg.DenseRegression.EffectiveMetricScale()))
	return nil
}
