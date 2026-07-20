package train

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/mrothroc/mixlab/gpu"
)

type namedFloatTensor struct {
	Name  string
	Shape []int
	Data  []float32
}

func writeNamedFloatSafetensorsAtomic(path string, tensors []namedFloatTensor, metadata map[string]string) error {
	if len(tensors) == 0 {
		return fmt.Errorf("cannot write empty training state")
	}
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
	header := make(map[string]safetensorHeaderEntry, len(tensors))
	var offset uint64
	for _, tensor := range tensors {
		if tensor.Name == "" || shapeProduct(tensor.Shape) != len(tensor.Data) {
			return fmt.Errorf("invalid state tensor %q shape=%v data=%d", tensor.Name, tensor.Shape, len(tensor.Data))
		}
		addSafetensorHeaderEntry(header, &offset, tensor.Name, "F32", tensor.Shape, uint64(len(tensor.Data)*4))
	}
	headerMap := make(map[string]json.RawMessage, len(header)+1)
	for name, entry := range header {
		blob, err := json.Marshal(entry)
		if err != nil {
			return err
		}
		headerMap[name] = blob
	}
	metaBlob, err := json.Marshal(metadata)
	if err != nil {
		return err
	}
	headerMap["__metadata__"] = metaBlob
	headerBytes, err := marshalSafetensorHeader(headerMap)
	if err != nil {
		return err
	}
	return atomicWriteFile(path, func(f *os.File) error {
		if err := binary.Write(f, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
			return err
		}
		if _, err := f.Write(headerBytes); err != nil {
			return err
		}
		for _, tensor := range tensors {
			if err := writeSafetensorPayloadBytes(f, tensor.Name, encodeFloat32Data(tensor.Data)); err != nil {
				return err
			}
		}
		return nil
	})
}

func atomicWriteJSON(path string, value any) error {
	blob, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	blob = append(blob, '\n')
	return atomicWriteFile(path, func(f *os.File) error {
		_, err := f.Write(blob)
		return err
	})
}

func atomicWriteFile(path string, write func(*os.File) error) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".mixlab-checkpoint-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	ok := false
	defer func() {
		_ = tmp.Close()
		if !ok {
			_ = os.Remove(tmpPath)
		}
	}()
	if err := write(tmp); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	ok = true
	return nil
}

func optimizerStateTensorName(kind gpu.OptimizerStateKind, weightIndex int) string {
	return fmt.Sprintf("optimizer.k%d.w%d", kind, weightIndex)
}

func appendWeightTensors(prefix string, weights [][]float32, shapes []WeightShape, tensors *[]namedFloatTensor, refs *[]resumeTensorRef) error {
	if len(weights) == 0 {
		return nil
	}
	if len(weights) > len(shapes) {
		return fmt.Errorf("%s state weight count=%d exceeds model shapes=%d", prefix, len(weights), len(shapes))
	}
	for i, weight := range weights {
		if len(weight) == 0 {
			continue
		}
		for _, value := range weight {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("%s state weight %d contains non-finite values", prefix, i)
			}
		}
		name := fmt.Sprintf("%s.w%d", prefix, i)
		shape := append([]int(nil), shapes[i].Shape...)
		*tensors = append(*tensors, namedFloatTensor{Name: name, Shape: shape, Data: weight})
		*refs = append(*refs, resumeTensorRef{Name: name, WeightIndex: i, Shape: shape})
	}
	return nil
}

func loadResumeState(manifest resumeManifest) (resumeLoadedState, error) {
	dir := filepath.Dir(manifest.ManifestPath)
	statePath := filepath.Join(dir, manifest.StateFile)
	blobs, err := loadSafetensors(statePath)
	if err != nil {
		return resumeLoadedState{}, err
	}
	trainer := gpu.TrainerStateSnapshot{Optimizer: manifest.Optimizer, Tensors: make([]gpu.TrainerOptimizerStateTensor, 0, len(manifest.OptimizerTensors))}
	for _, ref := range manifest.OptimizerTensors {
		data, err := decodeSafetensorFloat32(ref.Name, ref.Shape, blobs)
		if err != nil {
			return resumeLoadedState{}, err
		}
		trainer.Tensors = append(trainer.Tensors, gpu.TrainerOptimizerStateTensor{
			Kind: ref.Kind, WeightIndex: ref.WeightIndex, Shape: append([]int(nil), ref.Shape...), Data: data,
		})
	}
	loadWeights := func(refs []resumeTensorRef) ([][]float32, error) {
		maxIndex := -1
		for _, ref := range refs {
			if ref.WeightIndex > maxIndex {
				maxIndex = ref.WeightIndex
			}
		}
		if maxIndex < 0 {
			return nil, nil
		}
		out := make([][]float32, maxIndex+1)
		for _, ref := range refs {
			data, err := decodeSafetensorFloat32(ref.Name, ref.Shape, blobs)
			if err != nil {
				return nil, err
			}
			out[ref.WeightIndex] = data
		}
		return out, nil
	}
	swa, err := loadWeights(manifest.SWATensors)
	if err != nil {
		return resumeLoadedState{}, err
	}
	data2vec, err := loadWeights(manifest.Data2VecTensors)
	if err != nil {
		return resumeLoadedState{}, err
	}
	return resumeLoadedState{
		Manifest: manifest, ModelPath: filepath.Join(dir, manifest.ModelFile),
		Trainer: trainer, SWA: swa, Data2Vec: data2vec,
	}, nil
}

func checkpointBundleSize(manifestPath string, manifest resumeManifest) int64 {
	dir := filepath.Dir(manifestPath)
	var total int64
	for _, name := range []string{manifest.ModelFile, manifest.SWAFile, manifest.StateFile} {
		if name == "" {
			continue
		}
		if info, err := os.Stat(filepath.Join(dir, name)); err == nil {
			total += info.Size()
		}
	}
	return total
}

func allFiniteState(snapshot gpu.TrainerStateSnapshot) bool {
	for _, tensor := range snapshot.Tensors {
		for _, value := range tensor.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return false
			}
		}
	}
	return true
}

func restoreSWAWeights(dst *[][]float32, checkpoint [][]float32, shapes []WeightShape, enabled bool) error {
	if len(checkpoint) == 0 {
		return nil
	}
	if !enabled {
		return fmt.Errorf("checkpoint contains SWA state but training.swa_start is disabled")
	}
	if len(checkpoint) != len(shapes) {
		return fmt.Errorf("SWA weight count mismatch: checkpoint=%d model=%d", len(checkpoint), len(shapes))
	}
	restored := make([][]float32, len(shapes))
	for i := range shapes {
		if len(checkpoint[i]) == 0 {
			continue
		}
		want := shapeProduct(shapes[i].Shape)
		if len(checkpoint[i]) != want {
			return fmt.Errorf("SWA weight %d size=%d want=%d", i, len(checkpoint[i]), want)
		}
		restored[i] = append([]float32(nil), checkpoint[i]...)
	}
	*dst = restored
	return nil
}
