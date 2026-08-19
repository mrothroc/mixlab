package train

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

// Prepare-path tests for discrete multi-codebook shards.

func TestPrepareCodebookArrayWithBoundsLengthsAndAtomicLabels(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "codes.npy")
	labels := filepath.Join(dir, "labels.tsv")
	lengths := filepath.Join(dir, "lengths.tsv")
	output := filepath.Join(dir, "prepared")
	values := make([]int32, 6*3*2)
	for i := range values {
		values[i] = int32(i % 8)
	}
	writeNPYInt32(t, input, []int{6, 3, 2}, values)
	if err := os.WriteFile(labels, []byte("0\t0\n1\t1\n2\t0\n3\t1\n4\t0\n5\t1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lengths, []byte("0\t3\n1\t2\n2\t1\n3\t3\n4\t2\n5\t1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runPrepare(PrepareOptions{
		Input: input, Output: output, InputFormat: "codebooks",
		LabelFile: labels, LengthFile: lengths, CodebookVocabSize: 8,
		CodebookModality: "audio", ValSplit: 0.34,
	}); err != nil {
		t.Fatal(err)
	}
	manifest, err := data.LoadDatasetManifest(filepath.Join(output, data.DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Representation != data.DatasetRepresentationDiscreteCodebooks || manifest.Modality != "audio" ||
		manifest.NumCodebooks != 2 || manifest.CodebookVocabSize != 8 || manifest.RecordSeqLen != 3 ||
		manifest.TokenDType != data.DatasetTokenDTypeInt32 {
		t.Fatalf("manifest=%+v", manifest)
	}
	trainShards, err := filepath.Glob(filepath.Join(output, "train_*.bin"))
	if err != nil || len(trainShards) != 1 {
		t.Fatalf("train shards=%v err=%v", trainShards, err)
	}
	shard, err := data.LoadCodebookSequenceShard(trainShards[0])
	if err != nil {
		t.Fatal(err)
	}
	if shard.Records != 4 || shard.SeqLen != 3 || shard.NumCodebooks != 2 || shard.CodebookVocabSize != 8 ||
		!reflect.DeepEqual(shard.Labels, []int32{0, 1, 0, 1}) ||
		!reflect.DeepEqual(shard.Lengths, []int32{3, 2, 1, 3}) ||
		!reflect.DeepEqual(shard.Tokens, values[:24]) {
		t.Fatalf("train shard=%+v", shard)
	}
}

func TestPrepareCodebookArrayRejectsOutOfRangeIDWithCodebook(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "codes.npy")
	labels := filepath.Join(dir, "labels.tsv")
	values := []int32{0, 1, 2, 3, 4, 8, 6, 7}
	writeNPYInt32(t, input, []int{2, 2, 2}, values)
	if err := os.WriteFile(labels, []byte("0\t0\n1\t1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := runPrepare(PrepareOptions{
		Input: input, Output: filepath.Join(dir, "prepared"), InputFormat: "codebooks",
		LabelFile: labels, CodebookVocabSize: 8, CodebookModality: "audio", ValSplit: 0,
	})
	if err == nil || !strings.Contains(err.Error(), "value=8 outside [0,8)") || !strings.Contains(err.Error(), "codebook=1") {
		t.Fatalf("range error=%v", err)
	}
}
