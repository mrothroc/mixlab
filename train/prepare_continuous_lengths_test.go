package train

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestPrepareContinuousArrayWritesValidLengths(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "features.npy")
	labels := filepath.Join(dir, "labels.tsv")
	lengths := filepath.Join(dir, "lengths.tsv")
	output := filepath.Join(dir, "prepared")
	frames := make([]float32, 4*4)
	for i := range frames {
		frames[i] = float32(i)
	}
	writeNPYFloat32(t, input, []int{4, 4, 1}, frames)
	if err := os.WriteFile(labels, []byte("0\t0\n1\t1\n2\t0\n3\t1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lengths, []byte("0\t1\n1\t2\n2\t3\n3\t4\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runPrepare(PrepareOptions{
		Input: input, Output: output, InputFormat: "continuous", LabelFile: labels,
		LengthFile: lengths, ContinuousModality: "audio", ValSplit: 0,
	}); err != nil {
		t.Fatal(err)
	}
	shards, err := filepath.Glob(filepath.Join(output, "train_*.bin"))
	if err != nil || len(shards) != 1 {
		t.Fatalf("shards=%v err=%v", shards, err)
	}
	shard, err := data.LoadContinuousSequenceShard(shards[0])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(shard.Lengths, []int32{1, 2, 3, 4}) {
		t.Fatalf("continuous lengths=%v", shard.Lengths)
	}
	manifest, err := data.LoadDatasetManifest(filepath.Join(output, data.DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	// frames is the stored footprint: 4 records * record_seq_len 4. The real
	// lengths (1,2,3,4) surface through mean/max_sequence_tokens instead.
	if manifest.Splits["train"].Frames != 16 || manifest.Splits["train"].MaxSequenceTokens != 4 ||
		manifest.Splits["train"].MeanSequenceTokens != 2.5 {
		t.Fatalf("manifest split=%+v", manifest.Splits["train"])
	}
	if manifest.ShardFormat != data.DatasetShardFormatContinuousSequenceV2 {
		t.Fatalf("manifest shard_format=%q want %q", manifest.ShardFormat, data.DatasetShardFormatContinuousSequenceV2)
	}
}
