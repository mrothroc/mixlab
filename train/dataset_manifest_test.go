package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestValidateDatasetManifestForConfigOptionalAndCompatible(t *testing.T) {
	dir := t.TempDir()
	pattern := filepath.Join(dir, "train_*.bin")
	if manifest, path, err := validateDatasetManifestForConfig(objectiveTestConfig(), pattern); err != nil || manifest != nil || path != "" {
		t.Fatalf("legacy manifest lookup manifest=%+v path=%q err=%v", manifest, path, err)
	}

	cfg := objectiveTestConfig()
	writeTrainingDatasetManifest(t, dir, cfg.VocabSize)
	manifest, path, err := validateDatasetManifestForConfig(cfg, pattern)
	if err != nil {
		t.Fatal(err)
	}
	if manifest == nil || manifest.Modality != "text" || path != filepath.Join(dir, data.DatasetManifestFilename) {
		t.Fatalf("manifest=%+v path=%q", manifest, path)
	}
}

func TestValidateDatasetManifestForConfigRejectsVocabMismatch(t *testing.T) {
	dir := t.TempDir()
	writeTrainingDatasetManifest(t, dir, 31)
	_, _, err := validateDatasetManifestForConfig(objectiveTestConfig(), filepath.Join(dir, "train_*.bin"))
	if err == nil || !strings.Contains(err.Error(), "dataset vocab_size=31 does not match model vocab_size=32") {
		t.Fatalf("error=%v", err)
	}
}

func TestRunTrainRejectsDatasetManifestBeforeTrainerConstruction(t *testing.T) {
	dir := t.TempDir()
	writeTrainingDatasetManifest(t, dir, 31)
	cfg := objectiveTestConfig()
	_, err := runTrain(cfg, filepath.Join(dir, "train_*.bin"), TrainOptions{})
	if err == nil || !strings.Contains(err.Error(), "dataset vocab_size=31 does not match model vocab_size=32") {
		t.Fatalf("error=%v", err)
	}
}

func TestConfigureNucleotideDatasetEnablesPackedSegmentTrainingIR(t *testing.T) {
	dir := t.TempDir()
	writeNucleotideDatasetContract(t, dir, data.NucleotideAlphabetDNA)
	cfg := objectiveTestConfig()
	cfg.VocabSize = 9
	cfg.SeqLen = 8
	cfg.Training.BatchTokens = 8
	cfg.Training.Objective = arch.ObjectiveMLM
	cfg.Training.MLMMaskTokenID = 3
	cfg.Training.ReverseComplementProb = 0.5
	if err := configureDatasetForTraining(cfg, filepath.Join(dir, "train_*.bin"), cfg.Name); err != nil {
		t.Fatalf("configureDatasetForTraining: %v", err)
	}
	if !cfg.Training.DatasetSequencePacking || !cfg.Training.AttentionSegmentMaskEnabled() || len(cfg.Training.DatasetNucleotideComplement) != 9 {
		t.Fatalf("runtime training metadata=%+v", cfg.Training)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: arch.ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if !testProgramDeclaresInput(prog, "segment_ids") || !testProgramDeclaresInput(prog, "loss_mask") {
		t.Fatal("nucleotide training IR must consume segment_ids and loss_mask")
	}
}

func TestConfigureTextRecordDatasetEnablesMaskedCausalLossWithoutSegmentMask(t *testing.T) {
	dir := t.TempDir()
	writeTextRecordDatasetContract(t, dir, 32, 8)
	cfg := objectiveTestConfig()
	cfg.SeqLen = 8
	cfg.Training.BatchTokens = 16
	cfg.Training.Objective = arch.ObjectiveCausal
	if err := configureDatasetForTraining(cfg, filepath.Join(dir, "train_*.bin"), cfg.Name); err != nil {
		t.Fatal(err)
	}
	if !cfg.Training.RecordFramingEnabled() || cfg.Training.DatasetSequencePacking || cfg.Training.AttentionSegmentMaskEnabled() {
		t.Fatalf("runtime framing metadata=%+v", cfg.Training)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: arch.ObjectiveCausal})
	if err != nil {
		t.Fatal(err)
	}
	if !testProgramDeclaresInput(prog, "loss_mask") {
		t.Fatal("record-framed causal IR must consume loss_mask")
	}
	if testProgramDeclaresInput(prog, "segment_ids") {
		t.Fatal("one-record-per-row causal IR must not require segment_ids")
	}
}

func TestConfigureTextRecordDatasetRejectsIncompatibleConfig(t *testing.T) {
	dir := t.TempDir()
	writeTextRecordDatasetContract(t, dir, 32, 8)
	tests := []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{name: "seq len", edit: func(c *ArchConfig) { c.SeqLen = 7 }, want: "does not match"},
		{name: "objective", edit: func(c *ArchConfig) { c.Training.Objective = arch.ObjectiveMLM }, want: "supports training.objective"},
		{name: "segment mask", edit: func(c *ArchConfig) { c.Training.AttentionSegmentMask = arch.AttentionSegmentMaskBoundaryToken }, want: "unnecessary"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := objectiveTestConfig()
			cfg.SeqLen = 8
			cfg.Training.BatchTokens = 16
			tt.edit(cfg)
			err := configureDatasetForTraining(cfg, filepath.Join(dir, "train_*.bin"), cfg.Name)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}
}

func TestConfigureClassificationDatasetValidatesTaskContract(t *testing.T) {
	dir := t.TempDir()
	writeClassificationDatasetContract(t, dir, 32, 8, 3)
	cfg := objectiveTestConfig()
	cfg.SeqLen = 8
	cfg.Training.BatchTokens = 16
	cfg.Training.Objective = arch.ObjectiveClassification
	cfg.Training.Classification = &arch.ClassificationSpec{NumLabels: 3}
	if err := configureDatasetForTraining(cfg, filepath.Join(dir, "train_*.bin"), cfg.Name); err != nil {
		t.Fatal(err)
	}
	if !cfg.Training.DatasetRecordFraming || !cfg.Training.DatasetClassification || cfg.Training.DatasetNumLabels != 3 {
		t.Fatalf("runtime classification metadata=%+v", cfg.Training)
	}

	mismatch := *cfg
	mismatch.Training.DatasetRecordFraming = false
	mismatch.Training.DatasetClassification = false
	mismatch.Training.DatasetNumLabels = 0
	mismatch.Training.Classification = &arch.ClassificationSpec{NumLabels: 2}
	err := configureDatasetForTraining(&mismatch, filepath.Join(dir, "train_*.bin"), mismatch.Name)
	if err == nil || !strings.Contains(err.Error(), "task.num_labels=3") {
		t.Fatalf("error=%v", err)
	}
}

func testProgramDeclaresInput(prog *Program, name string) bool {
	for _, input := range prog.Inputs {
		if input.Name == name {
			return true
		}
	}
	return false
}

func TestConfigureNucleotideDatasetRejectsReverseComplementForRNA(t *testing.T) {
	dir := t.TempDir()
	writeNucleotideDatasetContract(t, dir, data.NucleotideAlphabetRNA)
	cfg := objectiveTestConfig()
	cfg.VocabSize = 9
	cfg.Training.Objective = arch.ObjectiveCausal
	cfg.Training.ReverseComplementProb = 0.5
	err := configureDatasetForTraining(cfg, filepath.Join(dir, "train_*.bin"), cfg.Name)
	if err == nil || !strings.Contains(err.Error(), "DNA-only") {
		t.Fatalf("error=%v", err)
	}
}

func writeNucleotideDatasetContract(t *testing.T, dir, alphabet string) {
	t.Helper()
	base := "T"
	complement := map[string]string{"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
	if alphabet == data.NucleotideAlphabetRNA {
		base = "U"
		complement = map[string]string{"A": "U", "C": "G", "G": "C", "U": "A", "N": "N"}
	}
	vocab := data.NucleotideVocabulary{
		Format: data.NucleotideVocabularyFormat, Version: data.NucleotideVocabularyVersion,
		Alphabet: alphabet, InvalidSymbolPolicy: "error", AmbiguousSymbols: []string{"N"},
		Tokens:      map[string]int{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<MASK>": 3, "A": 4, "C": 5, "G": 6, base: 7, "N": 8},
		Complements: complement,
	}
	blob, err := json.Marshal(vocab)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "nucleotide_vocab.json"), blob, 0o644); err != nil {
		t.Fatal(err)
	}
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens, Modality: "nucleotide", VocabSize: 9,
		TokenDType: data.DatasetTokenDTypeUint16, ShardFormat: data.DatasetShardFormatSequenceV1,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2, "mask": 3},
		Artifacts:       data.DatasetManifestArtifacts{Vocabulary: "nucleotide_vocab.json"},
		Splits:          map[string]data.DatasetSplit{"train": {Pattern: "train_*.bin", Tokens: 4, Shards: 1, Sequences: 1}},
	}
	blob, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTrainingDatasetManifest(t *testing.T, dir string, vocabSize int) {
	t.Helper()
	manifest := data.DatasetManifest{
		Format:         data.DatasetManifestFormat,
		Version:        data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens,
		Modality:       "text",
		VocabSize:      vocabSize,
		TokenDType:     data.DatasetTokenDTypeUint16,
		ShardFormat:    data.DatasetShardFormatTokenStreamV1,
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin", Tokens: 10, Shards: 1},
		},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTextRecordDatasetContract(t *testing.T, dir string, vocabSize, seqLen int) {
	t.Helper()
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens, Modality: "text", VocabSize: vocabSize,
		TokenDType: data.DatasetTokenDTypeUint16, ShardFormat: data.DatasetShardFormatSequenceV1,
		SequenceLayout: data.DatasetSequenceLayoutOneRecordRow, RecordSeqLen: seqLen,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2},
		Artifacts:       data.DatasetManifestArtifacts{Tokenizer: "tokenizer.json"},
		Splits: map[string]data.DatasetSplit{"train": {
			Pattern: "train_*.bin", Tokens: 4, Shards: 1, Sequences: 1, MaxSequenceTokens: 4,
		}},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeClassificationDatasetContract(t *testing.T, dir string, vocabSize, seqLen, numLabels int) {
	t.Helper()
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens, Modality: "text", VocabSize: vocabSize,
		TokenDType: data.DatasetTokenDTypeUint16, ShardFormat: data.DatasetShardFormatLabeledSequenceV1,
		SequenceLayout: data.DatasetSequenceLayoutOneRecordRow, RecordSeqLen: seqLen,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2},
		Artifacts:       data.DatasetManifestArtifacts{Tokenizer: "tokenizer.json"},
		Task:            &data.DatasetTask{Type: data.DatasetTaskSingleLabelClassification, NumLabels: numLabels},
		Splits: map[string]data.DatasetSplit{"train": {
			Pattern: "train_*.bin", Tokens: 8, Shards: 1, Sequences: 2, MaxSequenceTokens: 4,
			ClassCounts: map[string]int64{"0": 1, "1": 1},
		}},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}
