package train

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// configureDatasetForTraining validates representation metadata and attaches
// runtime-only sequence packing/augmentation state before IR construction.
func configureDatasetForTraining(cfg *ArchConfig, shardPattern, name string) error {
	manifest, manifestPath, err := validateDatasetManifestForConfig(cfg, shardPattern)
	if err != nil {
		return err
	}
	if manifest == nil {
		if cfg.ClassificationEnabled() {
			return fmt.Errorf("config %q training.objective=%q requires a labeled mixlab.dataset.json", cfg.Name, arch.ObjectiveClassification)
		}
		if cfg.Training.ReverseComplementProb > 0 {
			return fmt.Errorf("config %q training.reverse_complement_prob requires a nucleotide mixlab.dataset.json", cfg.Name)
		}
		return nil
	}
	fmt.Printf("  [%s] dataset manifest: modality=%s representation=%s vocab_size=%d (%s)\n",
		name, manifest.Modality, manifest.Representation, manifest.VocabSize, manifestPath)
	if manifest.ShardFormat != data.DatasetShardFormatSequenceV1 && manifest.ShardFormat != data.DatasetShardFormatLabeledSequenceV1 {
		if cfg.ClassificationEnabled() {
			return fmt.Errorf("config %q training.objective=%q requires shard_format=%q", cfg.Name, arch.ObjectiveClassification, data.DatasetShardFormatLabeledSequenceV1)
		}
		if cfg.Training.ReverseComplementProb > 0 {
			return fmt.Errorf("config %q training.reverse_complement_prob requires shard_format=%q", cfg.Name, data.DatasetShardFormatSequenceV1)
		}
		return nil
	}
	if manifest.EffectiveSequenceLayout() == data.DatasetSequenceLayoutOneRecordRow {
		if cfg.Training.ReverseComplementProb > 0 {
			return fmt.Errorf("config %q training.reverse_complement_prob requires packed nucleotide sequence data", cfg.Name)
		}
		if cfg.SeqLen != manifest.RecordSeqLen {
			return fmt.Errorf("dataset manifest %q record_seq_len=%d does not match config seq_len=%d", manifestPath, manifest.RecordSeqLen, cfg.SeqLen)
		}
		objective := cfg.Training.EffectiveObjective()
		if objective != arch.ObjectiveCausal && objective != arch.ObjectiveClassification {
			return fmt.Errorf("one-record-per-row framing supports training.objective=\"causal\" or \"classification\" in v1; got %q", objective)
		}
		if objective == arch.ObjectiveClassification {
			if manifest.ShardFormat != data.DatasetShardFormatLabeledSequenceV1 || manifest.Task == nil {
				return fmt.Errorf("config %q classification requires labeled sequence shards with task metadata", cfg.Name)
			}
			if manifest.Task.NumLabels != cfg.Training.Classification.NumLabels {
				return fmt.Errorf("dataset manifest %q task.num_labels=%d does not match training.classification.num_labels=%d", manifestPath, manifest.Task.NumLabels, cfg.Training.Classification.NumLabels)
			}
			cfg.Training.DatasetClassification = true
			cfg.Training.DatasetNumLabels = manifest.Task.NumLabels
		} else if manifest.ShardFormat == data.DatasetShardFormatLabeledSequenceV1 {
			return fmt.Errorf("labeled sequence dataset %q requires training.objective=%q", manifestPath, arch.ObjectiveClassification)
		}
		if cfg.Training.ExampleFramingEnabled() {
			return fmt.Errorf("config %q training.example_framing conflicts with manifest-backed one-record-per-row framing", cfg.Name)
		}
		if cfg.Training.EffectiveAttentionSegmentMask() != "" {
			return fmt.Errorf("config %q training.attention_segment_mask is unnecessary and unsupported with one-record-per-row framing", cfg.Name)
		}
		cfg.Training.DatasetRecordFraming = true
		cfg.Training.DatasetPADID = manifest.SpecialTokenIDs["pad"]
		cfg.Training.DatasetBOSID = manifest.SpecialTokenIDs["bos"]
		cfg.Training.DatasetEOSID = manifest.SpecialTokenIDs["eos"]
		fmt.Printf("  [%s] per-record framing: seq_len=%d bos_id=%d eos_id=%d pad_id=%d\n",
			name, manifest.RecordSeqLen, cfg.Training.DatasetBOSID, cfg.Training.DatasetEOSID, cfg.Training.DatasetPADID)
		if cfg.Training.DatasetClassification {
			fmt.Printf("  [%s] classification dataset: num_labels=%d pooling=%s\n",
				name, cfg.Training.DatasetNumLabels, cfg.EffectiveClassificationPooling())
		}
		return nil
	}
	if cfg.ClassificationEnabled() {
		return fmt.Errorf("config %q classification requires sequence_layout=%q", cfg.Name, data.DatasetSequenceLayoutOneRecordRow)
	}
	if manifest.Modality != "nucleotide" {
		return fmt.Errorf("dataset manifest %q uses record-oriented sequence shards for unsupported modality %q", manifestPath, manifest.Modality)
	}
	if mode := cfg.Training.EffectiveAttentionSegmentMask(); mode != "" && mode != arch.AttentionSegmentMaskBoundaryToken {
		return fmt.Errorf("config %q training.attention_segment_mask=%q conflicts with manifest-backed sequence packing; omit it", cfg.Name, mode)
	}
	if strings.TrimSpace(manifest.Artifacts.Vocabulary) == "" {
		return fmt.Errorf("nucleotide dataset manifest %q is missing artifacts.vocabulary", manifestPath)
	}
	vocabPath := filepath.Join(filepath.Dir(manifestPath), manifest.Artifacts.Vocabulary)
	vocab, err := data.LoadNucleotideVocabulary(vocabPath)
	if err != nil {
		return err
	}
	if vocab.Size() != cfg.VocabSize {
		return fmt.Errorf("nucleotide vocabulary %q size=%d does not match model vocab_size=%d", vocabPath, vocab.Size(), cfg.VocabSize)
	}
	for _, name := range []string{"pad", "bos", "eos", "mask"} {
		artifactID, ok := vocab.SpecialTokenID(name)
		if !ok {
			return fmt.Errorf("nucleotide vocabulary %q is missing %s token", vocabPath, name)
		}
		manifestID, ok := manifest.SpecialTokenIDs[name]
		if !ok || manifestID != artifactID {
			return fmt.Errorf("nucleotide special token %s mismatch: manifest=%d present=%t vocabulary=%d", name, manifestID, ok, artifactID)
		}
	}
	maskID, _ := vocab.SpecialTokenID("mask")
	objective := cfg.Training.EffectiveObjective()
	if objective == arch.ObjectiveMLM || objective == arch.ObjectiveMNTP || objective == arch.ObjectiveHybrid {
		if cfg.Training.MLMMaskTokenID != maskID {
			return fmt.Errorf("config %q training.mlm_mask_token_id=%d does not match nucleotide MASK id %d", cfg.Name, cfg.Training.MLMMaskTokenID, maskID)
		}
	}
	padID, _ := vocab.SpecialTokenID("pad")
	bosID, _ := vocab.SpecialTokenID("bos")
	eosID, _ := vocab.SpecialTokenID("eos")
	cfg.Training.DatasetSequencePacking = true
	cfg.Training.DatasetPADID = padID
	cfg.Training.DatasetBOSID = bosID
	cfg.Training.DatasetEOSID = eosID
	cfg.Training.AttentionSegmentBoundaryTokenID = bosID
	cfg.Training.DatasetNucleotideAlphabet = vocab.Alphabet
	cfg.Training.DatasetNucleotideVocabSource = vocabPath
	cfg.Training.DatasetTokenEligible = make([]uint8, vocab.Size())
	for token, id := range vocab.Tokens {
		if !strings.HasPrefix(token, "<") {
			cfg.Training.DatasetTokenEligible[id] = 1
		}
	}
	if vocab.Alphabet == data.NucleotideAlphabetDNA {
		complement, err := vocab.ComplementTokenIDs()
		if err != nil {
			return err
		}
		cfg.Training.DatasetNucleotideComplement = complement
	} else if cfg.Training.ReverseComplementProb > 0 {
		return fmt.Errorf("config %q training.reverse_complement_prob is DNA-only; dataset alphabet=%q", cfg.Name, vocab.Alphabet)
	}
	if err := arch.ValidateSegmentAttentionCompatibility(cfg, cfg.Name); err != nil {
		return err
	}
	if err := validateNucleotideRuntimeObjective(cfg); err != nil {
		return err
	}
	fmt.Printf("  [%s] nucleotide sequence packing: alphabet=%s bos_id=%d eos_id=%d pad_id=%d vocabulary=%s\n",
		name, vocab.Alphabet, bosID, eosID, padID, vocabPath)
	if cfg.Training.ReverseComplementProb > 0 {
		fmt.Printf("  [%s] DNA reverse-complement augmentation: probability=%g seed=%d\n",
			name, cfg.Training.ReverseComplementProb, cfg.Training.Seed)
	}
	return nil
}

func validateDatasetManifestForConfig(cfg *ArchConfig, shardPattern string) (*data.DatasetManifest, string, error) {
	if cfg == nil {
		return nil, "", fmt.Errorf("nil model config")
	}
	manifest, path, found, err := data.FindDatasetManifest(shardPattern)
	if err != nil {
		return nil, "", err
	}
	if !found {
		return nil, "", nil
	}
	if err := manifest.ValidateModelVocab(cfg.VocabSize); err != nil {
		return nil, "", fmt.Errorf("dataset manifest %q is incompatible with config %q: %w", path, cfg.Name, err)
	}
	return manifest, path, nil
}
