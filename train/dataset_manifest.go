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
		if cfg.RCEquivarianceEnabled() {
			return fmt.Errorf("config %q rc_equivariant=true requires a DNA nucleotide mixlab.dataset.json", cfg.Name)
		}
		if cfg.Training.ReverseComplementProb > 0 {
			return fmt.Errorf("config %q training.reverse_complement_prob requires a nucleotide mixlab.dataset.json", cfg.Name)
		}
		return nil
	}
	fmt.Printf("  [%s] dataset manifest: modality=%s representation=%s vocab_size=%d (%s)\n",
		name, manifest.Modality, manifest.Representation, manifest.VocabSize, manifestPath)
	if manifest.ShardFormat == data.DatasetShardFormatTokenStreamV1 && manifest.Modality == "nucleotide" {
		if manifest.EffectiveSequenceLayout() != data.DatasetSequenceLayoutContinuousStream {
			return fmt.Errorf("nucleotide token-stream manifest %q requires sequence_layout=%q", manifestPath, data.DatasetSequenceLayoutContinuousStream)
		}
		if cfg.ClassificationEnabled() {
			return fmt.Errorf("config %q training.objective=%q requires record-oriented labeled nucleotide data", cfg.Name, arch.ObjectiveClassification)
		}
		if !cfg.Training.ExampleFramingEnabled() {
			return fmt.Errorf("config %q continuous nucleotide streams require training.example_framing so recurrent state and causal targets reset at fixed row boundaries", cfg.Name)
		}
		if cfg.Training.EffectiveObjective() != arch.ObjectiveCausal {
			return fmt.Errorf("config %q continuous nucleotide streams with training.example_framing require training.objective=%q", cfg.Name, arch.ObjectiveCausal)
		}
		if cfg.Training.EffectiveAttentionSegmentMask() != "" {
			return fmt.Errorf("config %q continuous nucleotide streams do not use training.attention_segment_mask", cfg.Name)
		}
		vocab, vocabPath, err := configureNucleotideVocabularyForTraining(cfg, manifest, manifestPath)
		if err != nil {
			return err
		}
		bosID, _ := vocab.SpecialTokenID("bos")
		eosID, _ := vocab.SpecialTokenID("eos")
		padID, _ := vocab.SpecialTokenID("pad")
		framing := cfg.Training.ExampleFraming
		if framing.BosID != bosID || framing.EosID != eosID {
			return fmt.Errorf(
				"config %q training.example_framing bos_id/eos_id=%d/%d do not match nucleotide vocabulary ids %d/%d",
				cfg.Name, framing.BosID, framing.EosID, bosID, eosID,
			)
		}
		cfg.Training.DatasetNucleotideStream = true
		cfg.Training.DatasetBOSID = bosID
		cfg.Training.DatasetEOSID = eosID
		cfg.Training.DatasetPADID = padID
		fmt.Printf(
			"  [%s] continuous nucleotide stream: alphabet=%s content_len=%d bos_id=%d eos_id=%d vocabulary=%s\n",
			name, vocab.Alphabet, framing.ContentLen, bosID, eosID, vocabPath,
		)
		if cfg.Training.ReverseComplementProb > 0 {
			fmt.Printf("  [%s] DNA reverse-complement stream augmentation: probability=%g seed=%d\n",
				name, cfg.Training.ReverseComplementProb, cfg.Training.Seed)
		}
		return nil
	}
	if manifest.ShardFormat != data.DatasetShardFormatSequenceV1 && manifest.ShardFormat != data.DatasetShardFormatLabeledSequenceV1 {
		if cfg.ClassificationEnabled() {
			return fmt.Errorf("config %q training.objective=%q requires shard_format=%q", cfg.Name, arch.ObjectiveClassification, data.DatasetShardFormatLabeledSequenceV1)
		}
		if cfg.Training.ReverseComplementProb > 0 {
			return fmt.Errorf("config %q training.reverse_complement_prob requires a nucleotide sequence or continuous-stream dataset", cfg.Name)
		}
		if cfg.RCEquivarianceEnabled() {
			return fmt.Errorf("config %q rc_equivariant=true requires record-oriented nucleotide sequence shards", cfg.Name)
		}
		return nil
	}
	if manifest.EffectiveSequenceLayout() == data.DatasetSequenceLayoutOneRecordRow {
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
		if cfg.Training.ReverseComplementProb > 0 && objective != arch.ObjectiveClassification {
			return fmt.Errorf("config %q training.reverse_complement_prob with one-record-per-row data requires training.objective=%q", cfg.Name, arch.ObjectiveClassification)
		}
		if cfg.Training.ExampleFramingEnabled() {
			return fmt.Errorf("config %q training.example_framing conflicts with manifest-backed one-record-per-row framing", cfg.Name)
		}
		if cfg.Training.EffectiveAttentionSegmentMask() != "" {
			return fmt.Errorf("config %q training.attention_segment_mask is unnecessary and unsupported with one-record-per-row framing", cfg.Name)
		}
		if cfg.RCEquivarianceEnabled() || cfg.Training.ReverseComplementProb > 0 {
			if manifest.Modality != "nucleotide" {
				return fmt.Errorf("config %q reverse-complement features require a nucleotide dataset; got modality=%q", cfg.Name, manifest.Modality)
			}
			if _, _, err := configureNucleotideVocabularyForTraining(cfg, manifest, manifestPath); err != nil {
				return err
			}
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
		if cfg.Training.ReverseComplementProb > 0 {
			fmt.Printf("  [%s] DNA reverse-complement augmentation: probability=%g seed=%d\n",
				name, cfg.Training.ReverseComplementProb, cfg.Training.Seed)
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
	vocab, vocabPath, err := configureNucleotideVocabularyForTraining(cfg, manifest, manifestPath)
	if err != nil {
		return err
	}
	padID, _ := vocab.SpecialTokenID("pad")
	bosID, _ := vocab.SpecialTokenID("bos")
	eosID, _ := vocab.SpecialTokenID("eos")
	cfg.Training.DatasetSequencePacking = true
	cfg.Training.DatasetPADID = padID
	cfg.Training.DatasetBOSID = bosID
	cfg.Training.DatasetEOSID = eosID
	cfg.Training.AttentionSegmentBoundaryTokenID = bosID
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

func configureNucleotideVocabularyForTraining(
	cfg *ArchConfig,
	manifest *data.DatasetManifest,
	manifestPath string,
) (*data.NucleotideVocabulary, string, error) {
	if strings.TrimSpace(manifest.Artifacts.Vocabulary) == "" {
		return nil, "", fmt.Errorf("nucleotide dataset manifest %q is missing artifacts.vocabulary", manifestPath)
	}
	vocabPath := filepath.Join(filepath.Dir(manifestPath), manifest.Artifacts.Vocabulary)
	vocab, err := data.LoadNucleotideVocabulary(vocabPath)
	if err != nil {
		return nil, "", err
	}
	if vocab.Size() != cfg.VocabSize {
		return nil, "", fmt.Errorf("nucleotide vocabulary %q size=%d does not match model vocab_size=%d", vocabPath, vocab.Size(), cfg.VocabSize)
	}
	for _, name := range []string{"pad", "bos", "eos", "mask"} {
		artifactID, ok := vocab.SpecialTokenID(name)
		if !ok {
			return nil, "", fmt.Errorf("nucleotide vocabulary %q is missing %s token", vocabPath, name)
		}
		manifestID, ok := manifest.SpecialTokenIDs[name]
		if !ok || manifestID != artifactID {
			return nil, "", fmt.Errorf("nucleotide special token %s mismatch: manifest=%d present=%t vocabulary=%d", name, manifestID, ok, artifactID)
		}
	}
	maskID, _ := vocab.SpecialTokenID("mask")
	objective := cfg.Training.EffectiveObjective()
	if objective == arch.ObjectiveMLM || objective == arch.ObjectiveMNTP || objective == arch.ObjectiveHybrid {
		if cfg.Training.MLMMaskTokenID != maskID {
			return nil, "", fmt.Errorf("config %q training.mlm_mask_token_id=%d does not match nucleotide MASK id %d", cfg.Name, cfg.Training.MLMMaskTokenID, maskID)
		}
	}
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
			return nil, "", err
		}
		cfg.Training.DatasetNucleotideComplement = complement
	} else if cfg.Training.ReverseComplementProb > 0 || cfg.RCEquivarianceEnabled() {
		return nil, "", fmt.Errorf("config %q reverse-complement features are DNA-only; dataset alphabet=%q", cfg.Name, vocab.Alphabet)
	}
	return vocab, vocabPath, nil
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
