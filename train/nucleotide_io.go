package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

func loadSequenceVocabularyForConfig(path string, cfg *ArchConfig) (*data.NucleotideVocabulary, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	vocab, err := data.LoadNucleotideVocabulary(path)
	if err != nil {
		return nil, err
	}
	if cfg == nil {
		return nil, fmt.Errorf("nil model config")
	}
	if vocab.Size() != cfg.VocabSize {
		return nil, fmt.Errorf("-sequence-vocab size=%d does not match config vocab_size=%d", vocab.Size(), cfg.VocabSize)
	}
	return vocab, nil
}

func encodeScoreEBMSequenceFields(rec scoreEBMInputRecord, vocab *data.NucleotideVocabulary) (scoreEBMInputRecord, error) {
	hasString := strings.TrimSpace(rec.Sequence) != "" || strings.TrimSpace(rec.CleanSequence) != "" || strings.TrimSpace(rec.CorruptSequence) != ""
	if !hasString {
		return rec, nil
	}
	if vocab == nil {
		return scoreEBMInputRecord{}, fmt.Errorf("sequence string fields require -sequence-vocab")
	}
	if strings.TrimSpace(rec.Sequence) != "" {
		if len(rec.Tokens) > 0 || len(rec.Clean) > 0 || len(rec.Corrupt) > 0 || rec.CleanSequence != "" || rec.CorruptSequence != "" {
			return scoreEBMInputRecord{}, fmt.Errorf("sequence cannot be combined with token or pair fields")
		}
		ids, err := vocab.Encode(rec.Sequence)
		if err != nil {
			return scoreEBMInputRecord{}, err
		}
		rec.Tokens = ids
		canonical, _ := vocab.Decode(ids)
		rec.Sequence = canonical
		return rec, nil
	}
	if strings.TrimSpace(rec.CleanSequence) == "" || strings.TrimSpace(rec.CorruptSequence) == "" {
		return scoreEBMInputRecord{}, fmt.Errorf("sequence pair records require both clean_sequence and corrupt_sequence")
	}
	if len(rec.Tokens) > 0 || len(rec.Clean) > 0 || len(rec.Corrupt) > 0 {
		return scoreEBMInputRecord{}, fmt.Errorf("clean_sequence/corrupt_sequence cannot be combined with token fields")
	}
	clean, err := vocab.Encode(rec.CleanSequence)
	if err != nil {
		return scoreEBMInputRecord{}, fmt.Errorf("clean_sequence: %w", err)
	}
	corrupt, err := vocab.Encode(rec.CorruptSequence)
	if err != nil {
		return scoreEBMInputRecord{}, fmt.Errorf("corrupt_sequence: %w", err)
	}
	rec.Clean, rec.Corrupt = clean, corrupt
	rec.CleanSequence, _ = vocab.Decode(clean)
	rec.CorruptSequence, _ = vocab.Decode(corrupt)
	return rec, nil
}
