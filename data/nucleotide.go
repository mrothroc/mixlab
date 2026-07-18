package data

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

const (
	NucleotideVocabularyFormat  = "mixlab.nucleotide_vocabulary"
	NucleotideVocabularyVersion = 1
	NucleotideAlphabetDNA       = "dna"
	NucleotideAlphabetRNA       = "rna"
)

var nucleotideComplements = map[string]map[string]string{
	NucleotideAlphabetDNA: {
		"A": "T", "C": "G", "G": "C", "T": "A", "N": "N",
		"R": "Y", "Y": "R", "W": "W", "S": "S", "K": "M", "M": "K",
		"B": "V", "V": "B", "D": "H", "H": "D",
	},
	NucleotideAlphabetRNA: {
		"A": "U", "C": "G", "G": "C", "U": "A", "N": "N",
		"R": "Y", "Y": "R", "W": "W", "S": "S", "K": "M", "M": "K",
		"B": "V", "V": "B", "D": "H", "H": "D",
	},
}

// NucleotideVocabulary is the inspectable token contract emitted by FASTA
// preparation. Token IDs are deterministic for a given alphabet and set of
// ambiguous symbols.
type NucleotideVocabulary struct {
	Format              string            `json:"format"`
	Version             int               `json:"version"`
	Alphabet            string            `json:"alphabet"`
	InvalidSymbolPolicy string            `json:"invalid_symbol_policy"`
	AmbiguousSymbols    []string          `json:"ambiguous_symbols,omitempty"`
	Tokens              map[string]int    `json:"tokens"`
	Complements         map[string]string `json:"complements,omitempty"`

	idToToken []string
}

func LoadNucleotideVocabulary(path string) (*NucleotideVocabulary, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var vocab NucleotideVocabulary
	dec := json.NewDecoder(bytes.NewReader(blob))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&vocab); err != nil {
		return nil, fmt.Errorf("parse nucleotide vocabulary %q: %w", path, err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return nil, fmt.Errorf("parse nucleotide vocabulary %q: %w", path, err)
	}
	if err := vocab.Validate(); err != nil {
		return nil, fmt.Errorf("nucleotide vocabulary %q: %w", path, err)
	}
	return &vocab, nil
}

func (v *NucleotideVocabulary) Validate() error {
	if v == nil {
		return fmt.Errorf("vocabulary is nil")
	}
	v.Format = strings.ToLower(strings.TrimSpace(v.Format))
	v.Alphabet = strings.ToLower(strings.TrimSpace(v.Alphabet))
	v.InvalidSymbolPolicy = strings.ToLower(strings.TrimSpace(v.InvalidSymbolPolicy))
	if v.Format != NucleotideVocabularyFormat {
		return fmt.Errorf("format=%q, want %q", v.Format, NucleotideVocabularyFormat)
	}
	if v.Version != NucleotideVocabularyVersion {
		return fmt.Errorf("version=%d is unsupported; this build supports version %d", v.Version, NucleotideVocabularyVersion)
	}
	if v.Alphabet != NucleotideAlphabetDNA && v.Alphabet != NucleotideAlphabetRNA {
		return fmt.Errorf("alphabet=%q must be %q or %q", v.Alphabet, NucleotideAlphabetDNA, NucleotideAlphabetRNA)
	}
	switch v.InvalidSymbolPolicy {
	case "error", "map_to_n", "skip":
	default:
		return fmt.Errorf("invalid_symbol_policy=%q must be error, map_to_n, or skip", v.InvalidSymbolPolicy)
	}
	if len(v.Tokens) == 0 {
		return fmt.Errorf("tokens must not be empty")
	}
	fixedSpecials := map[string]int{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<MASK>": 3}
	for token, wantID := range fixedSpecials {
		if got, ok := v.Tokens[token]; !ok || got != wantID {
			return fmt.Errorf("tokens[%q]=%d present=%t; version %d requires id %d", token, got, ok, v.Version, wantID)
		}
	}
	required := []string{"A", "C", "G"}
	if v.Alphabet == NucleotideAlphabetDNA {
		required = append(required, "T")
	} else {
		required = append(required, "U")
	}
	biological := make(map[string]bool, len(required)+len(v.AmbiguousSymbols))
	for _, token := range required {
		biological[token] = true
	}
	for _, symbol := range v.AmbiguousSymbols {
		if len(symbol) != 1 || !strings.ContainsAny("NRYWSKMBDHV", symbol) {
			return fmt.Errorf("ambiguous_symbols contains unsupported symbol %q", symbol)
		}
		if biological[symbol] {
			return fmt.Errorf("ambiguous_symbols contains duplicate symbol %q", symbol)
		}
		biological[symbol] = true
	}
	if v.InvalidSymbolPolicy == "map_to_n" && !biological["N"] {
		return fmt.Errorf("invalid_symbol_policy=map_to_n requires N in ambiguous_symbols")
	}
	maxID := -1
	seen := make(map[int]string, len(v.Tokens))
	for token, id := range v.Tokens {
		if strings.TrimSpace(token) == "" {
			return fmt.Errorf("tokens contains an empty token")
		}
		if id < 0 || id >= 1<<16 {
			return fmt.Errorf("tokens[%q]=%d is outside uint16 range", token, id)
		}
		if previous, ok := seen[id]; ok {
			return fmt.Errorf("tokens %q and %q both use id %d", previous, token, id)
		}
		if !strings.HasPrefix(token, "<") && !biological[token] {
			return fmt.Errorf("tokens contains biological symbol %q not declared by alphabet/ambiguous_symbols", token)
		}
		seen[id] = token
		if id > maxID {
			maxID = id
		}
	}
	for _, token := range required {
		if _, ok := v.Tokens[token]; !ok {
			return fmt.Errorf("tokens is missing required token %q", token)
		}
	}
	for symbol := range biological {
		if _, ok := v.Tokens[symbol]; !ok {
			return fmt.Errorf("tokens is missing declared biological symbol %q", symbol)
		}
		want := nucleotideComplements[v.Alphabet][symbol]
		if got := v.Complements[symbol]; got != want {
			return fmt.Errorf("complements[%q]=%q, want canonical %q", symbol, got, want)
		}
		if !biological[want] {
			return fmt.Errorf("biological vocabulary is not complement-closed: %q requires %q", symbol, want)
		}
	}
	v.idToToken = make([]string, maxID+1)
	for token, id := range v.Tokens {
		v.idToToken[id] = token
	}
	for id, token := range v.idToToken {
		if token == "" {
			return fmt.Errorf("token id %d is unassigned; nucleotide vocabularies must be contiguous", id)
		}
	}
	return nil
}

func (v *NucleotideVocabulary) Size() int {
	if v == nil {
		return 0
	}
	if len(v.idToToken) == 0 {
		_ = v.Validate()
	}
	return len(v.idToToken)
}

func (v *NucleotideVocabulary) SpecialTokenID(name string) (int, bool) {
	if v == nil {
		return 0, false
	}
	id, ok := v.Tokens["<"+strings.ToUpper(strings.Trim(name, "<>"))+">"]
	return id, ok
}

// Encode converts a nucleotide string to base/ambiguity token IDs. FASTA
// preparation has already applied invalid-symbol policy, so native string I/O
// rejects unknown symbols instead of silently changing a sequence.
func (v *NucleotideVocabulary) Encode(sequence string) ([]int, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}
	out := make([]int, 0, len(sequence))
	for pos, r := range strings.ToUpper(strings.TrimSpace(sequence)) {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			continue
		}
		id, ok := v.Tokens[string(r)]
		if !ok || strings.HasPrefix(string(r), "<") {
			return nil, fmt.Errorf("sequence symbol %q at character %d is not in the %s vocabulary", r, pos, v.Alphabet)
		}
		out = append(out, id)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("sequence contains no nucleotide symbols")
	}
	return out, nil
}

// Decode returns a nucleotide string. Framing specials are omitted, EOS ends
// decoding, and MASK is rendered as "?" so generated uncertainty is visible.
func (v *NucleotideVocabulary) Decode(ids []int) (string, error) {
	if err := v.Validate(); err != nil {
		return "", err
	}
	var out strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(v.idToToken) {
			return "", fmt.Errorf("token id %d is outside [0,%d)", id, len(v.idToToken))
		}
		token := v.idToToken[id]
		switch token {
		case "<PAD>", "<BOS>":
			continue
		case "<EOS>":
			return out.String(), nil
		case "<MASK>":
			out.WriteByte('?')
		default:
			out.WriteString(token)
		}
	}
	return out.String(), nil
}

// ComplementTokenIDs returns an ID-to-ID complement lookup. It fails if any
// biological symbol lacks a complement, which keeps reverse-complement
// augmentation auditable instead of partially transforming records.
func (v *NucleotideVocabulary) ComplementTokenIDs() ([]int, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}
	out := make([]int, len(v.idToToken))
	for id := range out {
		out[id] = id
	}
	keys := make([]string, 0, len(v.Tokens))
	for token := range v.Tokens {
		keys = append(keys, token)
	}
	sort.Strings(keys)
	for _, token := range keys {
		if strings.HasPrefix(token, "<") {
			continue
		}
		complement, ok := v.Complements[token]
		if !ok {
			return nil, fmt.Errorf("token %q has no complement mapping", token)
		}
		to, ok := v.Tokens[complement]
		if !ok {
			return nil, fmt.Errorf("token %q complement %q is not in tokens", token, complement)
		}
		out[v.Tokens[token]] = to
	}
	return out, nil
}
