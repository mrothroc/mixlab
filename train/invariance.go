package train

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

const (
	invariancePairBinaryMagic   uint32 = 0x52564e49 // "INVR", little-endian on disk.
	invariancePairBinaryVersion uint32 = 1
	invariancePairSamplerSalt          = 0x1a4a71a6ce000001
)

type invariancePairRecord struct {
	ID       string `json:"id"`
	Family   string `json:"family,omitempty"`
	ViewA    []int  `json:"view_a"`
	ViewAPos int    `json:"view_a_pos"`
	ViewB    []int  `json:"view_b"`
	ViewBPos int    `json:"view_b_pos"`

	viewAPosSet bool
	viewBPosSet bool
}

func (r *invariancePairRecord) UnmarshalJSON(data []byte) error {
	type alias invariancePairRecord
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	var raw alias
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return err
	}
	*r = invariancePairRecord(raw)
	_, r.viewAPosSet = fields["view_a_pos"]
	_, r.viewBPosSet = fields["view_b_pos"]
	return nil
}

type invariancePairDecodeOptions struct {
	VocabSize     int
	MaxLen        int
	MaskTokenID   int
	SkipTokenIDs  map[int]bool
	RequireFamily bool
}

type invariancePairSampler struct {
	records []invariancePairRecord
	path    string
}

type invariancePairSummary struct {
	Total    int            `json:"total"`
	Families map[string]int `json:"families"`
}

func invarianceActive(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.InvarianceActive()
}

func newInvariancePairSampler(cfg *ArchConfig) (*invariancePairSampler, error) {
	if !invarianceActive(cfg) {
		return nil, nil
	}
	path := resolveConfigRelativePath(cfg.SourcePath, cfg.Training.Invariance.Path)
	records, err := loadInvariancePairs(path, cfg.Training.Invariance.Source, invariancePairDecodeOptions{
		VocabSize:     cfg.VocabSize,
		MaxLen:        cfg.SeqLen,
		MaskTokenID:   cfg.Training.MLMMaskTokenID,
		SkipTokenIDs:  invarianceSkipTokenIDs(cfg),
		RequireFamily: true,
	})
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("invariance pair file %q has no records", path)
	}
	return &invariancePairSampler{records: records, path: path}, nil
}

func loadInvariancePairs(path, source string, opts invariancePairDecodeOptions) ([]invariancePairRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open invariance pair file %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	source = strings.ToLower(strings.TrimSpace(source))
	if source == "" || source == arch.InvarianceSourceFile {
		var magic uint32
		if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
			return nil, fmt.Errorf("read invariance pair file %q: %w", path, err)
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return nil, fmt.Errorf("rewind invariance pair file %q: %w", path, err)
		}
		if magic == invariancePairBinaryMagic {
			return decodeInvariancePairBinary(f, path, opts)
		}
		return decodeInvariancePairJSONL(f, path, opts)
	}
	switch source {
	case arch.InvarianceSourceJSONL:
		return decodeInvariancePairJSONL(f, path, opts)
	case arch.InvarianceSourceBinary:
		return decodeInvariancePairBinary(f, path, opts)
	default:
		return nil, fmt.Errorf("invariance pair source %q is not supported", source)
	}
}

func decodeInvariancePairJSONL(r io.Reader, source string, opts invariancePairDecodeOptions) ([]invariancePairRecord, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	var out []invariancePairRecord
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec invariancePairRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("%s line %d: invalid JSON: %w", source, lineNo, err)
		}
		if err := validateInvariancePairRecord(rec, opts); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", source, lineNo, err)
		}
		out = append(out, rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read invariance pair file %q: %w", source, err)
	}
	return out, nil
}

func validateInvariancePairRecord(rec invariancePairRecord, opts invariancePairDecodeOptions) error {
	if strings.TrimSpace(rec.ID) == "" {
		return fmt.Errorf("id must be non-empty")
	}
	if opts.RequireFamily && strings.TrimSpace(rec.Family) == "" {
		return fmt.Errorf("family must be non-empty")
	}
	if len(rec.ViewA) == 0 || len(rec.ViewB) == 0 {
		return fmt.Errorf("view_a and view_b tokens must be non-empty")
	}
	if !rec.viewAPosSet || !rec.viewBPosSet {
		return fmt.Errorf("view_a_pos and view_b_pos are required")
	}
	if rec.ViewAPos < 0 || rec.ViewAPos >= len(rec.ViewA) {
		return fmt.Errorf("view_a_pos=%d must be within view_a length %d", rec.ViewAPos, len(rec.ViewA))
	}
	if rec.ViewBPos < 0 || rec.ViewBPos >= len(rec.ViewB) {
		return fmt.Errorf("view_b_pos=%d must be within view_b length %d", rec.ViewBPos, len(rec.ViewB))
	}
	if opts.MaxLen > 0 && (len(rec.ViewA) > opts.MaxLen || len(rec.ViewB) > opts.MaxLen) {
		return fmt.Errorf("view lengths %d/%d exceed max length %d", len(rec.ViewA), len(rec.ViewB), opts.MaxLen)
	}
	for name, tokens := range map[string][]int{"view_a": rec.ViewA, "view_b": rec.ViewB} {
		for i, token := range tokens {
			if token < 0 || (opts.VocabSize > 0 && token >= opts.VocabSize) {
				return fmt.Errorf("%s[%d]=%d out of range [0,%d)", name, i, token, opts.VocabSize)
			}
		}
	}
	if rec.ViewA[rec.ViewAPos] != rec.ViewB[rec.ViewBPos] {
		return fmt.Errorf("annotated target tokens must match across views")
	}
	if opts.MaskTokenID >= 0 && (rec.ViewA[rec.ViewAPos] == opts.MaskTokenID || rec.ViewB[rec.ViewBPos] == opts.MaskTokenID) {
		return fmt.Errorf("annotated target positions must not contain mlm_mask_token_id=%d", opts.MaskTokenID)
	}
	if opts.SkipTokenIDs != nil && (opts.SkipTokenIDs[rec.ViewA[rec.ViewAPos]] || opts.SkipTokenIDs[rec.ViewB[rec.ViewBPos]]) {
		return fmt.Errorf("annotated target positions must not contain training.invariance.skip_token_ids")
	}
	if minimalPairEqualIntSlices(rec.ViewA, rec.ViewB) {
		return fmt.Errorf("view_a and view_b tokens are identical")
	}
	return nil
}

func invarianceSkipTokenIDs(cfg *ArchConfig) map[int]bool {
	if cfg == nil || cfg.Training.Invariance == nil {
		return nil
	}
	out := make(map[int]bool, len(cfg.Training.Invariance.SkipTokenIDs))
	for _, id := range cfg.Training.Invariance.SkipTokenIDs {
		out[id] = true
	}
	return out
}

func writeInvariancePairBinary(w io.Writer, records []invariancePairRecord, vocabSize, maxLen int) error {
	header := []uint32{invariancePairBinaryMagic, invariancePairBinaryVersion, uint32(vocabSize), uint32(maxLen), uint32(len(records)), 0}
	for _, field := range header {
		if err := binary.Write(w, binary.LittleEndian, field); err != nil {
			return fmt.Errorf("write invariance pair binary header: %w", err)
		}
	}
	for i, rec := range records {
		fields := []uint32{uint32(len(rec.ViewA)), uint32(len(rec.ViewB)), uint32(len(rec.ID)), uint32(len(rec.Family))}
		for _, field := range fields {
			if err := binary.Write(w, binary.LittleEndian, field); err != nil {
				return fmt.Errorf("write invariance pair record %d: %w", i+1, err)
			}
		}
		for _, pos := range []int32{int32(rec.ViewAPos), int32(rec.ViewBPos)} {
			if err := binary.Write(w, binary.LittleEndian, pos); err != nil {
				return fmt.Errorf("write invariance pair positions %d: %w", i+1, err)
			}
		}
		for _, tokens := range [][]int{rec.ViewA, rec.ViewB} {
			for _, token := range tokens {
				if err := binary.Write(w, binary.LittleEndian, uint32(token)); err != nil {
					return fmt.Errorf("write invariance pair tokens %d: %w", i+1, err)
				}
			}
		}
		if _, err := io.WriteString(w, rec.ID); err != nil {
			return fmt.Errorf("write invariance pair id %d: %w", i+1, err)
		}
		if _, err := io.WriteString(w, rec.Family); err != nil {
			return fmt.Errorf("write invariance pair family %d: %w", i+1, err)
		}
	}
	return nil
}

func decodeInvariancePairBinary(r io.Reader, source string, opts invariancePairDecodeOptions) ([]invariancePairRecord, error) {
	var header [6]uint32
	for i := range header {
		if err := binary.Read(r, binary.LittleEndian, &header[i]); err != nil {
			return nil, fmt.Errorf("%s: read binary header: %w", source, err)
		}
	}
	if header[0] != invariancePairBinaryMagic {
		return nil, fmt.Errorf("%s: invalid invariance-pair binary magic 0x%x", source, header[0])
	}
	if header[1] != invariancePairBinaryVersion {
		return nil, fmt.Errorf("%s: unsupported invariance-pair binary version %d", source, header[1])
	}
	if opts.VocabSize > 0 && header[2] > 0 && int(header[2]) != opts.VocabSize {
		return nil, fmt.Errorf("%s: invariance-pair binary vocab_size=%d does not match config vocab_size=%d", source, header[2], opts.VocabSize)
	}
	recordCount := int(header[4])
	records := make([]invariancePairRecord, 0, recordCount)
	for i := 0; i < recordCount; i++ {
		var lens [4]uint32
		for j := range lens {
			if err := binary.Read(r, binary.LittleEndian, &lens[j]); err != nil {
				return nil, fmt.Errorf("%s record %d: read lengths: %w", source, i+1, err)
			}
		}
		var positions [2]int32
		for j := range positions {
			if err := binary.Read(r, binary.LittleEndian, &positions[j]); err != nil {
				return nil, fmt.Errorf("%s record %d: read positions: %w", source, i+1, err)
			}
		}
		viewA, err := readMinimalPairTokenVector(r, int(lens[0]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read view_a: %w", source, i+1, err)
		}
		viewB, err := readMinimalPairTokenVector(r, int(lens[1]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read view_b: %w", source, i+1, err)
		}
		id, err := readMinimalPairString(r, int(lens[2]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read id: %w", source, i+1, err)
		}
		family, err := readMinimalPairString(r, int(lens[3]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read family: %w", source, i+1, err)
		}
		rec := invariancePairRecord{ID: id, Family: family, ViewA: viewA, ViewAPos: int(positions[0]), ViewB: viewB, ViewBPos: int(positions[1]), viewAPosSet: true, viewBPosSet: true}
		if err := validateInvariancePairRecord(rec, opts); err != nil {
			return nil, fmt.Errorf("%s record %d: %w", source, i+1, err)
		}
		records = append(records, rec)
	}
	var extra [1]byte
	if n, err := r.Read(extra[:]); err != io.EOF {
		if err == nil && n > 0 {
			return nil, fmt.Errorf("%s: trailing bytes after %d records", source, recordCount)
		}
		return nil, fmt.Errorf("%s: read trailing bytes: %w", source, err)
	}
	return records, nil
}

func summarizeInvariancePairs(records []invariancePairRecord) invariancePairSummary {
	summary := invariancePairSummary{Total: len(records), Families: make(map[string]int)}
	for _, rec := range records {
		family := strings.TrimSpace(rec.Family)
		if family == "" {
			family = "unknown"
		}
		summary.Families[family]++
	}
	return summary
}

func maybeAttachInvariancePairs(sampler *invariancePairSampler, cfg *ArchConfig, step int, batch objectiveBatch, rawBatchSize, seqLen int, objective string) (objectiveBatch, error) {
	if sampler == nil || !invarianceActive(cfg) {
		return batch, nil
	}
	if rawBatchSize < 2 || rawBatchSize%2 != 0 || seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("invariance training requires an even batch of at least two rows, got rows=%d seq_len=%d", rawBatchSize, seqLen)
	}
	need := rawBatchSize * seqLen
	tokenOffset := 0
	maskedObjective := objective
	if cfg.Training.MultiheadEnabled() {
		head := cfg.Training.MultiheadExportHead()
		if head == nil {
			return objectiveBatch{}, fmt.Errorf("invariance training requires a multihead export head")
		}
		headIndex := -1
		for i, candidate := range cfg.Training.Heads {
			if candidate.Name == head.Name {
				headIndex = i
				break
			}
		}
		if headIndex < 0 {
			return objectiveBatch{}, fmt.Errorf("invariance export head %q is not present", head.Name)
		}
		tokenOffset = headIndex * need
		maskedObjective = head.Objective
	}
	if maskedObjective != arch.ObjectiveMLM && maskedObjective != arch.ObjectiveMNTP {
		return objectiveBatch{}, fmt.Errorf("invariance training requires MLM or MNTP logits, got %q", maskedObjective)
	}
	if len(batch.x) < tokenOffset+need || len(batch.y) < tokenOffset+need || len(batch.lossMask) < tokenOffset+need {
		return objectiveBatch{}, fmt.Errorf("invariance objective batch is too small for selected rows")
	}
	if len(batch.invarianceLossMask) < len(batch.x) {
		batch.invarianceLossMask = make([]float32, len(batch.x))
	} else {
		clear(batch.invarianceLossMask)
	}
	activePairs := int(math.Ceil(float64(rawBatchSize/2) * cfg.Training.Invariance.BatchFraction))
	if activePairs < 1 {
		activePairs = 1
	}
	if activePairs > rawBatchSize/2 {
		activePairs = rawBatchSize / 2
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, invariancePairSamplerSalt)
	pad := minimalPairPadTokenID(cfg)
	for pairIdx := 0; pairIdx < activePairs; pairIdx++ {
		rec := sampler.records[rng.Intn(len(sampler.records))]
		viewARow := tokenOffset + (2*pairIdx)*seqLen
		viewBRow := viewARow + seqLen
		if err := attachInvarianceView(batch, viewARow, rec.ViewA, rec.ViewAPos, pad, cfg.Training.MLMMaskTokenID, maskedObjective, seqLen); err != nil {
			return objectiveBatch{}, fmt.Errorf("invariance record %q view_a: %w", rec.ID, err)
		}
		if err := attachInvarianceView(batch, viewBRow, rec.ViewB, rec.ViewBPos, pad, cfg.Training.MLMMaskTokenID, maskedObjective, seqLen); err != nil {
			return objectiveBatch{}, fmt.Errorf("invariance record %q view_b: %w", rec.ID, err)
		}
		batch.invarianceLossMask[viewARow+rec.ViewAPos] = 1
		batch.invarianceLossMask[viewBRow+rec.ViewBPos] = 1
	}
	return batch, nil
}

func attachInvarianceView(batch objectiveBatch, rowStart int, tokens []int, targetPos, pad, maskTokenID int, objective string, seqLen int) error {
	if targetPos < 0 || targetPos >= len(tokens) || targetPos >= seqLen {
		return fmt.Errorf("target position %d is outside sequence", targetPos)
	}
	if len(batch.x) < rowStart+seqLen || len(batch.y) < rowStart+seqLen || len(batch.lossMask) < rowStart+seqLen {
		return fmt.Errorf("objective row is too short")
	}
	x := batch.x[rowStart : rowStart+seqLen]
	y := batch.y[rowStart : rowStart+seqLen]
	mask := batch.lossMask[rowStart : rowStart+seqLen]
	fillMinimalPairTokens(x, tokens, pad)
	if len(batch.unmaskedX) >= rowStart+seqLen {
		fillMinimalPairTokens(batch.unmaskedX[rowStart:rowStart+seqLen], tokens, pad)
	}
	clear(mask)
	if objective == arch.ObjectiveMNTP {
		fillMinimalPairTokens(y, tokens, pad)
		for i := 0; i+1 < len(tokens) && i+1 < seqLen; i++ {
			y[i] = tokens[i+1]
		}
		if targetPos+1 < len(tokens) && targetPos < seqLen-1 {
			mask[targetPos] = 1
		}
	} else {
		fillMinimalPairTokens(y, tokens, pad)
		mask[targetPos] = 1
	}
	x[targetPos] = maskTokenID
	return nil
}
