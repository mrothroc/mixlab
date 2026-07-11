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
	pllMarginPairBinaryMagic   uint32 = 0x4d4c4c50 // "PLLM", little-endian on disk.
	pllMarginPairBinaryVersion uint32 = 1
	pllMarginPairSamplerSalt          = 0x504c4c4d00000001
)

// pllMarginPairRecord holds a preferred/contrast view pair and an explicitly
// annotated unchanged target span. The views may have different lengths and
// target positions, but target IDs must be identical in order.
type pllMarginPairRecord struct {
	ID                 string `json:"id"`
	Family             string `json:"family,omitempty"`
	ViewPos            []int  `json:"view_pos"`
	TargetPosPositions []int  `json:"target_pos_positions"`
	ViewNeg            []int  `json:"view_neg"`
	TargetNegPositions []int  `json:"target_neg_positions"`
	TargetIDs          []int  `json:"target_ids"`

	viewPosSet            bool
	targetPosPositionsSet bool
	viewNegSet            bool
	targetNegPositionsSet bool
	targetIDsSet          bool
}

func (r *pllMarginPairRecord) UnmarshalJSON(data []byte) error {
	type alias pllMarginPairRecord
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
	*r = pllMarginPairRecord(raw)
	_, r.viewPosSet = fields["view_pos"]
	_, r.targetPosPositionsSet = fields["target_pos_positions"]
	_, r.viewNegSet = fields["view_neg"]
	_, r.targetNegPositionsSet = fields["target_neg_positions"]
	_, r.targetIDsSet = fields["target_ids"]
	return nil
}

type pllMarginPairDecodeOptions struct {
	VocabSize     int
	MaxLen        int
	MaskTokenID   int
	SkipTokenIDs  map[int]bool
	RequireFamily bool
}

type pllMarginPairSampler struct {
	records []pllMarginPairRecord
	path    string
}

type pllMarginPairSummary struct {
	Total    int            `json:"total"`
	Families map[string]int `json:"families"`
}

func pllMarginActive(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.PLLMarginActive()
}

func newPLLMarginPairSampler(cfg *ArchConfig) (*pllMarginPairSampler, error) {
	if !pllMarginActive(cfg) {
		return nil, nil
	}
	path := resolveConfigRelativePath(cfg.SourcePath, cfg.Training.PLLMargin.Path)
	records, err := loadPLLMarginPairs(path, cfg.Training.PLLMargin.Source, pllMarginPairDecodeOptions{
		VocabSize:     cfg.VocabSize,
		MaxLen:        cfg.SeqLen,
		MaskTokenID:   cfg.Training.MLMMaskTokenID,
		SkipTokenIDs:  pllMarginSkipTokenIDs(cfg),
		RequireFamily: true,
	})
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("PLL margin pair file %q has no records", path)
	}
	return &pllMarginPairSampler{records: records, path: path}, nil
}

func loadPLLMarginPairs(path, source string, opts pllMarginPairDecodeOptions) ([]pllMarginPairRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open PLL margin pair file %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	source = strings.ToLower(strings.TrimSpace(source))
	if source == "" || source == arch.PLLMarginSourceFile {
		var magic uint32
		if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
			return nil, fmt.Errorf("read PLL margin pair file %q: %w", path, err)
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return nil, fmt.Errorf("rewind PLL margin pair file %q: %w", path, err)
		}
		if magic == pllMarginPairBinaryMagic {
			return decodePLLMarginPairBinary(f, path, opts)
		}
		return decodePLLMarginPairJSONL(f, path, opts)
	}
	switch source {
	case arch.PLLMarginSourceJSONL:
		return decodePLLMarginPairJSONL(f, path, opts)
	case arch.PLLMarginSourceBinary:
		return decodePLLMarginPairBinary(f, path, opts)
	default:
		return nil, fmt.Errorf("PLL margin pair source %q is not supported", source)
	}
}

func decodePLLMarginPairJSONL(r io.Reader, source string, opts pllMarginPairDecodeOptions) ([]pllMarginPairRecord, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	var out []pllMarginPairRecord
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec pllMarginPairRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("%s line %d: invalid JSON: %w", source, lineNo, err)
		}
		if err := validatePLLMarginPairRecord(rec, opts); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", source, lineNo, err)
		}
		out = append(out, rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read PLL margin pair file %q: %w", source, err)
	}
	return out, nil
}

func validatePLLMarginPairRecord(rec pllMarginPairRecord, opts pllMarginPairDecodeOptions) error {
	if strings.TrimSpace(rec.ID) == "" {
		return fmt.Errorf("id must be non-empty")
	}
	if opts.RequireFamily && strings.TrimSpace(rec.Family) == "" {
		return fmt.Errorf("family must be non-empty")
	}
	if !rec.viewPosSet || !rec.viewNegSet || !rec.targetPosPositionsSet || !rec.targetNegPositionsSet || !rec.targetIDsSet {
		return fmt.Errorf("view_pos, target_pos_positions, view_neg, target_neg_positions, and target_ids are required")
	}
	if len(rec.ViewPos) == 0 || len(rec.ViewNeg) == 0 {
		return fmt.Errorf("view_pos and view_neg tokens must be non-empty")
	}
	if len(rec.TargetIDs) == 0 || len(rec.TargetPosPositions) != len(rec.TargetIDs) || len(rec.TargetNegPositions) != len(rec.TargetIDs) {
		return fmt.Errorf("target position arrays must be non-empty and match target_ids length")
	}
	if opts.MaxLen > 0 && (len(rec.ViewPos) > opts.MaxLen || len(rec.ViewNeg) > opts.MaxLen) {
		return fmt.Errorf("view lengths %d/%d exceed max length %d", len(rec.ViewPos), len(rec.ViewNeg), opts.MaxLen)
	}
	for name, tokens := range map[string][]int{"view_pos": rec.ViewPos, "view_neg": rec.ViewNeg} {
		for i, token := range tokens {
			if token < 0 || (opts.VocabSize > 0 && token >= opts.VocabSize) {
				return fmt.Errorf("%s[%d]=%d out of range [0,%d)", name, i, token, opts.VocabSize)
			}
		}
	}
	if err := validatePLLMarginTargetPositions("target_pos_positions", rec.TargetPosPositions, rec.ViewPos, rec.TargetIDs, opts); err != nil {
		return err
	}
	if err := validatePLLMarginTargetPositions("target_neg_positions", rec.TargetNegPositions, rec.ViewNeg, rec.TargetIDs, opts); err != nil {
		return err
	}
	if minimalPairEqualIntSlices(rec.ViewPos, rec.ViewNeg) {
		return fmt.Errorf("view_pos and view_neg tokens are identical")
	}
	return nil
}

func validatePLLMarginTargetPositions(name string, positions, tokens, targetIDs []int, opts pllMarginPairDecodeOptions) error {
	previous := -1
	for i, position := range positions {
		if position <= previous || position < 0 || position >= len(tokens) {
			return fmt.Errorf("%s[%d]=%d must be strictly increasing and within view length %d", name, i, position, len(tokens))
		}
		previous = position
		if tokens[position] != targetIDs[i] {
			return fmt.Errorf("%s[%d]=%d does not select target_ids[%d]=%d", name, i, position, i, targetIDs[i])
		}
		if opts.MaskTokenID >= 0 && targetIDs[i] == opts.MaskTokenID {
			return fmt.Errorf("annotated target positions must not contain mlm_mask_token_id=%d", opts.MaskTokenID)
		}
		if opts.SkipTokenIDs != nil && opts.SkipTokenIDs[targetIDs[i]] {
			return fmt.Errorf("annotated target positions must not contain training.pll_margin.skip_token_ids")
		}
	}
	return nil
}

func pllMarginSkipTokenIDs(cfg *ArchConfig) map[int]bool {
	if cfg == nil || cfg.Training.PLLMargin == nil {
		return nil
	}
	out := make(map[int]bool, len(cfg.Training.PLLMargin.SkipTokenIDs))
	for _, id := range cfg.Training.PLLMargin.SkipTokenIDs {
		out[id] = true
	}
	return out
}

func writePLLMarginPairBinary(w io.Writer, records []pllMarginPairRecord, vocabSize, maxLen int) error {
	header := []uint32{pllMarginPairBinaryMagic, pllMarginPairBinaryVersion, uint32(vocabSize), uint32(maxLen), uint32(len(records)), 0}
	for _, field := range header {
		if err := binary.Write(w, binary.LittleEndian, field); err != nil {
			return fmt.Errorf("write PLL margin pair binary header: %w", err)
		}
	}
	for i, rec := range records {
		fields := []uint32{uint32(len(rec.ViewPos)), uint32(len(rec.ViewNeg)), uint32(len(rec.TargetIDs)), uint32(len(rec.ID)), uint32(len(rec.Family))}
		for _, field := range fields {
			if err := binary.Write(w, binary.LittleEndian, field); err != nil {
				return fmt.Errorf("write PLL margin pair record %d: %w", i+1, err)
			}
		}
		for _, positions := range [][]int{rec.TargetPosPositions, rec.TargetNegPositions} {
			for _, position := range positions {
				if err := binary.Write(w, binary.LittleEndian, int32(position)); err != nil {
					return fmt.Errorf("write PLL margin pair positions %d: %w", i+1, err)
				}
			}
		}
		for _, tokens := range [][]int{rec.TargetIDs, rec.ViewPos, rec.ViewNeg} {
			for _, token := range tokens {
				if err := binary.Write(w, binary.LittleEndian, uint32(token)); err != nil {
					return fmt.Errorf("write PLL margin pair tokens %d: %w", i+1, err)
				}
			}
		}
		if _, err := io.WriteString(w, rec.ID); err != nil {
			return fmt.Errorf("write PLL margin pair id %d: %w", i+1, err)
		}
		if _, err := io.WriteString(w, rec.Family); err != nil {
			return fmt.Errorf("write PLL margin pair family %d: %w", i+1, err)
		}
	}
	return nil
}

func decodePLLMarginPairBinary(r io.Reader, source string, opts pllMarginPairDecodeOptions) ([]pllMarginPairRecord, error) {
	var header [6]uint32
	for i := range header {
		if err := binary.Read(r, binary.LittleEndian, &header[i]); err != nil {
			return nil, fmt.Errorf("%s: read binary header: %w", source, err)
		}
	}
	if header[0] != pllMarginPairBinaryMagic {
		return nil, fmt.Errorf("%s: invalid PLL margin pair binary magic 0x%x", source, header[0])
	}
	if header[1] != pllMarginPairBinaryVersion {
		return nil, fmt.Errorf("%s: unsupported PLL margin pair binary version %d", source, header[1])
	}
	if opts.VocabSize > 0 && header[2] > 0 && int(header[2]) != opts.VocabSize {
		return nil, fmt.Errorf("%s: PLL margin pair binary vocab_size=%d does not match config vocab_size=%d", source, header[2], opts.VocabSize)
	}
	records := make([]pllMarginPairRecord, 0, int(header[4]))
	for i := 0; i < int(header[4]); i++ {
		var lens [5]uint32
		for j := range lens {
			if err := binary.Read(r, binary.LittleEndian, &lens[j]); err != nil {
				return nil, fmt.Errorf("%s record %d: read lengths: %w", source, i+1, err)
			}
		}
		count := int(lens[2])
		readPositions := func() ([]int, error) {
			out := make([]int, count)
			for j := range out {
				var position int32
				if err := binary.Read(r, binary.LittleEndian, &position); err != nil {
					return nil, err
				}
				out[j] = int(position)
			}
			return out, nil
		}
		posPositions, err := readPositions()
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read preferred positions: %w", source, i+1, err)
		}
		negPositions, err := readPositions()
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read contrast positions: %w", source, i+1, err)
		}
		targetIDs, err := readMinimalPairTokenVector(r, count)
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read targets: %w", source, i+1, err)
		}
		viewPos, err := readMinimalPairTokenVector(r, int(lens[0]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read preferred view: %w", source, i+1, err)
		}
		viewNeg, err := readMinimalPairTokenVector(r, int(lens[1]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read contrast view: %w", source, i+1, err)
		}
		id, err := readMinimalPairString(r, int(lens[3]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read id: %w", source, i+1, err)
		}
		family, err := readMinimalPairString(r, int(lens[4]))
		if err != nil {
			return nil, fmt.Errorf("%s record %d: read family: %w", source, i+1, err)
		}
		rec := pllMarginPairRecord{
			ID: id, Family: family, ViewPos: viewPos, TargetPosPositions: posPositions, ViewNeg: viewNeg, TargetNegPositions: negPositions, TargetIDs: targetIDs,
			viewPosSet: true, targetPosPositionsSet: true, viewNegSet: true, targetNegPositionsSet: true, targetIDsSet: true,
		}
		if err := validatePLLMarginPairRecord(rec, opts); err != nil {
			return nil, fmt.Errorf("%s record %d: %w", source, i+1, err)
		}
		records = append(records, rec)
	}
	var extra [1]byte
	if n, err := r.Read(extra[:]); err != io.EOF {
		if err == nil && n > 0 {
			return nil, fmt.Errorf("%s: trailing bytes after %d records", source, len(records))
		}
		return nil, fmt.Errorf("%s: read trailing bytes: %w", source, err)
	}
	return records, nil
}

func summarizePLLMarginPairs(records []pllMarginPairRecord) pllMarginPairSummary {
	summary := pllMarginPairSummary{Total: len(records), Families: make(map[string]int)}
	for _, rec := range records {
		family := strings.TrimSpace(rec.Family)
		if family == "" {
			family = "unknown"
		}
		summary.Families[family]++
	}
	return summary
}

func maybeAttachPLLMarginPairs(sampler *pllMarginPairSampler, cfg *ArchConfig, step int, batch objectiveBatch, rawBatchSize, seqLen int, objective string) (objectiveBatch, error) {
	if sampler == nil || !pllMarginActive(cfg) {
		return batch, nil
	}
	if rawBatchSize < 2 || rawBatchSize%2 != 0 || seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("PLL margin training requires an even batch of at least two rows, got rows=%d seq_len=%d", rawBatchSize, seqLen)
	}
	need := rawBatchSize * seqLen
	tokenOffset := 0
	maskedObjective := objective
	if cfg.Training.MultiheadEnabled() {
		head := cfg.Training.MultiheadExportHead()
		if head == nil {
			return objectiveBatch{}, fmt.Errorf("PLL margin training requires a multihead export head")
		}
		headIndex := -1
		for i, candidate := range cfg.Training.Heads {
			if candidate.Name == head.Name {
				headIndex = i
				break
			}
		}
		if headIndex < 0 {
			return objectiveBatch{}, fmt.Errorf("PLL margin export head %q is not present", head.Name)
		}
		tokenOffset = headIndex * need
		maskedObjective = head.Objective
	}
	if maskedObjective != arch.ObjectiveMLM && maskedObjective != arch.ObjectiveMNTP {
		return objectiveBatch{}, fmt.Errorf("PLL margin training requires MLM or MNTP logits, got %q", maskedObjective)
	}
	if len(batch.x) < tokenOffset+need || len(batch.y) < tokenOffset+need || len(batch.lossMask) < tokenOffset+need {
		return objectiveBatch{}, fmt.Errorf("PLL margin objective batch is too small for selected rows")
	}
	if len(batch.pllMarginLossMask) < len(batch.x) {
		batch.pllMarginLossMask = make([]float32, len(batch.x))
	} else {
		clear(batch.pllMarginLossMask)
	}
	activePairs := int(math.Ceil(float64(rawBatchSize/2) * cfg.Training.PLLMargin.BatchFraction))
	if activePairs < 1 {
		activePairs = 1
	}
	if activePairs > rawBatchSize/2 {
		activePairs = rawBatchSize / 2
	}
	rng := deterministicObjectiveRNG(cfg.Training.Seed, step, pllMarginPairSamplerSalt)
	pad := minimalPairPadTokenID(cfg)
	for pairIdx := 0; pairIdx < activePairs; pairIdx++ {
		rec := sampler.records[rng.Intn(len(sampler.records))]
		preferredStart := tokenOffset + (2*pairIdx)*seqLen
		contrastStart := preferredStart + seqLen
		if err := attachPLLMarginView(&batch, preferredStart, rec.ViewPos, rec.TargetPosPositions, rec.TargetIDs, pad, cfg.Training.MLMMaskTokenID, seqLen); err != nil {
			return objectiveBatch{}, fmt.Errorf("PLL margin record %q preferred view: %w", rec.ID, err)
		}
		if err := attachPLLMarginView(&batch, contrastStart, rec.ViewNeg, rec.TargetNegPositions, rec.TargetIDs, pad, cfg.Training.MLMMaskTokenID, seqLen); err != nil {
			return objectiveBatch{}, fmt.Errorf("PLL margin record %q contrast view: %w", rec.ID, err)
		}
	}
	return batch, nil
}

// attachPLLMarginView deliberately clears the ordinary MLM/MNTP mask. The
// contrast view is used only as a ranking context; the dedicated op applies
// the positive-view anchor and the directional paired margin.
func attachPLLMarginView(batch *objectiveBatch, rowStart int, tokens, positions, targetIDs []int, pad, maskTokenID, seqLen int) error {
	if len(positions) == 0 || len(positions) != len(targetIDs) {
		return fmt.Errorf("target positions must be non-empty and match target IDs")
	}
	if len(batch.x) < rowStart+seqLen || len(batch.y) < rowStart+seqLen || len(batch.lossMask) < rowStart+seqLen || len(batch.pllMarginLossMask) < rowStart+seqLen {
		return fmt.Errorf("objective row is too short")
	}
	x := batch.x[rowStart : rowStart+seqLen]
	y := batch.y[rowStart : rowStart+seqLen]
	fillMinimalPairTokens(x, tokens, pad)
	fillMinimalPairTokens(y, tokens, pad)
	if len(batch.unmaskedX) >= rowStart+seqLen {
		fillMinimalPairTokens(batch.unmaskedX[rowStart:rowStart+seqLen], tokens, pad)
	}
	clear(batch.lossMask[rowStart : rowStart+seqLen])
	for i, position := range positions {
		if position < 0 || position >= len(tokens) || position >= seqLen {
			return fmt.Errorf("target position %d is outside sequence", position)
		}
		if tokens[position] != targetIDs[i] {
			return fmt.Errorf("target token mismatch at position %d", position)
		}
		x[position] = maskTokenID
		y[position] = targetIDs[i]
		batch.pllMarginLossMask[rowStart+position] = 1
	}
	return nil
}
