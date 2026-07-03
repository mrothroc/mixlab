package train

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

const (
	minimalPairBinaryMagic    uint32 = 0x52504c4d // "MLPR", little-endian on disk.
	minimalPairBinaryVersion  uint32 = 2
	minimalPairBinaryVersion1 uint32 = 1
)

type minimalPairRecord struct {
	ID          string `json:"id"`
	Clean       []int  `json:"clean"`
	Corrupt     []int  `json:"corrupt"`
	CleanSpan   []int  `json:"clean_span,omitempty"`
	CorruptSpan []int  `json:"corrupt_span,omitempty"`
	Family      string `json:"family,omitempty"`
}

type minimalPairDecodeOptions struct {
	VocabSize         int
	MaxLen            int
	RequireFamily     bool
	EnergyAggregation string
}

type minimalPairSampler struct {
	records []minimalPairRecord
	path    string
}

func minimalPairActive(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.MultiheadEnabled() && cfg.Training.MinimalPair != nil
}

func newMinimalPairSampler(cfg *ArchConfig) (*minimalPairSampler, error) {
	if !minimalPairActive(cfg) {
		return nil, nil
	}
	path := resolveConfigRelativePath(cfg.SourcePath, cfg.Training.MinimalPair.Path)
	opts := minimalPairDecodeOptions{
		VocabSize:         cfg.VocabSize,
		EnergyAggregation: cfg.Training.MinimalPair.EnergyAggregationMode(),
	}
	if cfg.Training.MinimalPair.UsesDifferingSpanEnergy() {
		opts.MaxLen = cfg.SeqLen
	}
	records, err := loadMinimalPairs(path, cfg.Training.MinimalPair.Source, minimalPairDecodeOptions{
		VocabSize:         opts.VocabSize,
		MaxLen:            opts.MaxLen,
		EnergyAggregation: opts.EnergyAggregation,
	})
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("minimal pair file %q has no records", path)
	}
	return &minimalPairSampler{records: records, path: path}, nil
}

func resolveConfigRelativePath(configPath, path string) string {
	path = strings.TrimSpace(path)
	if path == "" || filepath.IsAbs(path) || strings.TrimSpace(configPath) == "" {
		return path
	}
	dir := filepath.Dir(configPath)
	if dir == "." || dir == "" {
		return path
	}
	return filepath.Join(dir, path)
}

func loadMinimalPairs(path, source string, opts minimalPairDecodeOptions) ([]minimalPairRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open minimal pair file %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "", arch.MinimalPairSourceJSONL:
		return decodeMinimalPairJSONLWithOptions(f, path, opts)
	case arch.MinimalPairSourceBinary:
		return decodeMinimalPairBinary(f, path, opts)
	default:
		return nil, fmt.Errorf("minimal pair source %q is not supported", source)
	}
}

func decodeMinimalPairJSONL(r io.Reader, source string, vocabSize int) ([]minimalPairRecord, error) {
	return decodeMinimalPairJSONLWithOptions(r, source, minimalPairDecodeOptions{VocabSize: vocabSize})
}

func decodeMinimalPairJSONLWithOptions(r io.Reader, source string, opts minimalPairDecodeOptions) ([]minimalPairRecord, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	var out []minimalPairRecord
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec minimalPairRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("%s line %d: invalid JSON: %w", source, lineNo, err)
		}
		if err := validateMinimalPairRecordWithOptions(rec, opts); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", source, lineNo, err)
		}
		if minimalPairDecodeUsesDifferingSpan(opts) {
			if err := ensureMinimalPairRecordSpans(&rec); err != nil {
				return nil, fmt.Errorf("%s line %d: %w", source, lineNo, err)
			}
		}
		out = append(out, rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read minimal pair file %q: %w", source, err)
	}
	return out, nil
}

func validateMinimalPairRecordWithOptions(rec minimalPairRecord, opts minimalPairDecodeOptions) error {
	if strings.TrimSpace(rec.ID) == "" {
		return fmt.Errorf("id must be non-empty")
	}
	if opts.RequireFamily && strings.TrimSpace(rec.Family) == "" {
		return fmt.Errorf("family must be non-empty")
	}
	if len(rec.Clean) == 0 {
		return fmt.Errorf("clean tokens must be non-empty")
	}
	if len(rec.Corrupt) == 0 {
		return fmt.Errorf("corrupt tokens must be non-empty")
	}
	if opts.MaxLen > 0 {
		if len(rec.Clean) > opts.MaxLen {
			return fmt.Errorf("clean length %d exceeds max length %d", len(rec.Clean), opts.MaxLen)
		}
		if len(rec.Corrupt) > opts.MaxLen {
			return fmt.Errorf("corrupt length %d exceeds max length %d", len(rec.Corrupt), opts.MaxLen)
		}
	}
	for name, toks := range map[string][]int{"clean": rec.Clean, "corrupt": rec.Corrupt} {
		for i, tok := range toks {
			if tok < 0 || (opts.VocabSize > 0 && tok >= opts.VocabSize) {
				return fmt.Errorf("%s[%d]=%d out of range [0,%d)", name, i, tok, opts.VocabSize)
			}
		}
	}
	if _, _, ok, err := parseMinimalPairSpan("clean_span", rec.CleanSpan, len(rec.Clean)); err != nil || ok {
		if err != nil {
			return err
		}
	}
	if _, _, ok, err := parseMinimalPairSpan("corrupt_span", rec.CorruptSpan, len(rec.Corrupt)); err != nil || ok {
		if err != nil {
			return err
		}
	}
	return nil
}

func minimalPairDecodeUsesDifferingSpan(opts minimalPairDecodeOptions) bool {
	return opts.EnergyAggregation == arch.MinimalPairEnergySpan
}

func parseMinimalPairSpan(name string, span []int, n int) (int, int, bool, error) {
	if len(span) == 0 {
		return 0, 0, false, nil
	}
	if len(span) != 2 {
		return 0, 0, false, fmt.Errorf("%s must be a [start,end] pair", name)
	}
	start, end := span[0], span[1]
	if start < 0 || end <= start || end > n {
		return 0, 0, false, fmt.Errorf("%s=[%d,%d] must be a non-empty range within [0,%d]", name, start, end, n)
	}
	return start, end, true, nil
}

func ensureMinimalPairRecordSpans(rec *minimalPairRecord) error {
	if rec == nil {
		return fmt.Errorf("nil minimal pair record")
	}
	if minimalPairEqualIntSlices(rec.Clean, rec.Corrupt) {
		return fmt.Errorf("clean and corrupt tokens are identical; cannot derive differing span")
	}
	cStart, cEnd, cOK, err := parseMinimalPairSpan("clean_span", rec.CleanSpan, len(rec.Clean))
	if err != nil {
		return err
	}
	oStart, oEnd, oOK, err := parseMinimalPairSpan("corrupt_span", rec.CorruptSpan, len(rec.Corrupt))
	if err != nil {
		return err
	}
	if !cOK || !oOK {
		cStart, cEnd, oStart, oEnd, err = deriveMinimalPairDifferingSpans(rec.Clean, rec.Corrupt)
		if err != nil {
			return err
		}
	}
	rec.CleanSpan = []int{cStart, cEnd}
	rec.CorruptSpan = []int{oStart, oEnd}
	return nil
}

func deriveMinimalPairDifferingSpans(clean, corrupt []int) (int, int, int, int, error) {
	if len(clean) == 0 || len(corrupt) == 0 {
		return 0, 0, 0, 0, fmt.Errorf("cannot derive span from empty token sequence")
	}
	if minimalPairEqualIntSlices(clean, corrupt) {
		return 0, 0, 0, 0, fmt.Errorf("clean and corrupt tokens are identical; cannot derive differing span")
	}
	cStart, cEnd, oStart, oEnd := prefixSuffixDifferingSpans(clean, corrupt)
	if cStart < cEnd && oStart < oEnd {
		return cStart, cEnd, oStart, oEnd, nil
	}
	return lcsDifferingSpans(clean, corrupt)
}

func prefixSuffixDifferingSpans(clean, corrupt []int) (int, int, int, int) {
	prefix := 0
	for prefix < len(clean) && prefix < len(corrupt) && clean[prefix] == corrupt[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < len(clean)-prefix && suffix < len(corrupt)-prefix &&
		clean[len(clean)-1-suffix] == corrupt[len(corrupt)-1-suffix] {
		suffix++
	}
	cStart, cEnd := prefix, len(clean)-suffix
	oStart, oEnd := prefix, len(corrupt)-suffix
	return cStart, cEnd, oStart, oEnd
}

func lcsDifferingSpans(clean, corrupt []int) (int, int, int, int, error) {
	n, m := len(clean), len(corrupt)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			switch {
			case clean[i] == corrupt[j]:
				dp[i][j] = dp[i+1][j+1] + 1
			case dp[i+1][j] >= dp[i][j+1]:
				dp[i][j] = dp[i+1][j]
			default:
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	cMatched := make([]bool, n)
	oMatched := make([]bool, m)
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case clean[i] == corrupt[j]:
			cMatched[i] = true
			oMatched[j] = true
			i++
			j++
		case dp[i+1][j] >= dp[i][j+1]:
			i++
		default:
			j++
		}
	}
	cStart, cEnd := unmatchedSpan(cMatched)
	oStart, oEnd := unmatchedSpan(oMatched)
	cStart, cEnd = expandEmptySpanPair(cStart, cEnd, n, oStart)
	oStart, oEnd = expandEmptySpanPair(oStart, oEnd, m, cStart)
	if cStart >= cEnd || oStart >= oEnd {
		return 0, 0, 0, 0, fmt.Errorf("could not derive non-empty differing spans")
	}
	return cStart, cEnd, oStart, oEnd, nil
}

func unmatchedSpan(matched []bool) (int, int) {
	start := len(matched)
	end := -1
	for i, ok := range matched {
		if !ok {
			if i < start {
				start = i
			}
			end = i + 1
		}
	}
	if end < 0 {
		return 0, 0
	}
	return start, end
}

func expandEmptySpanPair(start, end, n, anchor int) (int, int) {
	if start < end {
		return start, end
	}
	if n <= 0 {
		return 0, 0
	}
	if anchor < 0 {
		anchor = 0
	}
	if anchor >= n {
		anchor = n - 1
	}
	return anchor, anchor + 1
}

func minimalPairEqualIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type minimalPairSummary struct {
	Records       int            `json:"records"`
	MaxCleanLen   int            `json:"max_clean_len"`
	MaxCorruptLen int            `json:"max_corrupt_len"`
	Families      map[string]int `json:"families"`
}

func summarizeMinimalPairs(records []minimalPairRecord) minimalPairSummary {
	s := minimalPairSummary{Records: len(records), Families: make(map[string]int)}
	for _, rec := range records {
		if len(rec.Clean) > s.MaxCleanLen {
			s.MaxCleanLen = len(rec.Clean)
		}
		if len(rec.Corrupt) > s.MaxCorruptLen {
			s.MaxCorruptLen = len(rec.Corrupt)
		}
		family := strings.TrimSpace(rec.Family)
		if family == "" {
			family = "unknown"
		}
		s.Families[family]++
	}
	return s
}

func writeMinimalPairBinary(w io.Writer, records []minimalPairRecord, vocabSize, maxLen int) error {
	header := []uint32{
		minimalPairBinaryMagic,
		minimalPairBinaryVersion,
		uint32(vocabSize),
		uint32(maxLen),
		uint32(len(records)),
		0,
	}
	for _, v := range header {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fmt.Errorf("write minimal pair binary header: %w", err)
		}
	}
	for i, rec := range records {
		if err := writeMinimalPairBinaryRecord(w, rec); err != nil {
			return fmt.Errorf("write minimal pair binary record %d: %w", i, err)
		}
	}
	return nil
}

func writeMinimalPairBinaryRecord(w io.Writer, rec minimalPairRecord) error {
	fields := []uint32{uint32(len(rec.Clean)), uint32(len(rec.Corrupt)), uint32(len(rec.ID)), uint32(len(rec.Family))}
	for _, v := range fields {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	spanFields := minimalPairSpanFields(rec)
	for _, v := range spanFields {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	for _, tok := range rec.Clean {
		if err := binary.Write(w, binary.LittleEndian, uint32(tok)); err != nil {
			return err
		}
	}
	for _, tok := range rec.Corrupt {
		if err := binary.Write(w, binary.LittleEndian, uint32(tok)); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, rec.ID); err != nil {
		return err
	}
	if _, err := io.WriteString(w, rec.Family); err != nil {
		return err
	}
	return nil
}

func minimalPairSpanFields(rec minimalPairRecord) [4]int32 {
	out := [4]int32{-1, -1, -1, -1}
	if len(rec.CleanSpan) == 2 {
		out[0] = int32(rec.CleanSpan[0])
		out[1] = int32(rec.CleanSpan[1])
	}
	if len(rec.CorruptSpan) == 2 {
		out[2] = int32(rec.CorruptSpan[0])
		out[3] = int32(rec.CorruptSpan[1])
	}
	return out
}

func decodeMinimalPairBinary(r io.Reader, source string, opts minimalPairDecodeOptions) ([]minimalPairRecord, error) {
	header := make([]uint32, 6)
	for i := range header {
		if err := binary.Read(r, binary.LittleEndian, &header[i]); err != nil {
			return nil, fmt.Errorf("%s: read binary header: %w", source, err)
		}
	}
	if header[0] != minimalPairBinaryMagic {
		return nil, fmt.Errorf("%s: invalid minimal-pair binary magic 0x%x", source, header[0])
	}
	if header[1] != minimalPairBinaryVersion && header[1] != minimalPairBinaryVersion1 {
		return nil, fmt.Errorf("%s: unsupported minimal-pair binary version %d", source, header[1])
	}
	fileVocab := int(header[2])
	if opts.VocabSize > 0 && fileVocab > 0 && fileVocab != opts.VocabSize {
		return nil, fmt.Errorf("%s: minimal-pair binary vocab_size=%d does not match config vocab_size=%d", source, fileVocab, opts.VocabSize)
	}
	recordCount := int(header[4])
	if recordCount < 0 {
		return nil, fmt.Errorf("%s: invalid record count %d", source, recordCount)
	}
	records := make([]minimalPairRecord, 0, recordCount)
	for i := 0; i < recordCount; i++ {
		rec, err := readMinimalPairBinaryRecord(r, header[1])
		if err != nil {
			return nil, fmt.Errorf("%s record %d: %w", source, i+1, err)
		}
		if err := validateMinimalPairRecordWithOptions(rec, opts); err != nil {
			return nil, fmt.Errorf("%s record %d: %w", source, i+1, err)
		}
		if minimalPairDecodeUsesDifferingSpan(opts) {
			if err := ensureMinimalPairRecordSpans(&rec); err != nil {
				return nil, fmt.Errorf("%s record %d: %w", source, i+1, err)
			}
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

func readMinimalPairBinaryRecord(r io.Reader, version uint32) (minimalPairRecord, error) {
	var lens [4]uint32
	for i := range lens {
		if err := binary.Read(r, binary.LittleEndian, &lens[i]); err != nil {
			return minimalPairRecord{}, err
		}
	}
	var spans [4]int32
	for i := range spans {
		spans[i] = -1
	}
	if version >= minimalPairBinaryVersion {
		for i := range spans {
			if err := binary.Read(r, binary.LittleEndian, &spans[i]); err != nil {
				return minimalPairRecord{}, fmt.Errorf("read spans: %w", err)
			}
		}
	}
	clean, err := readMinimalPairTokenVector(r, int(lens[0]))
	if err != nil {
		return minimalPairRecord{}, fmt.Errorf("read clean tokens: %w", err)
	}
	corrupt, err := readMinimalPairTokenVector(r, int(lens[1]))
	if err != nil {
		return minimalPairRecord{}, fmt.Errorf("read corrupt tokens: %w", err)
	}
	id, err := readMinimalPairString(r, int(lens[2]))
	if err != nil {
		return minimalPairRecord{}, fmt.Errorf("read id: %w", err)
	}
	family, err := readMinimalPairString(r, int(lens[3]))
	if err != nil {
		return minimalPairRecord{}, fmt.Errorf("read family: %w", err)
	}
	rec := minimalPairRecord{ID: id, Clean: clean, Corrupt: corrupt, Family: family}
	if spans[0] >= 0 || spans[1] >= 0 {
		rec.CleanSpan = []int{int(spans[0]), int(spans[1])}
	}
	if spans[2] >= 0 || spans[3] >= 0 {
		rec.CorruptSpan = []int{int(spans[2]), int(spans[3])}
	}
	return rec, nil
}

func readMinimalPairTokenVector(r io.Reader, n int) ([]int, error) {
	if n < 0 || n > scoreDiffusionMaxJSONLLine/4 {
		return nil, fmt.Errorf("invalid token vector length %d", n)
	}
	out := make([]int, n)
	for i := range out {
		var tok uint32
		if err := binary.Read(r, binary.LittleEndian, &tok); err != nil {
			return nil, err
		}
		out[i] = int(tok)
	}
	return out, nil
}

func readMinimalPairString(r io.Reader, n int) (string, error) {
	if n < 0 || n > scoreDiffusionMaxJSONLLine {
		return "", fmt.Errorf("invalid string length %d", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func maybeAttachMinimalPairs(sampler *minimalPairSampler, cfg *ArchConfig, step int, batch objectiveBatch, rawBatchSize, seqLen int) (objectiveBatch, error) {
	if sampler == nil || !minimalPairActive(cfg) {
		return batch, nil
	}
	if rawBatchSize <= 0 || seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid minimal-pair batch shape rows=%d seq_len=%d", rawBatchSize, seqLen)
	}
	if rawBatchSize%2 != 0 {
		return objectiveBatch{}, fmt.Errorf("minimal-pair energy training requires an even number of sequence rows per batch, got %d", rawBatchSize)
	}
	need := rawBatchSize * seqLen
	if len(batch.x) < need*len(cfg.Training.Heads) || len(batch.y) < need*len(cfg.Training.Heads) || len(batch.lossMask) < need*len(cfg.Training.Heads) {
		return objectiveBatch{}, fmt.Errorf("minimal-pair objective batch too small for heads=%d need=%d", len(cfg.Training.Heads), need)
	}
	spanMode := cfg.Training.MinimalPair.UsesDifferingSpanEnergy()
	if spanMode && len(batch.energySpanMask) < need*len(cfg.Training.Heads) {
		batch.energySpanMask = make([]float32, need*len(cfg.Training.Heads))
	}
	totalPairs := rawBatchSize / 2
	activePairs := int(math.Ceil(float64(totalPairs) * cfg.Training.MinimalPair.PairBatchFraction))
	if activePairs < 1 {
		activePairs = 1
	}
	if activePairs > totalPairs {
		activePairs = totalPairs
	}
	pad := minimalPairPadTokenID(cfg)
	for headIdx, head := range cfg.Training.Heads {
		if head.Objective != arch.ObjectiveEnergy {
			continue
		}
		rng := deterministicObjectiveRNG(cfg.Training.Seed, step, 0xeb100d5eed000001+uint64(headIdx))
		tokenOffset := headIdx * need
		rowOffset := headIdx * rawBatchSize
		clear(batch.lossMask[tokenOffset : tokenOffset+need])
		if spanMode {
			clear(batch.energySpanMask[tokenOffset : tokenOffset+need])
		}
		for pairIdx := 0; pairIdx < totalPairs; pairIdx++ {
			rec := sampler.records[0]
			active := pairIdx < activePairs
			if active {
				rec = sampler.records[rng.Intn(len(sampler.records))]
			}
			if active && spanMode {
				if err := ensureMinimalPairRecordSpans(&rec); err != nil {
					return objectiveBatch{}, fmt.Errorf("minimal-pair record %q: %w", rec.ID, err)
				}
			}
			cleanRow := tokenOffset + (2*pairIdx)*seqLen
			corruptRow := cleanRow + seqLen
			fillMinimalPairRow(batch.x[cleanRow:cleanRow+seqLen], batch.y[cleanRow:cleanRow+seqLen], rec.Clean, pad)
			fillMinimalPairRow(batch.x[corruptRow:corruptRow+seqLen], batch.y[corruptRow:corruptRow+seqLen], rec.Corrupt, pad)
			if active {
				batch.lossMask[cleanRow] = 1
				batch.lossMask[corruptRow] = 1
				if spanMode {
					fillMinimalPairSpanMask(batch.energySpanMask[cleanRow:cleanRow+seqLen], rec.CleanSpan)
					fillMinimalPairSpanMask(batch.energySpanMask[corruptRow:corruptRow+seqLen], rec.CorruptSpan)
				}
			}
			if len(batch.unmaskedX) >= tokenOffset+need {
				copy(batch.unmaskedX[cleanRow:cleanRow+seqLen], batch.x[cleanRow:cleanRow+seqLen])
				copy(batch.unmaskedX[corruptRow:corruptRow+seqLen], batch.x[corruptRow:corruptRow+seqLen])
			}
		}
		if len(batch.diffusionBlockStart) >= rowOffset+rawBatchSize && len(batch.diffusionBlockEnd) >= rowOffset+rawBatchSize {
			for row := 0; row < rawBatchSize; row++ {
				batch.diffusionBlockStart[rowOffset+row] = 0
				batch.diffusionBlockEnd[rowOffset+row] = int32(seqLen)
			}
		}
	}
	return batch, nil
}

func fillMinimalPairSpanMask(dst []float32, span []int) {
	clear(dst)
	start, end, ok, err := parseMinimalPairSpan("span", span, len(dst))
	if err != nil || !ok {
		return
	}
	for i := start; i < end; i++ {
		dst[i] = 1
	}
}

func fillMinimalPairRow(dstX, dstY []int, tokens []int, pad int) {
	for i := range dstX {
		dstX[i] = pad
		dstY[i] = pad
	}
	n := len(tokens)
	if n > len(dstX) {
		n = len(dstX)
	}
	copy(dstX[:n], tokens[:n])
	copy(dstY[:n], tokens[:n])
}

func minimalPairPadTokenID(cfg *ArchConfig) int {
	if cfg != nil && cfg.Training.MLMMaskTokenID >= 0 && cfg.Training.MLMMaskTokenID < cfg.VocabSize {
		return cfg.Training.MLMMaskTokenID
	}
	return 0
}
