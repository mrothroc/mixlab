package data

import (
	"fmt"
	"path/filepath"
	"sort"
)

// NewClassificationValSet loads a labeled record split exactly once. A
// maxBatches value of zero evaluates the full split; a positive value caps the
// number of fixed-shape batches while retaining the split total for reporting.
func NewClassificationValSet(pattern string, maxBatches, batchTokens, seqLen int) (*ValSet, error) {
	return NewClassificationValSetWithOptions(pattern, maxBatches, batchTokens, seqLen, LoaderOptions{})
}

// NewClassificationValSetWithOptions loads classification validation data,
// optionally using the same deterministic length buckets as training.
func NewClassificationValSetWithOptions(pattern string, maxBatches, batchTokens, seqLen int, opts LoaderOptions) (*ValSet, error) {
	if maxBatches < 0 {
		return nil, fmt.Errorf("classification validation batch limit must be >= 0, got %d", maxBatches)
	}
	if batchTokens <= 0 || seqLen <= 0 || (len(opts.LengthBuckets) == 0 && batchTokens%seqLen != 0) {
		return nil, fmt.Errorf("invalid classification validation batch shape: batchTokens=%d seqLen=%d", batchTokens, seqLen)
	}

	manifest, _, found, err := FindDatasetManifest(pattern)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("classification validation requires %s beside the labeled shards", DatasetManifestFilename)
	}
	if len(opts.LengthBuckets) > 0 {
		return newBucketedClassificationValSet(pattern, manifest, maxBatches, batchTokens, seqLen, opts)
	}
	if isContinuousSequenceShardFormat(manifest.ShardFormat) {
		return newContinuousClassificationValSet(pattern, manifest, maxBatches, batchTokens, seqLen)
	}
	if manifest.ShardFormat == DatasetShardFormatCodebookSequenceV1 {
		return newCodebookClassificationValSet(pattern, manifest, maxBatches, batchTokens, seqLen)
	}
	if manifest.ShardFormat != DatasetShardFormatLabeledSequenceV1 ||
		manifest.EffectiveSequenceLayout() != DatasetSequenceLayoutOneRecordRow {
		return nil, fmt.Errorf(
			"classification validation requires shard_format=%q and sequence_layout=%q",
			DatasetShardFormatLabeledSequenceV1, DatasetSequenceLayoutOneRecordRow,
		)
	}
	if seqLen != manifest.RecordSeqLen {
		return nil, fmt.Errorf("classification validation requires seq_len=%d, got %d", manifest.RecordSeqLen, seqLen)
	}
	padID, padOK := manifest.SpecialTokenIDs["pad"]
	bosID, bosOK := manifest.SpecialTokenIDs["bos"]
	eosID, eosOK := manifest.SpecialTokenIDs["eos"]
	if !padOK || !bosOK || !eosOK {
		return nil, fmt.Errorf("classification validation manifest requires special_token_ids pad, bos, and eos")
	}

	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no classification validation shard files matched %q", pattern)
	}

	var records [][]uint16
	var labels []int32
	for _, file := range files {
		shardRecords, shardLabels, err := LoadLabeledSequenceShard(file)
		if err != nil {
			return nil, err
		}
		records = append(records, shardRecords...)
		labels = append(labels, shardLabels...)
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("classification validation split %q contains no records", pattern)
	}
	if len(labels) != len(records) {
		return nil, fmt.Errorf("classification validation split has %d records but %d labels", len(records), len(labels))
	}

	batchSize := batchTokens / seqLen
	evaluated := len(records)
	if maxBatches > 0 && maxBatches*batchSize < evaluated {
		evaluated = maxBatches * batchSize
	}
	batchCount := (evaluated + batchSize - 1) / batchSize
	vs := &ValSet{
		Batches:           make([]ValBatch, 0, batchCount),
		TotalExamples:     len(records),
		EvaluatedExamples: evaluated,
	}
	for start := 0; start < evaluated; start += batchSize {
		realRows := minInt(batchSize, evaluated-start)
		batch := newRecordBatch(batchTokens, seqLen, true)
		for row := 0; row < realRows; row++ {
			if err := frameRecordRow(&batch, row, records[start+row], labels[start+row], seqLen, padID, bosID, eosID, true); err != nil {
				return nil, fmt.Errorf("frame classification validation record %d: %w", start+row, err)
			}
		}
		// Fixed-shape GPU programs cannot accept a partial final batch. Duplicate
		// its first real row, then let ExampleCount exclude those rows.
		for row := realRows; row < batchSize; row++ {
			if err := frameRecordRow(&batch, row, records[start], labels[start], seqLen, padID, bosID, eosID, true); err != nil {
				return nil, fmt.Errorf("pad classification validation batch: %w", err)
			}
		}
		vs.Batches = append(vs.Batches, ValBatch{
			X: batch.X, Y: batch.Y, LossMask: batch.LossMask,
			SegmentIDs: batch.SegmentIDs, MaskEligible: batch.MaskEligible,
			Labels: batch.Labels, ValidMask: batch.ValidMask,
			ExampleCount: realRows,
		})
	}
	return vs, nil
}

func newBucketedClassificationValSet(pattern string, manifest *DatasetManifest, maxBatches, batchTokens, seqLen int, opts LoaderOptions) (*ValSet, error) {
	if manifest == nil || (!isContinuousSequenceShardFormat(manifest.ShardFormat) && manifest.ShardFormat != DatasetShardFormatCodebookSequenceV1) {
		return nil, fmt.Errorf("length-bucketed classification validation requires continuous-frame or discrete-codebook shards")
	}
	if seqLen < manifest.RecordSeqLen {
		return nil, fmt.Errorf("classification validation config seq_len=%d is smaller than dataset record_seq_len=%d", seqLen, manifest.RecordSeqLen)
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no classification validation shard files matched %q", pattern)
	}
	total := 0
	for _, file := range files {
		var lengths []int32
		if manifest.ShardFormat == DatasetShardFormatCodebookSequenceV1 {
			lengths, _, _, _, err = loadCodebookSequenceLengths(file)
		} else {
			lengths, _, _, err = loadContinuousSequenceLengths(file)
		}
		if err != nil {
			return nil, err
		}
		total += len(lengths)
	}
	opts.NoShardShuffle = true
	loader, err := NewLoaderWithOptions(pattern, 0, opts)
	if err != nil {
		return nil, err
	}
	vs := &ValSet{Batches: make([]ValBatch, 0), TotalExamples: total}
	for vs.EvaluatedExamples < total && (maxBatches == 0 || len(vs.Batches) < maxBatches) {
		batch, err := loader.NextBatchDetailed(batchTokens, seqLen)
		if err != nil {
			return nil, err
		}
		if batch.ExampleCount <= 0 {
			return nil, fmt.Errorf("length-bucketed validation batch has no real examples")
		}
		vs.Batches = append(vs.Batches, ValBatch(batch))
		vs.EvaluatedExamples += batch.ExampleCount
	}
	if len(vs.Batches) == 0 {
		return nil, fmt.Errorf("classification validation split %q contains no records", pattern)
	}
	return vs, nil
}

func newCodebookClassificationValSet(
	pattern string,
	manifest *DatasetManifest,
	maxBatches, batchTokens, seqLen int,
) (*ValSet, error) {
	if manifest == nil || manifest.Representation != DatasetRepresentationDiscreteCodebooks {
		return nil, fmt.Errorf("codebook classification validation requires a discrete codebook manifest")
	}
	if seqLen != manifest.RecordSeqLen {
		return nil, fmt.Errorf("codebook classification validation requires seq_len=%d, got %d", manifest.RecordSeqLen, seqLen)
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no codebook classification validation shard files matched %q", pattern)
	}
	Q := manifest.NumCodebooks
	var tokens, lengths, labels []int32
	for _, file := range files {
		shard, err := LoadCodebookSequenceShard(file)
		if err != nil {
			return nil, err
		}
		if shard.SeqLen != seqLen || shard.NumCodebooks != Q || shard.CodebookVocabSize != manifest.CodebookVocabSize {
			return nil, fmt.Errorf("codebook validation shard %q domain does not match manifest", file)
		}
		tokens = append(tokens, shard.Tokens...)
		lengths = append(lengths, shard.Lengths...)
		labels = append(labels, shard.Labels...)
	}
	if len(labels) == 0 {
		return nil, fmt.Errorf("codebook classification validation split %q contains no records", pattern)
	}
	batchSize := batchTokens / seqLen
	evaluated := len(labels)
	if maxBatches > 0 && maxBatches*batchSize < evaluated {
		evaluated = maxBatches * batchSize
	}
	batchCount := (evaluated + batchSize - 1) / batchSize
	vs := &ValSet{Batches: make([]ValBatch, 0, batchCount), TotalExamples: len(labels), EvaluatedExamples: evaluated}
	recordWidth := seqLen * Q
	for start := 0; start < evaluated; start += batchSize {
		realRows := minInt(batchSize, evaluated-start)
		batchTokensData := make([]int32, batchSize*recordWidth)
		batchLabels := make([]int32, batchSize)
		validMask := make([]float32, batchTokens)
		for row := 0; row < realRows; row++ {
			source := start + row
			copy(batchTokensData[row*recordWidth:], tokens[source*recordWidth:(source+1)*recordWidth])
			batchLabels[row] = labels[source]
			for pos := 0; pos < int(lengths[source]); pos++ {
				validMask[row*seqLen+pos] = 1
			}
		}
		for row := realRows; row < batchSize; row++ {
			copy(batchTokensData[row*recordWidth:], batchTokensData[:recordWidth])
			batchLabels[row] = batchLabels[0]
			copy(validMask[row*seqLen:(row+1)*seqLen], validMask[:seqLen])
		}
		vs.Batches = append(vs.Batches, ValBatch{
			Codebooks: batchTokensData, Labels: batchLabels, ValidMask: validMask, ExampleCount: realRows,
		})
	}
	return vs, nil
}

func newContinuousClassificationValSet(
	pattern string,
	manifest *DatasetManifest,
	maxBatches, batchTokens, seqLen int,
) (*ValSet, error) {
	if manifest == nil || manifest.Representation != DatasetRepresentationContinuousFrames {
		return nil, fmt.Errorf("continuous classification validation requires a continuous frame manifest")
	}
	if seqLen != manifest.RecordSeqLen {
		return nil, fmt.Errorf("continuous classification validation requires seq_len=%d, got %d", manifest.RecordSeqLen, seqLen)
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no continuous classification validation shard files matched %q", pattern)
	}
	featureDim := manifest.FeatureDim
	var frames []float32
	var labels []int32
	for _, file := range files {
		shard, err := LoadContinuousSequenceShard(file)
		if err != nil {
			return nil, err
		}
		if shard.SeqLen != seqLen || shard.FeatureDim != featureDim {
			return nil, fmt.Errorf(
				"continuous validation shard %q shape [T=%d,F=%d] does not match manifest [T=%d,F=%d]",
				file, shard.SeqLen, shard.FeatureDim, seqLen, featureDim,
			)
		}
		frames = append(frames, shard.Frames...)
		labels = append(labels, shard.Labels...)
	}
	if len(labels) == 0 {
		return nil, fmt.Errorf("continuous classification validation split %q contains no records", pattern)
	}
	batchSize := batchTokens / seqLen
	evaluated := len(labels)
	if maxBatches > 0 && maxBatches*batchSize < evaluated {
		evaluated = maxBatches * batchSize
	}
	batchCount := (evaluated + batchSize - 1) / batchSize
	vs := &ValSet{
		Batches: make([]ValBatch, 0, batchCount), TotalExamples: len(labels), EvaluatedExamples: evaluated,
	}
	recordWidth := seqLen * featureDim
	for start := 0; start < evaluated; start += batchSize {
		realRows := minInt(batchSize, evaluated-start)
		batchFrames := make([]float32, batchSize*recordWidth)
		batchLabels := make([]int32, batchSize)
		validMask := make([]float32, batchTokens)
		for row := 0; row < realRows; row++ {
			source := start + row
			copy(batchFrames[row*recordWidth:], frames[source*recordWidth:(source+1)*recordWidth])
			batchLabels[row] = labels[source]
		}
		for row := realRows; row < batchSize; row++ {
			copy(batchFrames[row*recordWidth:], batchFrames[:recordWidth])
			batchLabels[row] = batchLabels[0]
		}
		for i := range validMask {
			validMask[i] = 1
		}
		vs.Batches = append(vs.Batches, ValBatch{
			Frames: batchFrames, Labels: batchLabels, ValidMask: validMask, ExampleCount: realRows,
		})
	}
	return vs, nil
}

func newRecordBatch(batchTokens, seqLen int, labeled bool) Batch {
	batch := Batch{
		X:            make([]int, batchTokens),
		Y:            make([]int, batchTokens),
		LossMask:     make([]float32, batchTokens),
		MaskEligible: make([]uint8, batchTokens),
	}
	if labeled {
		batch.SegmentIDs = make([]int32, batchTokens)
		batch.Labels = make([]int32, batchTokens/seqLen)
		batch.ValidMask = make([]float32, batchTokens)
	}
	return batch
}

func frameRecordRow(batch *Batch, row int, record []uint16, label int32, seqLen, padID, bosID, eosID int, labeled bool) error {
	if len(record) > seqLen-2 {
		return fmt.Errorf("record has %d tokens but seq_len=%d permits at most %d", len(record), seqLen, seqLen-2)
	}
	rowStart := row * seqLen
	rowEnd := rowStart + seqLen
	if row < 0 || rowEnd > len(batch.X) || (labeled && row >= len(batch.Labels)) {
		return fmt.Errorf("row %d is outside fixed batch shape", row)
	}
	for i := rowStart; i < rowEnd; i++ {
		batch.X[i], batch.Y[i] = padID, padID
		batch.MaskEligible[i] = 0
		batch.LossMask[i] = 0
		if labeled {
			batch.SegmentIDs[i] = 0
			batch.ValidMask[i] = 0
		}
	}
	batch.X[rowStart] = bosID
	if labeled {
		batch.ValidMask[rowStart] = 1
	}
	for i, token := range record {
		pos := rowStart + i + 1
		batch.X[pos] = int(token)
		batch.MaskEligible[pos] = 1
		if labeled {
			batch.ValidMask[pos] = 1
		}
	}
	eosInput := rowStart + len(record) + 1
	batch.X[eosInput] = eosID
	if labeled {
		batch.ValidMask[eosInput] = 1
	}
	for i := rowStart; i < eosInput; i++ {
		batch.Y[i] = batch.X[i+1]
		batch.LossMask[i] = 1
	}
	if labeled {
		for i := eosInput + 1; i < rowEnd; i++ {
			batch.SegmentIDs[i] = 1
		}
		batch.Labels[row] = label
	}
	return nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
