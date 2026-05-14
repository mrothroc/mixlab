//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/logits"
	"github.com/mrothroc/mixlab/logprobs"
	"github.com/mrothroc/mixlab/ranks"
	"github.com/mrothroc/mixlab/uncertainty"
)

// TestRunEvalLogitsAlignedWithLogprobsFloat32Raw runs eval with both
// -logprobs-out and -logits-out (float32, raw) and validates the
// position-by-position acceptance criteria from the feature spec:
//   - records align by token ID with logprobs.bin
//   - file matches the documented binary format (magic / vocab / total tokens)
//   - logsumexp(logits[i]) - logits[i, token_id[i+1]] recovers NLL[i] from
//     -logprobs-out's output within float tolerance
func TestRunEvalLogitsAlignedWithLogprobsFloat32Raw(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	logitsPath := filepath.Join(dataDir, "logits.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogprobsOut: logprobsPath,
		LogitsOut:   logitsPath,
		LogitsDType: logits.DTypeFloat32,
		LogitsForm:  logits.FormRaw,
	}); err != nil {
		t.Fatalf("RunEvalExports: %v", err)
	}

	logprobBlob, err := os.ReadFile(logprobsPath)
	if err != nil {
		t.Fatalf("ReadFile(logprobs): %v", err)
	}
	logprobHeader, logprobRecs, err := logprobs.Read(bytes.NewReader(logprobBlob))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}

	logitsBlob, err := os.ReadFile(logitsPath)
	if err != nil {
		t.Fatalf("ReadFile(logits): %v", err)
	}
	logitsHeader, logitsRecs, err := logits.Read(bytes.NewReader(logitsBlob))
	if err != nil {
		t.Fatalf("logits.Read: %v", err)
	}
	if logitsHeader.Magic != logits.Magic {
		t.Fatalf("logits magic = %#x, want %#x", logitsHeader.Magic, logits.Magic)
	}
	if logitsHeader.DType != logits.DTypeFloat32 || logitsHeader.Form != logits.FormRaw {
		t.Fatalf("logits header dtype=%s form=%s, want float32/raw", logitsHeader.DType, logitsHeader.Form)
	}
	if logitsHeader.VocabSize != logprobHeader.VocabSize {
		t.Fatalf("vocab mismatch: logprobs=%d logits=%d", logprobHeader.VocabSize, logitsHeader.VocabSize)
	}
	if logitsHeader.TotalTokens != logprobHeader.TotalTokens {
		t.Fatalf("totalTokens mismatch: logprobs=%d logits=%d", logprobHeader.TotalTokens, logitsHeader.TotalTokens)
	}
	if len(logitsRecs) != len(logprobRecs) {
		t.Fatalf("record count mismatch: logprobs=%d logits=%d", len(logprobRecs), len(logitsRecs))
	}

	vocab := int(logitsHeader.VocabSize)
	for i, rec := range logitsRecs {
		if rec.TokenID != logprobRecs[i].TokenID {
			t.Fatalf("record[%d] token mismatch: logprobs=%d logits=%d", i, logprobRecs[i].TokenID, rec.TokenID)
		}
		if len(rec.Values) != vocab {
			t.Fatalf("record[%d] values length = %d, want %d", i, len(rec.Values), vocab)
		}

		// Numerical consistency: logsumexp(row) - row[target] should recover
		// the NLL from -logprobs-out for the same position.
		maxV := rec.Values[0]
		for _, v := range rec.Values[1:] {
			if v > maxV {
				maxV = v
			}
		}
		sumExp := 0.0
		for _, v := range rec.Values {
			sumExp += math.Exp(float64(v - maxV))
		}
		logNorm := float64(maxV) + math.Log(sumExp)
		recoveredNLL := logNorm - float64(rec.Values[rec.TokenID])
		want := float64(logprobRecs[i].NLL)
		if math.Abs(recoveredNLL-want) > 1e-4 {
			t.Fatalf("record[%d] recovered NLL = %g, want ≈ %g (diff=%g)", i, recoveredNLL, want, math.Abs(recoveredNLL-want))
		}
	}
}

// TestRunEvalLogitsFloat16RawNLLParity is the same NLL-recovery acceptance
// check but at the default on-disk dtype (float16) with raw logits. Float16
// quantisation widens the tolerance versus float32.
func TestRunEvalLogitsFloat16RawNLLParity(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	logitsPath := filepath.Join(dataDir, "logits.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogprobsOut: logprobsPath,
		LogitsOut:   logitsPath,
		LogitsDType: logits.DTypeFloat16,
		LogitsForm:  logits.FormRaw,
	}); err != nil {
		t.Fatalf("RunEvalExports: %v", err)
	}

	logprobBlob, err := os.ReadFile(logprobsPath)
	if err != nil {
		t.Fatalf("ReadFile(logprobs): %v", err)
	}
	_, logprobRecs, err := logprobs.Read(bytes.NewReader(logprobBlob))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}

	logitsBlob, err := os.ReadFile(logitsPath)
	if err != nil {
		t.Fatalf("ReadFile(logits): %v", err)
	}
	header, logitsRecs, err := logits.Read(bytes.NewReader(logitsBlob))
	if err != nil {
		t.Fatalf("logits.Read: %v", err)
	}
	if header.DType != logits.DTypeFloat16 {
		t.Fatalf("dtype = %s, want float16", header.DType)
	}

	for i, rec := range logitsRecs {
		maxV := rec.Values[0]
		for _, v := range rec.Values[1:] {
			if v > maxV {
				maxV = v
			}
		}
		sumExp := 0.0
		for _, v := range rec.Values {
			sumExp += math.Exp(float64(v - maxV))
		}
		logNorm := float64(maxV) + math.Log(sumExp)
		recoveredNLL := logNorm - float64(rec.Values[rec.TokenID])
		want := float64(logprobRecs[i].NLL)
		// Float16 quantises ~3 decimal digits — allow up to 5e-2 absolute, or
		// 5% relative on larger NLLs.
		tol := 5e-2
		if relTol := math.Abs(want) * 0.05; relTol > tol {
			tol = relTol
		}
		if math.Abs(recoveredNLL-want) > tol {
			t.Fatalf("record[%d] recovered NLL = %g, want ≈ %g (diff=%g tol=%g)", i, recoveredNLL, want, math.Abs(recoveredNLL-want), tol)
		}
	}
}

// TestRunEvalLogitsLogprobsForm verifies that when -logits-form=logprobs is
// requested the stored rows are valid log-probabilities (sum-exp ≈ 1) and
// recover the same NLL.
func TestRunEvalLogitsLogprobsForm(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	logitsPath := filepath.Join(dataDir, "logits.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogprobsOut: logprobsPath,
		LogitsOut:   logitsPath,
		LogitsDType: logits.DTypeFloat32,
		LogitsForm:  logits.FormLogprobs,
	}); err != nil {
		t.Fatalf("RunEvalExports: %v", err)
	}

	_, logprobRecs, err := logprobs.Read(readFile(t, logprobsPath))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}
	header, logitsRecs, err := logits.Read(readFile(t, logitsPath))
	if err != nil {
		t.Fatalf("logits.Read: %v", err)
	}
	if header.Form != logits.FormLogprobs {
		t.Fatalf("form = %s, want logprobs", header.Form)
	}

	for i, rec := range logitsRecs {
		// exp(logprobs) sums to ≈1.
		sumProb := 0.0
		for _, v := range rec.Values {
			sumProb += math.Exp(float64(v))
		}
		if math.Abs(sumProb-1.0) > 1e-4 {
			t.Fatalf("record[%d] sum(exp(logprobs)) = %g, want ≈1", i, sumProb)
		}
		// -logprobs[target] == NLL[i].
		got := -float64(rec.Values[rec.TokenID])
		want := float64(logprobRecs[i].NLL)
		if math.Abs(got-want) > 1e-4 {
			t.Fatalf("record[%d] -logprob[target] = %g, want ≈ %g", i, got, want)
		}
	}
}

// TestRunEvalLogitsOnly verifies that -logits-out alone (no other export
// flags) produces a valid logits file.
func TestRunEvalLogitsOnly(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logitsPath := filepath.Join(dataDir, "logits.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogitsOut:   logitsPath,
		LogitsDType: logits.DTypeFloat16,
		LogitsForm:  logits.FormRaw,
	}); err != nil {
		t.Fatalf("RunEvalExports(logits only): %v", err)
	}
	header, recs, err := logits.Read(readFile(t, logitsPath))
	if err != nil {
		t.Fatalf("logits.Read: %v", err)
	}
	if header.Magic != logits.Magic {
		t.Fatalf("magic = %#x, want %#x", header.Magic, logits.Magic)
	}
	if len(recs) == 0 {
		t.Fatal("no records emitted")
	}
}

// TestRunEvalRejectsAllEmpty makes sure RunEvalExports fails fast when no
// export path is set.
func TestRunEvalRejectsAllEmpty(t *testing.T) {
	fixture := newInferenceSessionFixture(t)
	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{}); err == nil {
		t.Fatal("RunEvalExports with all empty outputs succeeded, want error")
	}
}

func readFile(t *testing.T, path string) *bytes.Reader {
	t.Helper()
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", path, err)
	}
	return bytes.NewReader(blob)
}

// TestRunEvalAllFourExportsAligned runs all four per-token export flags at
// once and asserts the feature-spec acceptance criterion: TotalTokens matches
// across all four files and TokenID matches per index. This is the only test
// that exercises the simultaneous emission of logprobs, ranks, uncertainty,
// and logits from a single eval pass.
func TestRunEvalAllFourExportsAligned(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	ranksPath := filepath.Join(dataDir, "ranks.bin")
	uncertaintyPath := filepath.Join(dataDir, "uncertainty.bin")
	logitsPath := filepath.Join(dataDir, "logits.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogprobsOut:    logprobsPath,
		RanksOut:       ranksPath,
		UncertaintyOut: uncertaintyPath,
		LogitsOut:      logitsPath,
		LogitsDType:    logits.DTypeFloat32,
		LogitsForm:     logits.FormRaw,
	}); err != nil {
		t.Fatalf("RunEvalExports(all four): %v", err)
	}

	lpHeader, lpRecs, err := logprobs.Read(readFile(t, logprobsPath))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}
	rkHeader, rkRecs, err := ranks.Read(readFile(t, ranksPath))
	if err != nil {
		t.Fatalf("ranks.Read: %v", err)
	}
	uHeader, uRecs, err := uncertainty.Read(readFile(t, uncertaintyPath))
	if err != nil {
		t.Fatalf("uncertainty.Read: %v", err)
	}
	lgHeader, lgRecs, err := logits.Read(readFile(t, logitsPath))
	if err != nil {
		t.Fatalf("logits.Read: %v", err)
	}

	// TotalTokens / VocabSize agree across all four files.
	if lpHeader.TotalTokens != rkHeader.TotalTokens || lpHeader.TotalTokens != uHeader.TotalTokens || lpHeader.TotalTokens != lgHeader.TotalTokens {
		t.Fatalf("TotalTokens mismatch: logprobs=%d ranks=%d uncertainty=%d logits=%d",
			lpHeader.TotalTokens, rkHeader.TotalTokens, uHeader.TotalTokens, lgHeader.TotalTokens)
	}
	if lpHeader.VocabSize != rkHeader.VocabSize || lpHeader.VocabSize != uHeader.VocabSize || lpHeader.VocabSize != lgHeader.VocabSize {
		t.Fatalf("VocabSize mismatch: logprobs=%d ranks=%d uncertainty=%d logits=%d",
			lpHeader.VocabSize, rkHeader.VocabSize, uHeader.VocabSize, lgHeader.VocabSize)
	}

	n := len(lpRecs)
	if len(rkRecs) != n || len(uRecs) != n || len(lgRecs) != n {
		t.Fatalf("record count mismatch: logprobs=%d ranks=%d uncertainty=%d logits=%d",
			n, len(rkRecs), len(uRecs), len(lgRecs))
	}

	// Per-index TokenID alignment across all four exports.
	for i := range n {
		tok := lpRecs[i].TokenID
		if rkRecs[i].TokenID != tok || uRecs[i].TokenID != tok || lgRecs[i].TokenID != tok {
			t.Fatalf("record[%d] TokenID mismatch: logprobs=%d ranks=%d uncertainty=%d logits=%d",
				i, tok, rkRecs[i].TokenID, uRecs[i].TokenID, lgRecs[i].TokenID)
		}
	}
}

// TestRunEvalLogitsFloat16LogprobsParity closes the last corner of the
// {dtype × form} matrix: float16 storage of pre-log-softmaxed values. The
// stored row is a discrete log-prob distribution that should sum to ≈1 in
// probability space (with float16 tolerance) and recover the per-token NLL.
func TestRunEvalLogitsFloat16LogprobsParity(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	logitsPath := filepath.Join(dataDir, "logits.bin")
	if err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogprobsOut: logprobsPath,
		LogitsOut:   logitsPath,
		LogitsDType: logits.DTypeFloat16,
		LogitsForm:  logits.FormLogprobs,
	}); err != nil {
		t.Fatalf("RunEvalExports(float16, logprobs): %v", err)
	}

	_, logprobRecs, err := logprobs.Read(readFile(t, logprobsPath))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}
	header, logitsRecs, err := logits.Read(readFile(t, logitsPath))
	if err != nil {
		t.Fatalf("logits.Read: %v", err)
	}
	if header.DType != logits.DTypeFloat16 || header.Form != logits.FormLogprobs {
		t.Fatalf("header dtype=%s form=%s, want float16/logprobs", header.DType, header.Form)
	}

	for i, rec := range logitsRecs {
		// Probability sum check: float16 quantisation of a 1024-way
		// distribution accumulates error roughly proportional to vocab × ulp;
		// allow a generous tolerance.
		sumProb := 0.0
		for _, v := range rec.Values {
			sumProb += math.Exp(float64(v))
		}
		if math.Abs(sumProb-1.0) > 5e-2 {
			t.Fatalf("record[%d] sum(exp(logprobs)) = %g, want ≈1 (float16 tol)", i, sumProb)
		}

		got := -float64(rec.Values[rec.TokenID])
		want := float64(logprobRecs[i].NLL)
		tol := 5e-2
		if relTol := math.Abs(want) * 0.05; relTol > tol {
			tol = relTol
		}
		if math.Abs(got-want) > tol {
			t.Fatalf("record[%d] -logprob[target] = %g, want ≈ %g (diff=%g tol=%g)", i, got, want, math.Abs(got-want), tol)
		}
	}
}

// TestRunEvalRejectsDuplicateExportPaths ensures runEvalExports refuses to
// open two different writers against the same file. Without this check, two
// writers would share an inode and corrupt each other's output.
func TestRunEvalRejectsDuplicateExportPaths(t *testing.T) {
	fixture := newInferenceSessionFixture(t)
	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	dup := filepath.Join(dataDir, "shared.bin")

	err := RunEvalExports(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, EvalExportOptions{
		LogprobsOut: dup,
		LogitsOut:   dup,
		LogitsDType: logits.DTypeFloat16,
		LogitsForm:  logits.FormRaw,
	})
	if err == nil {
		t.Fatal("RunEvalExports with duplicate paths succeeded, want error")
	}
	// File must not have been created (validation runs before any os.Create).
	if _, statErr := os.Stat(dup); !os.IsNotExist(statErr) {
		t.Fatalf("duplicate-path validation should run before any file is created; got stat err=%v", statErr)
	}
}
