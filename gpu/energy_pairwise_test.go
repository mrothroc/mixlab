//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestEnergyPairwiseLossMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const rows = 6
	logits := []float32{0.1, 1.1, 2.0, 1.0, 0.5, 0.3}
	mask := []float32{1, 1, 1, 1, 0, 0}
	for _, tc := range []struct {
		name string
		kind int
	}{
		{"logistic", ir.EnergyPairLossLogistic},
		{"hinge", ir.EnergyPairLossHinge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prog := ir.NewProgram(1)
			prog.DeclareInput("logits", ir.TensorFloat32, []int{rows, 1})
			prog.DeclareInput("mask", ir.TensorFloat32, []int{rows})
			prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
			prog.DeclareOutput("accuracy", ir.TensorFloat32, []int{1})
			prog.DeclareOutput("clean_mean", ir.TensorFloat32, []int{1})
			prog.DeclareOutput("corrupt_mean", ir.TensorFloat32, []int{1})
			prog.EnergyPairwiseLoss("logits", "mask", tc.kind, 1.0, "loss", "accuracy", "clean_mean", "corrupt_mean")
			gpuProg, err := LowerIRProgram(prog)
			if err != nil {
				t.Fatalf("LowerIRProgram: %v", err)
			}
			defer gpuProg.Destroy()
			dummy, err := FromData([]float32{0}, 1, 1)
			if err != nil {
				t.Fatalf("FromData: %v", err)
			}
			defer FreeHandle(dummy)
			inputs := []TensorInput{
				{Name: "logits", DType: TensorFloat32, Shape: []int{rows, 1}, Data: logits},
				{Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: mask},
			}
			gotLoss, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "loss")
			if err != nil {
				t.Fatalf("EvalProgramOutput(loss): %v", err)
			}
			gotAcc, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "accuracy")
			if err != nil {
				t.Fatalf("EvalProgramOutput(accuracy): %v", err)
			}
			gotClean, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "clean_mean")
			if err != nil {
				t.Fatalf("EvalProgramOutput(clean_mean): %v", err)
			}
			gotCorrupt, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "corrupt_mean")
			if err != nil {
				t.Fatalf("EvalProgramOutput(corrupt_mean): %v", err)
			}
			wantLoss, wantAcc, wantClean, wantCorrupt := energyPairwiseOracle(logits, mask, tc.kind, 1)
			for name, pair := range map[string][2]float32{
				"loss":         {gotLoss[0], wantLoss},
				"accuracy":     {gotAcc[0], wantAcc},
				"clean_mean":   {gotClean[0], wantClean},
				"corrupt_mean": {gotCorrupt[0], wantCorrupt},
			} {
				if math.Abs(float64(pair[0]-pair[1])) > 1e-5 {
					t.Fatalf("%s=%g want %g", name, pair[0], pair[1])
				}
			}
		})
	}
}

func TestEnergySpanPoolAndPairwiseLossMatchCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const rows = 4
	const seqLen = 3
	logits := []float32{
		1, 2, 3,
		4, 5, 7,
		9, 9, 9,
		8, 8, 8,
	}
	spanMask := []float32{
		0, 1, 0,
		0, 1, 1,
		0, 0, 0,
		0, 0, 0,
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{rows * seqLen, 1})
	prog.DeclareInput("span_mask", ir.TensorFloat32, []int{rows * seqLen})
	prog.DeclareOutput("pooled", ir.TensorFloat32, []int{rows, 1})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("accuracy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("clean_mean", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("corrupt_mean", ir.TensorFloat32, []int{1})
	prog.EnergySpanPool("logits", "span_mask", seqLen, "pooled")
	prog.EnergySpanPairwiseLoss("logits", "span_mask", seqLen, ir.EnergyPairLossLogistic, 1, "loss", "accuracy", "clean_mean", "corrupt_mean")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(dummy)
	inputs := []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{rows * seqLen, 1}, Data: logits},
		{Name: "span_mask", DType: TensorFloat32, Shape: []int{rows * seqLen}, Data: spanMask},
	}
	gotPooled, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "pooled")
	if err != nil {
		t.Fatalf("EvalProgramOutput(pooled): %v", err)
	}
	wantPooled := []float32{2, 6, 0, 0}
	for i := range wantPooled {
		if math.Abs(float64(gotPooled[i]-wantPooled[i])) > 1e-5 {
			t.Fatalf("pooled[%d]=%g want %g (all=%v)", i, gotPooled[i], wantPooled[i], gotPooled)
		}
	}
	gotLoss, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput(loss): %v", err)
	}
	gotAcc, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "accuracy")
	if err != nil {
		t.Fatalf("EvalProgramOutput(accuracy): %v", err)
	}
	gotClean, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "clean_mean")
	if err != nil {
		t.Fatalf("EvalProgramOutput(clean_mean): %v", err)
	}
	gotCorrupt, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "corrupt_mean")
	if err != nil {
		t.Fatalf("EvalProgramOutput(corrupt_mean): %v", err)
	}
	wantLoss, wantAcc, wantClean, wantCorrupt := energyPairwiseOracle(wantPooled, []float32{1, 1, 0, 0}, ir.EnergyPairLossLogistic, 1)
	for name, pair := range map[string][2]float32{
		"loss":         {gotLoss[0], wantLoss},
		"accuracy":     {gotAcc[0], wantAcc},
		"clean_mean":   {gotClean[0], wantClean},
		"corrupt_mean": {gotCorrupt[0], wantCorrupt},
	} {
		if math.Abs(float64(pair[0]-pair[1])) > 1e-5 {
			t.Fatalf("%s=%g want %g", name, pair[0], pair[1])
		}
	}
}

func TestSpanPLLPoolAndPairwiseLossMatchCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const rows = 4
	const seqLen = 2
	const vocab = 3
	logits := []float32{
		2, 0, 0,
		0, 2, 0,
		0, 2, 0,
		0, 0, 2,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
	}
	targets := []int32{0, 1, 1, 2, 0, 0, 1, 1}
	spanMask := []float32{1, 1, 1, 1, 0, 0, 0, 0}
	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{rows * seqLen, vocab})
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows * seqLen})
	prog.DeclareInput("span_mask", ir.TensorFloat32, []int{rows * seqLen})
	prog.DeclareOutput("scores", ir.TensorFloat32, []int{rows, 1})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("accuracy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("clean_mean", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("corrupt_mean", ir.TensorFloat32, []int{1})
	prog.SpanPLLPool("logits", "targets", "span_mask", seqLen, "scores")
	prog.SpanPLLPairwiseLoss("logits", "targets", "span_mask", seqLen, ir.EnergyPairLossLogistic, 1, "loss", "accuracy", "clean_mean", "corrupt_mean")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(dummy)
	inputs := []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{rows * seqLen, vocab}, Data: logits},
		{Name: "targets", DType: TensorInt32, Shape: []int{rows * seqLen}, Data: targets},
		{Name: "span_mask", DType: TensorFloat32, Shape: []int{rows * seqLen}, Data: spanMask},
	}
	gotScores, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "scores")
	if err != nil {
		t.Fatalf("EvalProgramOutput(scores): %v", err)
	}
	wantScores := spanPLLPoolOracle(logits, targets, spanMask, rows, seqLen, vocab)
	for i := range wantScores {
		if math.Abs(float64(gotScores[i]-wantScores[i])) > 1e-5 {
			t.Fatalf("scores[%d]=%g want %g (all=%v)", i, gotScores[i], wantScores[i], gotScores)
		}
	}
	gotLoss, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput(loss): %v", err)
	}
	gotAcc, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "accuracy")
	if err != nil {
		t.Fatalf("EvalProgramOutput(accuracy): %v", err)
	}
	gotClean, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "clean_mean")
	if err != nil {
		t.Fatalf("EvalProgramOutput(clean_mean): %v", err)
	}
	gotCorrupt, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "corrupt_mean")
	if err != nil {
		t.Fatalf("EvalProgramOutput(corrupt_mean): %v", err)
	}
	wantLoss, wantAcc, wantClean, wantCorrupt := spanPLLPairwiseOracle(wantScores, []float32{1, 1, 0, 0}, ir.EnergyPairLossLogistic, 1)
	for name, pair := range map[string][2]float32{
		"loss":         {gotLoss[0], wantLoss},
		"accuracy":     {gotAcc[0], wantAcc},
		"clean_mean":   {gotClean[0], wantClean},
		"corrupt_mean": {gotCorrupt[0], wantCorrupt},
	} {
		if math.Abs(float64(pair[0]-pair[1])) > 1e-5 {
			t.Fatalf("%s=%g want %g", name, pair[0], pair[1])
		}
	}
}

func energyPairwiseOracle(logits, mask []float32, kind int, margin float64) (float32, float32, float32, float32) {
	lossSum := 0.0
	correct := 0.0
	cleanSum := 0.0
	corruptSum := 0.0
	count := 0.0
	for i := 0; i+1 < len(logits); i += 2 {
		if mask[i] <= 0 || mask[i+1] <= 0 {
			continue
		}
		clean := float64(logits[i])
		corrupt := float64(logits[i+1])
		diff := clean - corrupt
		if kind == ir.EnergyPairLossHinge {
			lossSum += math.Max(0, margin+diff)
		} else {
			lossSum += math.Max(diff, 0) + math.Log1p(math.Exp(-math.Abs(diff)))
		}
		if clean < corrupt {
			correct++
		}
		cleanSum += clean
		corruptSum += corrupt
		count++
	}
	if count == 0 {
		return 0, 0, 0, 0
	}
	return float32(lossSum / count), float32(correct / count), float32(cleanSum / count), float32(corruptSum / count)
}

func spanPLLPoolOracle(logits []float32, targets []int32, mask []float32, rows, seqLen, vocab int) []float32 {
	out := make([]float32, rows)
	for row := 0; row < rows; row++ {
		sum := 0.0
		for pos := 0; pos < seqLen; pos++ {
			idx := row*seqLen + pos
			if mask[idx] <= 0 {
				continue
			}
			base := idx * vocab
			maxLogit := float64(logits[base])
			for v := 1; v < vocab; v++ {
				if val := float64(logits[base+v]); val > maxLogit {
					maxLogit = val
				}
			}
			expSum := 0.0
			for v := 0; v < vocab; v++ {
				expSum += math.Exp(float64(logits[base+v]) - maxLogit)
			}
			sum += float64(logits[base+int(targets[idx])]) - maxLogit - math.Log(expSum)
		}
		out[row] = float32(sum)
	}
	return out
}

func spanPLLPairwiseOracle(scores, mask []float32, kind int, margin float64) (float32, float32, float32, float32) {
	lossSum := 0.0
	correct := 0.0
	cleanSum := 0.0
	corruptSum := 0.0
	count := 0.0
	for i := 0; i+1 < len(scores); i += 2 {
		if mask[i] <= 0 || mask[i+1] <= 0 {
			continue
		}
		clean := float64(scores[i])
		corrupt := float64(scores[i+1])
		diff := corrupt - clean
		if kind == ir.EnergyPairLossHinge {
			lossSum += math.Max(0, margin+diff)
		} else {
			lossSum += math.Max(diff, 0) + math.Log1p(math.Exp(-math.Abs(diff)))
		}
		if clean > corrupt {
			correct++
		}
		cleanSum += clean
		corruptSum += corrupt
		count++
	}
	if count == 0 {
		return 0, 0, 0, 0
	}
	return float32(lossSum / count), float32(correct / count), float32(cleanSum / count), float32(corruptSum / count)
}
