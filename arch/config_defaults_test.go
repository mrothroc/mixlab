package arch

import "testing"

func TestTrainingSpecApplyDefaultsDerivesLearningRates(t *testing.T) {
	spec := TrainingSpec{LR: 1e-3}
	spec.ApplyDefaults()

	const wantLR = float32(1e-3)
	if spec.EmbedLR != wantLR || spec.MatrixLR != wantLR || spec.ScalarLR != wantLR || spec.HeadLR != wantLR {
		t.Fatalf("per-class LRs = embed:%g matrix:%g scalar:%g head:%g, want all %g",
			spec.EmbedLR, spec.MatrixLR, spec.ScalarLR, spec.HeadLR, wantLR)
	}

	defaults := DefaultTrainingSpec()
	if spec.Beta1 != defaults.Beta1 {
		t.Fatalf("Beta1=%g, want %g", spec.Beta1, defaults.Beta1)
	}
	if spec.Beta2 != defaults.Beta2 {
		t.Fatalf("Beta2=%g, want %g", spec.Beta2, defaults.Beta2)
	}
	if spec.Epsilon != defaults.Epsilon {
		t.Fatalf("Epsilon=%g, want %g", spec.Epsilon, defaults.Epsilon)
	}
	if spec.TTTLR != defaults.TTTLR {
		t.Fatalf("TTTLR=%g, want %g", spec.TTTLR, defaults.TTTLR)
	}
	if spec.TTTMode != defaults.TTTMode {
		t.Fatalf("TTTMode=%q, want %q", spec.TTTMode, defaults.TTTMode)
	}
	if spec.QAT != defaults.QAT {
		t.Fatalf("QAT=%q, want %q", spec.QAT, defaults.QAT)
	}
	if spec.MuonMomentum != defaults.MuonMomentum {
		t.Fatalf("MuonMomentum=%g, want %g", spec.MuonMomentum, defaults.MuonMomentum)
	}
	if spec.MuonBackendSteps != defaults.MuonBackendSteps {
		t.Fatalf("MuonBackendSteps=%d, want %d", spec.MuonBackendSteps, defaults.MuonBackendSteps)
	}
	if spec.SWADecay != defaults.SWADecay {
		t.Fatalf("SWADecay=%g, want %g", spec.SWADecay, defaults.SWADecay)
	}
	if spec.SWAInterval != defaults.SWAInterval {
		t.Fatalf("SWAInterval=%d, want %d", spec.SWAInterval, defaults.SWAInterval)
	}
}
