//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestHGRN2ScanMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B  = 2
		T  = 5
		H  = 3
		Ds = 4
		Dv = 2
	)

	q := patternedFloats(B*T*H*Ds, 0.11)
	k := patternedFloats(B*T*H*Ds, 0.07)
	v := patternedFloats(B*T*H*Dv, 0.13)
	gate := make([]float32, B*T*H*Ds)
	for i := range gate {
		gate[i] = 0.35 + 0.4*float32((i%7))/6
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("q", ir.TensorFloat32, []int{B, T, H, Ds})
	prog.DeclareInput("k", ir.TensorFloat32, []int{B, T, H, Ds})
	prog.DeclareInput("v", ir.TensorFloat32, []int{B, T, H, Dv})
	prog.DeclareInput("gate", ir.TensorFloat32, []int{B, T, H, Ds})
	prog.DeclareOutput("scan", ir.TensorFloat32, []int{B * T * H, Dv})
	prog.HGRN2Scan("q", "k", "v", "gate", "scan", B, T, H, Ds, Dv)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData(dummy): %v", err)
	}
	defer FreeHandle(dummy)

	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "q", DType: TensorFloat32, Shape: []int{B, T, H, Ds}, Data: q},
		{Name: "k", DType: TensorFloat32, Shape: []int{B, T, H, Ds}, Data: k},
		{Name: "v", DType: TensorFloat32, Shape: []int{B, T, H, Dv}, Data: v},
		{Name: "gate", DType: TensorFloat32, Shape: []int{B, T, H, Ds}, Data: gate},
	}, "scan")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuHGRN2Scan(q, k, v, gate, B, T, H, Ds, Dv)
	if diff := maxAbsDiffFloat32(got, want); diff > 1e-5 {
		t.Fatalf("HGRN2Scan L_inf=%g, want <= 1e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func TestMLSTMScanMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B  = 2
		T  = 4
		H  = 2
		Dk = 3
		Dv = 3
	)

	q := patternedFloats(B*T*H*Dk, 0.09)
	k := patternedFloats(B*T*H*Dk, 0.08)
	v := patternedFloats(B*T*H*Dv, 0.12)
	inputGate := patternedFloats(B*T*H, 0.17)
	forgetGate := patternedFloats(B*T*H, 0.05)

	prog := ir.NewProgram(1)
	prog.DeclareInput("q", ir.TensorFloat32, []int{B, T, H, Dk})
	prog.DeclareInput("k", ir.TensorFloat32, []int{B, T, H, Dk})
	prog.DeclareInput("v", ir.TensorFloat32, []int{B, T, H, Dv})
	prog.DeclareInput("ig", ir.TensorFloat32, []int{B, T, H})
	prog.DeclareInput("fg", ir.TensorFloat32, []int{B, T, H})
	prog.DeclareOutput("scan", ir.TensorFloat32, []int{B * T * H, Dv})
	prog.MLSTMScan("q", "k", "v", "ig", "fg", "scan", B, T, H, Dk, Dv)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData(dummy): %v", err)
	}
	defer FreeHandle(dummy)

	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "q", DType: TensorFloat32, Shape: []int{B, T, H, Dk}, Data: q},
		{Name: "k", DType: TensorFloat32, Shape: []int{B, T, H, Dk}, Data: k},
		{Name: "v", DType: TensorFloat32, Shape: []int{B, T, H, Dv}, Data: v},
		{Name: "ig", DType: TensorFloat32, Shape: []int{B, T, H}, Data: inputGate},
		{Name: "fg", DType: TensorFloat32, Shape: []int{B, T, H}, Data: forgetGate},
	}, "scan")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuMLSTMScan(q, k, v, inputGate, forgetGate, B, T, H, Dk, Dv)
	if diff := maxAbsDiffFloat32(got, want); diff > 1e-5 {
		t.Fatalf("MLSTMScan L_inf=%g, want <= 1e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func patternedFloats(n int, scale float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(scale * (math.Sin(float64(i+1))*0.7 + math.Cos(float64((i+3)%11))*0.3))
	}
	return out
}

func cpuHGRN2Scan(q, k, v, gate []float32, B, T, H, Ds, Dv int) []float32 {
	state := make([]float32, B*H*Ds*Dv)
	out := make([]float32, B*T*H*Dv)
	qIdx := func(b, t, h, d int) int { return (((b*T+t)*H+h)*Ds + d) }
	vIdx := func(b, t, h, d int) int { return (((b*T+t)*H+h)*Dv + d) }
	sIdx := func(b, h, ds, dv int) int { return (((b*H+h)*Ds+ds)*Dv + dv) }
	oIdx := func(b, t, h, d int) int { return (((b*T+t)*H+h)*Dv + d) }
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < H; h++ {
				for ds := 0; ds < Ds; ds++ {
					for dv := 0; dv < Dv; dv++ {
						state[sIdx(b, h, ds, dv)] = gate[qIdx(b, t, h, ds)]*state[sIdx(b, h, ds, dv)] + k[qIdx(b, t, h, ds)]*v[vIdx(b, t, h, dv)]
					}
				}
				for dv := 0; dv < Dv; dv++ {
					var y float32
					for ds := 0; ds < Ds; ds++ {
						y += q[qIdx(b, t, h, ds)] * state[sIdx(b, h, ds, dv)]
					}
					out[oIdx(b, t, h, dv)] = y
				}
			}
		}
	}
	return out
}

func cpuMLSTMScan(q, k, v, inputGate, forgetGate []float32, B, T, H, Dk, Dv int) []float32 {
	c := make([]float32, B*H*Dk*Dv)
	n := make([]float32, B*H*Dk)
	m := make([]float32, B*H)
	out := make([]float32, B*T*H*Dv)
	qIdx := func(b, t, h, d int) int { return (((b*T+t)*H+h)*Dk + d) }
	vIdx := func(b, t, h, d int) int { return (((b*T+t)*H+h)*Dv + d) }
	gIdx := func(b, t, h int) int { return (b*T+t)*H + h }
	cIdx := func(b, h, dk, dv int) int { return (((b*H+h)*Dk+dk)*Dv + dv) }
	nIdx := func(b, h, dk int) int { return (b*H+h)*Dk + dk }
	mIdx := func(b, h int) int { return b*H + h }
	oIdx := func(b, t, h, d int) int { return (((b*T+t)*H+h)*Dv + d) }
	qScale := float32(1 / math.Sqrt(float64(Dk)))
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < H; h++ {
				ig := inputGate[gIdx(b, t, h)]
				fg := forgetGate[gIdx(b, t, h)]
				mNext := float32(math.Max(float64(fg+m[mIdx(b, h)]), float64(ig)))
				iGate := float32(math.Exp(float64(ig - mNext)))
				fGate := float32(math.Exp(float64(fg + m[mIdx(b, h)] - mNext)))
				for dk := 0; dk < Dk; dk++ {
					for dv := 0; dv < Dv; dv++ {
						c[cIdx(b, h, dk, dv)] = fGate*c[cIdx(b, h, dk, dv)] + iGate*k[qIdx(b, t, h, dk)]*v[vIdx(b, t, h, dv)]
					}
					n[nIdx(b, h, dk)] = fGate*n[nIdx(b, h, dk)] + iGate*k[qIdx(b, t, h, dk)]
				}
				var denomRaw float32
				for dk := 0; dk < Dk; dk++ {
					denomRaw += q[qIdx(b, t, h, dk)] * qScale * n[nIdx(b, h, dk)]
				}
				denom := float32(math.Max(math.Abs(float64(denomRaw)), 1.0))
				for dv := 0; dv < Dv; dv++ {
					var numerator float32
					for dk := 0; dk < Dk; dk++ {
						numerator += q[qIdx(b, t, h, dk)] * qScale * c[cIdx(b, h, dk, dv)]
					}
					out[oIdx(b, t, h, dv)] = numerator / denom
				}
				m[mIdx(b, h)] = mNext
			}
		}
	}
	return out
}
