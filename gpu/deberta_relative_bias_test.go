//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestDebertaRelativeBiasMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B      = 2
		T      = 7
		H      = 2
		D      = 3
		window = 4
	)
	R := 2*window - 1
	q := patternedFloats(B*H*T*D, 0.11)
	k := patternedFloats(B*H*T*D, 0.07)
	posKey := patternedFloats(H*R*D, 0.13)
	posQuery := patternedFloats(H*R*D, 0.17)

	prog := ir.NewProgram(1)
	prog.DeclareInput("q", ir.TensorFloat32, []int{B, H, T, D})
	prog.DeclareInput("k", ir.TensorFloat32, []int{B, H, T, D})
	prog.DeclareInput("pk", ir.TensorFloat32, []int{H, R, D})
	prog.DeclareInput("pq", ir.TensorFloat32, []int{H, R, D})
	prog.DeclareOutput("bias", ir.TensorFloat32, []int{B, H, T, T})
	prog.DebertaRelativeBias("q", "k", "pk", "pq", "bias", B, T, H, D, window)

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
		{Name: "q", DType: TensorFloat32, Shape: []int{B, H, T, D}, Data: q},
		{Name: "k", DType: TensorFloat32, Shape: []int{B, H, T, D}, Data: k},
		{Name: "pk", DType: TensorFloat32, Shape: []int{H, R, D}, Data: posKey},
		{Name: "pq", DType: TensorFloat32, Shape: []int{H, R, D}, Data: posQuery},
	}, "bias")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuDebertaRelativeBias(q, k, posKey, posQuery, B, T, H, D, window)
	if diff := maxAbsDiffFloat32(got, want); diff > 1e-5 {
		t.Fatalf("DebertaRelativeBias L_inf=%g, want <= 1e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func TestDebertaRelativeBiasRepeatedEvalMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B      = 2
		T      = 32
		H      = 3
		D      = 4
		window = 16
	)
	R := 2*window - 1
	q := patternedFloats(B*H*T*D, 0.031)
	k := patternedFloats(B*H*T*D, 0.043)
	posKey := patternedFloats(H*R*D, 0.059)
	posQuery := patternedFloats(H*R*D, 0.071)

	prog := ir.NewProgram(1)
	prog.DeclareInput("q", ir.TensorFloat32, []int{B, H, T, D})
	prog.DeclareInput("k", ir.TensorFloat32, []int{B, H, T, D})
	prog.DeclareInput("pk", ir.TensorFloat32, []int{H, R, D})
	prog.DeclareInput("pq", ir.TensorFloat32, []int{H, R, D})
	prog.DeclareOutput("bias", ir.TensorFloat32, []int{B, H, T, T})
	prog.DebertaRelativeBias("q", "k", "pk", "pq", "bias", B, T, H, D, window)

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

	want := cpuDebertaRelativeBias(q, k, posKey, posQuery, B, T, H, D, window)
	inputs := []TensorInput{
		{Name: "q", DType: TensorFloat32, Shape: []int{B, H, T, D}, Data: q},
		{Name: "k", DType: TensorFloat32, Shape: []int{B, H, T, D}, Data: k},
		{Name: "pk", DType: TensorFloat32, Shape: []int{H, R, D}, Data: posKey},
		{Name: "pq", DType: TensorFloat32, Shape: []int{H, R, D}, Data: posQuery},
	}
	for iter := 0; iter < 20; iter++ {
		got, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "bias")
		if err != nil {
			t.Fatalf("EvalProgramOutput iter %d: %v", iter, err)
		}
		if diff := maxAbsDiffFloat32(got, want); diff > 1e-5 {
			t.Fatalf("iter %d DebertaRelativeBias L_inf=%g, want <= 1e-5", iter, diff)
		}
	}
}

func TestDebertaRelativeBucketIndexMatchesGPTBertReference(t *testing.T) {
	const (
		T      = 8
		window = 4
	)
	got := make([]int, 0, T*T)
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			got = append(got, cpuGPTBertRelativeBucketIndex(i-j, window, T))
		}
	}
	want := []int{
		3, 2, 1, 0, 0, 0, 0, 0,
		4, 3, 2, 1, 0, 0, 0, 0,
		5, 4, 3, 2, 1, 0, 0, 0,
		6, 5, 4, 3, 2, 1, 0, 0,
		6, 6, 5, 4, 3, 2, 1, 0,
		6, 6, 6, 5, 4, 3, 2, 1,
		6, 6, 6, 6, 5, 4, 3, 2,
		6, 6, 6, 6, 6, 5, 4, 3,
	}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("bucket[%d]=%d want %d\nall=%v", i, got[i], want[i], got)
		}
	}
}

func TestDebertaRelativeBiasUsesSameQMinusKBucketForP2C(t *testing.T) {
	const (
		B      = 1
		T      = 4
		H      = 1
		D      = 1
		window = 4
	)
	R := 2*window - 1
	q := make([]float32, B*H*T*D)
	k := make([]float32, B*H*T*D)
	posKey := make([]float32, H*R*D)
	posQuery := make([]float32, H*R*D)
	for i := range k {
		k[i] = 1
	}
	for r := 0; r < R; r++ {
		posQuery[r] = float32(r)
	}

	got := cpuDebertaRelativeBias(q, k, posKey, posQuery, B, T, H, D, window)
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			want := float32(cpuGPTBertRelativeBucketIndex(i-j, window, T))
			if got[i*T+j] != want {
				t.Fatalf("bias[%d,%d]=%g want q-k bucket %g; got matrix=%v", i, j, got[i*T+j], want, got)
			}
		}
	}
}

func cpuDebertaRelativeBias(q, k, posKey, posQuery []float32, B, T, H, D, window int) []float32 {
	R := 2*window - 1
	out := make([]float32, B*H*T*T)
	qIdx := func(b, h, t, d int) int { return (((b*H+h)*T+t)*D + d) }
	pIdx := func(h, r, d int) int { return ((h*R+r)*D + d) }
	oIdx := func(b, h, i, j int) int { return (((b*H+h)*T+i)*T + j) }
	clip := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= R {
			return R - 1
		}
		return v
	}
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			for i := 0; i < T; i++ {
				for j := 0; j < T; j++ {
					relIdx := clip(cpuGPTBertRelativeBucketIndex(i-j, window, T))
					var sum float32
					for d := 0; d < D; d++ {
						sum += q[qIdx(b, h, i, d)]*posKey[pIdx(h, relIdx, d)] +
							k[qIdx(b, h, j, d)]*posQuery[pIdx(h, relIdx, d)]
					}
					out[oIdx(b, h, i, j)] = sum
				}
			}
		}
	}
	return out
}

func cpuGPTBertRelativeBucketIndex(rel, bucketSize, maxPosition int) int {
	if bucketSize <= 1 {
		return 0
	}
	mid := bucketSize / 2
	absPos := 0
	if rel < mid && rel > -mid {
		absPos = mid - 1
	} else {
		absPos = absInt(rel)
		if max := maxPosition - 1; absPos > max {
			absPos = max
		}
	}
	bucketPos := rel
	if absPos > mid {
		logPos := bucketSize - 1
		if mid > 0 && maxPosition-1 > mid {
			denom := math.Log(float64(maxPosition-1) / float64(mid))
			if denom > 0 && !math.IsInf(denom, 0) && !math.IsNaN(denom) {
				scaled := math.Log(float64(absPos)/float64(mid)) / denom * float64(mid-1)
				logPos = int(math.Ceil(scaled)) + mid
			}
		}
		if rel < 0 {
			bucketPos = -logPos
		} else {
			bucketPos = logPos
		}
	}
	maxBucket := bucketSize - 1
	if bucketPos < -maxBucket {
		bucketPos = -maxBucket
	}
	if bucketPos > maxBucket {
		bucketPos = maxBucket
	}
	return bucketPos + maxBucket
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
