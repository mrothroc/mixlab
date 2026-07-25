//go:build mlx

package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func BenchmarkMamba3SelectiveScanForwardBackward(b *testing.B) {
	benchmarkMamba3SelectiveScan(b, true, false)
}

func BenchmarkMamba3SelectiveScanForwardBackwardColdCompile(b *testing.B) {
	benchmarkMamba3SelectiveScan(b, false, false)
}

func BenchmarkMamba3SelectiveScanForward(b *testing.B) {
	benchmarkMamba3SelectiveScan(b, true, true)
}

func BenchmarkMamba3CanonicalTrainingStep(b *testing.B) {
	if !gpu.Available() {
		b.Skip("MLX backend not available")
	}

	useConv := true
	chunkSize := 64
	training := TrainingSpec{
		Optimizer:   "adamw",
		Steps:       b.N + 1,
		LR:          3e-4,
		BatchTokens: 512,
		Seed:        17,
		GradClip:    1,
	}
	training.ApplyDefaults()
	cfg := &ArchConfig{
		Name:          "mamba3_canonical_training_benchmark",
		ModelDim:      128,
		VocabSize:     1024,
		SeqLen:        512,
		TieEmbeddings: false,
		Blocks: []BlockSpec{
			{
				Type:          "mamba3-canonical",
				InnerDim:      128,
				StateSize:     16,
				NGroups:       4,
				DTRank:        8,
				ConvKernel:    4,
				UseConv:       &useConv,
				ScanChunkSize: &chunkSize,
			},
			{Type: "swiglu"},
			{
				Type:          "mamba3-canonical",
				InnerDim:      128,
				StateSize:     16,
				NGroups:       4,
				DTRank:        8,
				ConvKernel:    4,
				UseConv:       &useConv,
				ScanChunkSize: &chunkSize,
			},
			{Type: "swiglu"},
		},
		Training: training,
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		b.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		b.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()

	xTok := make([]int, cfg.Training.BatchTokens)
	yTok := make([]int, cfg.Training.BatchTokens)
	for i := range xTok {
		xTok[i] = i % cfg.VocabSize
		yTok[i] = (i + 1) % cfg.VocabSize
	}
	runStep := func() {
		if err := trainer.SubmitStepGPU(xTok, yTok, 1, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			b.Fatalf("SubmitStepGPU: %v", err)
		}
		if _, err := trainer.CollectLossGPU(); err != nil {
			b.Fatalf("CollectLossGPU: %v", err)
		}
	}
	runStep()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runStep()
	}
}

func benchmarkMamba3SelectiveScan(b *testing.B, warmup, forwardOnly bool) {
	if !gpu.Available() {
		b.Skip("MLX backend not available")
	}

	const (
		B = 2
		T = 1024
		D = 128
		N = 16
		G = 4
	)

	prog := arch.NewProgram(7)
	prog.DeclareInput("dummy", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	prog.Mamba3SelectiveScan("w0", "w1", "w2", "w3", "w4", "w5", "w6", "y", B, T, D, N, G)
	prog.MeanAxis("y", 1, "loss_rows")
	prog.MeanAxis("loss_rows", 0, "loss")

	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		b.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	rng := rand.New(rand.NewSource(20260505))
	x := seededFloats(rng, B*T*D, 0.2)
	dt := seededFloats(rng, B*T*D, 0.05)
	lambdaInput := seededFloats(rng, B*T*D, 0.05)
	theta := seededFloats(rng, B*T*D*(N/2), 0.05)
	aLog := make([]float32, D*N)
	for d := 0; d < D; d++ {
		for n := 0; n < N; n++ {
			aLog[d*N+n] = float32(math.Log(float64(n+1))) - 2.0
		}
	}
	bProj := seededFloats(rng, B*T*G*N, 0.1)
	cProj := seededFloats(rng, B*T*G*N, 0.1)

	weights := [][]float32{x, dt, lambdaInput, theta, aLog, bProj, cProj}
	shapes := [][2]int{
		{B * T, D},
		{B * T, D},
		{B * T, D},
		{B * T, D * (N / 2)},
		{D, N},
		{B * T, G * N},
		{B * T, G * N},
	}

	handles := make([]int64, len(weights))
	for i := range weights {
		handles[i], err = gpu.FromData(weights[i], shapes[i][0], shapes[i][1])
		if err != nil {
			b.Fatalf("FromData(%d): %v", i, err)
		}
		defer gpu.FreeHandle(handles[i])
	}

	inputs := []gpu.TensorInput{{Name: "dummy", DType: gpu.TensorFloat32, Shape: []int{1}, Data: []float32{0}}}
	if warmup {
		if forwardOnly {
			if _, err := gpu.EvalProgramOutput(gpuProg, handles, inputs, "loss"); err != nil {
				b.Fatalf("warmup EvalProgramOutput: %v", err)
			}
		} else if _, _, err := gpu.EvalProgramGradientsForOutput(gpuProg, handles, inputs, "loss"); err != nil {
			b.Fatalf("warmup EvalProgramGradientsForOutput: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if forwardOnly {
			if _, err := gpu.EvalProgramOutput(gpuProg, handles, inputs, "loss"); err != nil {
				b.Fatalf("EvalProgramOutput iteration %d: %v", i, err)
			}
		} else if _, _, err := gpu.EvalProgramGradientsForOutput(gpuProg, handles, inputs, "loss"); err != nil {
			b.Fatalf("EvalProgramGradientsForOutput iteration %d: %v", i, err)
		}
	}
}
