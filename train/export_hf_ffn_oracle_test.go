package train

import (
	"sort"
	"strings"
	"testing"
)

func moeCPUForward(t *testing.T, cfg *ArchConfig, block BlockSpec, x [][][]float64, w [][]float64) [][][]float64 {
	t.Helper()
	batch := len(x)
	seqLen := len(x[0])
	rows := batch * seqLen
	dim := cfg.ModelDim
	experts := block.NumExperts
	topK := effectiveHFMoETopK(block)
	xNorm := rmsNormCPU(x, w[0])
	flat := flatten3DCPU(xNorm)
	router := matmul2DCPU(flat, w[1], rows, dim, experts)
	probs := softmaxRowsFloat64(router, rows, experts)
	expert := BlockSpec{Type: "swiglu"}
	if block.ExpertBlock != nil {
		expert = *block.ExpertBlock
	}
	expertType := strings.ToLower(strings.TrimSpace(expert.Type))
	if expertType == "" {
		expertType = "swiglu"
	}
	perExpert := 3
	if expertType == "mlp" {
		perExpert = 2
	}
	deltaFlat := make([]float64, rows*dim)
	ffn := ffnDimForConfig(cfg)
	for r := 0; r < rows; r++ {
		order := make([]int, experts)
		for e := 0; e < experts; e++ {
			order[e] = e
		}
		sort.SliceStable(order, func(i, j int) bool {
			return probs[r*experts+order[i]] > probs[r*experts+order[j]]
		})
		denom := 0.0
		for k := 0; k < topK; k++ {
			denom += probs[r*experts+order[k]]
		}
		if denom == 0 {
			denom = 1
		}
		row := flat[r*dim : (r+1)*dim]
		for k := 0; k < topK; k++ {
			e := order[k]
			weight := probs[r*experts+e] / denom
			base := 2 + e*perExpert
			var expertOut []float64
			switch expertType {
			case "swiglu", "geglu":
				gate := matmulRowFloat64(row, w[base], dim, ffn)
				up := matmulRowFloat64(row, w[base+1], dim, ffn)
				for i := range gate {
					if expertType == "swiglu" {
						gate[i] = sigmoid(gate[i])
					} else {
						gate[i] = gelu(gate[i])
					}
					gate[i] *= up[i]
				}
				expertOut = matmulRowFloat64(gate, w[base+2], ffn, dim)
			case "mlp":
				up := matmulRowFloat64(row, w[base], dim, ffn)
				act := strings.ToLower(strings.TrimSpace(expert.Activation))
				if act == "" {
					act = "silu"
				}
				slope := expert.LeakySlope
				if slope == 0 {
					slope = 0.5
				}
				for i := range up {
					switch act {
					case "silu":
						up[i] = silu(up[i])
					case "gelu":
						up[i] = gelu(up[i])
					case "relu":
						if up[i] < 0 {
							up[i] = 0
						}
					case "leaky_relu_sq":
						if up[i] < 0 {
							up[i] *= slope
						}
						up[i] *= up[i]
					default:
						t.Fatalf("unsupported MoE MLP activation %q", act)
					}
				}
				expertOut = matmulRowFloat64(up, w[base+1], ffn, dim)
			default:
				t.Fatalf("unsupported MoE expert type %q", expertType)
			}
			for d := 0; d < dim; d++ {
				deltaFlat[r*dim+d] += weight * expertOut[d]
			}
		}
	}
	return add3D(x, unflatten3DCPU(deltaFlat, batch, seqLen, dim))
}

func gatedGLUCPUForward(t *testing.T, cfg *ArchConfig, x [][][]float64, w [][]float64, gateActivation string) [][][]float64 {
	t.Helper()
	dim := cfg.ModelDim
	ffn := ffnDimForConfig(cfg)
	xNorm := rmsNormCPU(x, w[0])
	gate := matmul3DCPU(xNorm, w[1], dim, ffn)
	up := matmul3DCPU(xNorm, w[2], dim, ffn)
	for b := range gate {
		for i := range gate[b] {
			for d := range gate[b][i] {
				switch gateActivation {
				case "sigmoid":
					gate[b][i][d] = sigmoid(gate[b][i][d]) * up[b][i][d]
				case "gelu":
					gate[b][i][d] = gelu(gate[b][i][d]) * up[b][i][d]
				default:
					t.Fatalf("unsupported gate activation %q", gateActivation)
				}
			}
		}
	}
	down := matmul3DCPU(gate, w[3], ffn, dim)
	return add3D(x, down)
}

func mlpCPUForward(t *testing.T, cfg *ArchConfig, block BlockSpec, x [][][]float64, w [][]float64) [][][]float64 {
	t.Helper()
	dim := cfg.ModelDim
	ffn := ffnDimForConfig(cfg)
	xNorm := rmsNormCPU(x, w[0])
	up := matmul3DCPU(xNorm, w[1], dim, ffn)
	act := strings.ToLower(strings.TrimSpace(block.Activation))
	if act == "" {
		act = "silu"
	}
	slope := block.LeakySlope
	if slope == 0 {
		slope = 0.5
	}
	for b := range up {
		for i := range up[b] {
			for d := range up[b][i] {
				switch act {
				case "silu":
					up[b][i][d] = silu(up[b][i][d])
				case "gelu":
					up[b][i][d] = gelu(up[b][i][d])
				case "relu":
					if up[b][i][d] < 0 {
						up[b][i][d] = 0
					}
				case "leaky_relu_sq":
					v := up[b][i][d]
					if v < 0 {
						v *= slope
					}
					up[b][i][d] = v * v
				default:
					t.Fatalf("unsupported mlp activation %q", act)
				}
			}
		}
	}
	down := matmul3DCPU(up, w[2], ffn, dim)
	return add3D(x, down)
}
