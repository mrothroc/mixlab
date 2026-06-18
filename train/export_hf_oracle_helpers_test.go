package train

import "math"

func rmsNormCPU(x [][][]float64, scale []float64) [][][]float64 {
	out := make3D(len(x), len(x[0]), len(x[0][0]))
	for b := range x {
		for t := range x[b] {
			sum := 0.0
			for _, v := range x[b][t] {
				sum += v * v
			}
			inv := 1.0 / math.Sqrt(sum/float64(len(x[b][t]))+1e-5)
			for d, v := range x[b][t] {
				out[b][t][d] = v * inv * scale[d]
			}
		}
	}
	return out
}

func rmsNormHeadsCPU(x [][][][]float64, scale []float64) {
	for b := range x {
		for h := range x[b] {
			for t := range x[b][h] {
				sum := 0.0
				for _, v := range x[b][h][t] {
					sum += v * v
				}
				inv := 1.0 / math.Sqrt(sum/float64(len(x[b][h][t]))+1e-5)
				for d, v := range x[b][h][t] {
					x[b][h][t][d] = v * inv * scale[d]
				}
			}
		}
	}
}

func matmul3DCPU(x [][][]float64, w []float64, inDim, outDim int) [][][]float64 {
	out := make3D(len(x), len(x[0]), outDim)
	for b := range x {
		for t := range x[b] {
			for o := 0; o < outDim; o++ {
				sum := 0.0
				for i := 0; i < inDim; i++ {
					sum += x[b][t][i] * w[i*outDim+o]
				}
				out[b][t][o] = sum
			}
		}
	}
	return out
}

func matmul3DBiasCPU(x [][][]float64, w, bias []float64, inDim, outDim int) [][][]float64 {
	out := matmul3DCPU(x, w, inDim, outDim)
	if bias == nil {
		return out
	}
	for b := range out {
		for t := range out[b] {
			for d := 0; d < outDim; d++ {
				out[b][t][d] += bias[d]
			}
		}
	}
	return out
}

func matmul2DCPU(x []float64, w []float64, rows, inDim, outDim int) []float64 {
	out := make([]float64, rows*outDim)
	for r := 0; r < rows; r++ {
		for o := 0; o < outDim; o++ {
			sum := 0.0
			for i := 0; i < inDim; i++ {
				sum += x[r*inDim+i] * w[i*outDim+o]
			}
			out[r*outDim+o] = sum
		}
	}
	return out
}

func matmul2DBiasCPU(x []float64, w, bias []float64, rows, inDim, outDim int) []float64 {
	out := matmul2DCPU(x, w, rows, inDim, outDim)
	if bias == nil {
		return out
	}
	for r := 0; r < rows; r++ {
		for d := 0; d < outDim; d++ {
			out[r*outDim+d] += bias[d]
		}
	}
	return out
}

func matmulRowFloat64(x []float64, w []float64, inDim, outDim int) []float64 {
	out := make([]float64, outDim)
	for o := 0; o < outDim; o++ {
		sum := 0.0
		for i := 0; i < inDim; i++ {
			sum += x[i] * w[i*outDim+o]
		}
		out[o] = sum
	}
	return out
}

func softmaxRowsFloat64(logits []float64, rows, cols int) []float64 {
	out := make([]float64, len(logits))
	for r := 0; r < rows; r++ {
		row := logits[r*cols : (r+1)*cols]
		maxV := row[0]
		for _, v := range row[1:] {
			if v > maxV {
				maxV = v
			}
		}
		sum := 0.0
		for c, v := range row {
			out[r*cols+c] = math.Exp(v - maxV)
			sum += out[r*cols+c]
		}
		for c := 0; c < cols; c++ {
			out[r*cols+c] /= sum
		}
	}
	return out
}

func flatten3DCPU(x [][][]float64) []float64 {
	out := make([]float64, 0, len(x)*len(x[0])*len(x[0][0]))
	for b := range x {
		for t := range x[b] {
			out = append(out, x[b][t]...)
		}
	}
	return out
}

func slice3DLastDimCPU(x [][][]float64, start, end int) [][][]float64 {
	out := make3D(len(x), len(x[0]), end-start)
	for b := range x {
		for t := range x[b] {
			copy(out[b][t], x[b][t][start:end])
		}
	}
	return out
}

func unflatten3DCPU(flat []float64, batch, seqLen, dim int) [][][]float64 {
	out := make3D(batch, seqLen, dim)
	for b := 0; b < batch; b++ {
		for t := 0; t < seqLen; t++ {
			start := (b*seqLen + t) * dim
			copy(out[b][t], flat[start:start+dim])
		}
	}
	return out
}

func splitHeadsCPU(x [][][]float64, heads, headDim int) [][][][]float64 {
	out := make4D(len(x), heads, len(x[0]), headDim)
	for b := range x {
		for t := range x[b] {
			for h := 0; h < heads; h++ {
				copy(out[b][h][t], x[b][t][h*headDim:(h+1)*headDim])
			}
		}
	}
	return out
}

func repeatKVHeadsCPU(x [][][][]float64, heads, groupSize int) [][][][]float64 {
	if groupSize <= 1 {
		return x
	}
	out := make4D(len(x), heads, len(x[0][0]), len(x[0][0][0]))
	for b := range x {
		for h := 0; h < heads; h++ {
			src := h / groupSize
			for t := range x[b][src] {
				copy(out[b][h][t], x[b][src][t])
			}
		}
	}
	return out
}

func mergeHeadsCPU(x [][][][]float64, dim int) [][][]float64 {
	batch := len(x)
	heads := len(x[0])
	seqLen := len(x[0][0])
	headDim := len(x[0][0][0])
	out := make3D(batch, seqLen, dim)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for t := 0; t < seqLen; t++ {
				copy(out[b][t][h*headDim:(h+1)*headDim], x[b][h][t])
			}
		}
	}
	return out
}

func scale3D(a [][][]float64, scale float64) [][][]float64 {
	out := make3D(len(a), len(a[0]), len(a[0][0]))
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				out[i][j][k] = a[i][j][k] * scale
			}
		}
	}
	return out
}

func add3D(a, b [][][]float64) [][][]float64 {
	out := make3D(len(a), len(a[0]), len(a[0][0]))
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				out[i][j][k] = a[i][j][k] + b[i][j][k]
			}
		}
	}
	return out
}

func mul3DInPlace(a, b [][][]float64) {
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				a[i][j][k] *= b[i][j][k]
			}
		}
	}
}

func gelu3DInPlace(a [][][]float64) {
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				a[i][j][k] = gelu(a[i][j][k])
			}
		}
	}
}

func cpuCrossEntropy(logits [][][]float64, targets [][]int) float64 {
	total := 0.0
	count := 0
	for b := range logits {
		for t := range logits[b] {
			row := logits[b][t]
			maxV := row[0]
			for _, v := range row[1:] {
				if v > maxV {
					maxV = v
				}
			}
			sum := 0.0
			for _, v := range row {
				sum += math.Exp(v - maxV)
			}
			total += -(row[targets[b][t]] - maxV - math.Log(sum))
			count++
		}
	}
	return total / float64(count)
}

func halfRotationRoPEForTest(x []float64, seqLen, headDim, ropeDims int) []float64 {
	out := append([]float64(nil), x...)
	if ropeDims <= 0 || ropeDims >= headDim {
		ropeDims = headDim
	}
	half := ropeDims / 2
	for t := 0; t < seqLen; t++ {
		for d := 0; d < half; d++ {
			i := t*headDim + d
			j := t*headDim + d + half
			freq := math.Exp(float64(d) * (-math.Log(10000.0) * 2.0 / float64(ropeDims)))
			angle := float64(t) * freq
			c := math.Cos(angle)
			s := math.Sin(angle)
			a := x[i]
			b := x[j]
			out[i] = a*c - b*s
			out[j] = b*c + a*s
		}
	}
	return out
}

func hfAdjacentRoPEForTest(x []float64, positions, heads, seqLen, headDim, ropeDims int) []float64 {
	out := append([]float64(nil), x...)
	if ropeDims <= 0 || ropeDims >= headDim {
		ropeDims = headDim
	}
	pairs := ropeDims / 2
	for h := 0; h < heads; h++ {
		for t := 0; t < seqLen; t++ {
			for p := 0; p < pairs; p++ {
				base := ((h*seqLen + t) * headDim) + p*2
				even := x[base]
				odd := x[base+1]
				freq := math.Exp(float64(p) * (-math.Log(10000.0) * 2.0 / float64(ropeDims)))
				angle := float64(positions+t) * freq
				c := math.Cos(angle)
				s := math.Sin(angle)
				out[base] = even*c - odd*s
				out[base+1] = even*s + odd*c
			}
		}
	}
	return out
}

func softmaxCPU(x []float64) []float64 {
	maxV := math.Inf(-1)
	for _, v := range x {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		if math.IsInf(v, -1) {
			out[i] = 0
			continue
		}
		out[i] = math.Exp(v - maxV)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func maxAbsDiff3D(a, b [][][]float64) float64 {
	maxDiff := 0.0
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				if diff := math.Abs(a[i][j][k] - b[i][j][k]); diff > maxDiff {
					maxDiff = diff
				}
			}
		}
	}
	return maxDiff
}

func make3D(a, b, c int) [][][]float64 {
	out := make([][][]float64, a)
	for i := range out {
		out[i] = make([][]float64, b)
		for j := range out[i] {
			out[i][j] = make([]float64, c)
		}
	}
	return out
}

func flattenTokensInt32(tokens [][]int) []int32 {
	out := make([]int32, 0, len(tokens)*len(tokens[0]))
	for b := range tokens {
		for _, tok := range tokens[b] {
			out = append(out, int32(tok))
		}
	}
	return out
}

func make4D(a, b, c, d int) [][][][]float64 {
	out := make([][][][]float64, a)
	for i := range out {
		out[i] = make([][][]float64, b)
		for j := range out[i] {
			out[i][j] = make([][]float64, c)
			for k := range out[i][j] {
				out[i][j][k] = make([]float64, d)
			}
		}
	}
	return out
}

func flattenHead(x [][]float64) []float64 {
	out := make([]float64, 0, len(x)*len(x[0]))
	for i := range x {
		out = append(out, x[i]...)
	}
	return out
}

func unflattenHead(flat []float64, out [][]float64) {
	width := len(out[0])
	for i := range out {
		copy(out[i], flat[i*width:(i+1)*width])
	}
}

func toFloat64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func ffnDimForConfig(cfg *ArchConfig) int {
	ffn := int(math.Round(float64(cfg.ModelDim) * cfg.EffectiveMLPMult()))
	if ffn < cfg.ModelDim {
		return cfg.ModelDim
	}
	return ffn
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func silu(x float64) float64 {
	return x / (1.0 + math.Exp(-x))
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
}
