//go:build mlx && cgo && (darwin || linux)

package gpu

import "fmt"

// Scatter-form scalar oracle, independent of MLX's convolution/patch VJP.
// Non-square images and unequal channels catch transposition mistakes.
func gridConvTransposeCPUFixtures() []gridReferenceCase {
	var cases []gridReferenceCase
	const batch, height, width, inChannels, outChannels = 2, 3, 4, 2, 3
	for _, spec := range [][3]int{{1, 1, 0}, {3, 1, 1}, {4, 2, 1}, {6, 2, 2}, {3, 3, 0}, {5, 2, 0}} {
		k, stride, padding := spec[0], spec[1], spec[2]
		h, w := (height-1)*stride-2*padding+k, (width-1)*stride-2*padding+k
		x := make([]float32, batch*height*width*inChannels)
		weight := make([]float32, inChannels*k*k*outChannels)
		bias := []float32{.1, -.2, .3}
		for i := range x {
			x[i] = float32((i*17)%29-14) / 19
		}
		for i := range weight {
			weight[i] = float32((i*7)%23-11) / 31
		}
		out := make([]float32, batch*h*w*outChannels)
		for i := range out {
			out[i] = bias[i%outChannels]
		}
		// Enumerate the input-to-output scatter edges once for forward and VJP.
		edges := func(visit func(int, int, int)) {
			for b := 0; b < batch; b++ {
				for y := 0; y < height; y++ {
					for col := 0; col < width; col++ {
						for ci := 0; ci < inChannels; ci++ {
							xi := ((b*height+y)*width+col)*inChannels + ci
							for ky := 0; ky < k; ky++ {
								for kx := 0; kx < k; kx++ {
									oy, ox := y*stride-padding+ky, col*stride-padding+kx
									if oy < 0 || oy >= h || ox < 0 || ox >= w {
										continue
									}
									for co := 0; co < outChannels; co++ {
										wi := ((ci*k+ky)*k+kx)*outChannels + co
										oi := ((b*h+oy)*w+ox)*outChannels + co
										visit(xi, wi, oi)
									}
								}
							}
						}
					}
				}
			}
		}
		edges(func(xi, wi, oi int) { out[oi] += x[xi] * weight[wi] })
		cot, dy := make([]float32, len(out)), make([]float32, len(out))
		dx, dw, db := make([]float32, len(x)), make([]float32, len(weight)), make([]float32, len(bias))
		for i := range out {
			cot[i] = float32(i%11-5) / 7
			dy[i] = 2 * out[i] * cot[i] * cot[i] / float32(len(out))
			db[i%outChannels] += dy[i]
		}
		edges(func(xi, wi, oi int) {
			dx[xi] += dy[oi] * weight[wi]
			dw[wi] += dy[oi] * x[xi]
		})
		xShape, wShape, outShape := []int{batch, height, width, inChannels}, []int{inChannels, k, k, outChannels}, []int{batch, h, w, outChannels}
		cases = append(cases, gridReferenceCase{
			Name: fmt.Sprintf("transpose_cpu_k%d_s%d_p%d", k, stride, padding), Op: "conv_transpose2d", Kernel: k, Stride: stride, Padding: padding,
			Weights: []gridReferenceTensor{{Shape: xShape, Data: x}, {Shape: wShape, Data: weight}, {Shape: []int{outChannels}, Data: bias}},
			Grads:   []gridReferenceTensor{{Shape: xShape, Data: dx}, {Shape: wShape, Data: dw}, {Shape: []int{outChannels}, Data: db}},
			Output:  gridReferenceTensor{Shape: outShape, Data: out}, Cotangent: gridReferenceTensor{Shape: outShape, Data: cot},
		})
	}
	return cases
}
