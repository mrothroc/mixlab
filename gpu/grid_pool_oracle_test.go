//go:build mlx && cgo && (darwin || linux)

package gpu

import "fmt"

// Independent scalar oracle for the loss used by grid_reference_test:
// mean((pool(x) * cotangent)^2). Strict > preserves the first tied maximum.
func gridPoolCPUFixtures() []gridReferenceCase {
	var cases []gridReferenceCase
	const batch, height, width, channels = 2, 7, 9, 3
	for _, k := range []int{1, 2, 3, 4} {
		for _, tied := range []bool{false, true} {
			x := make([]float32, batch*height*width*channels)
			for i := range x {
				if tied {
					x[i] = float32((i/channels)%3 - 2) // repeated negative/zero maxima
				} else {
					x[i] = float32((i*37)%97-48) / 11
				}
			}
			// Distinct channel constants verify nonzero gradients on exact ties.
			if tied {
				for i := 0; i < len(x); i += channels {
					x[i+1], x[i+2] = -2, 3
				}
			}
			h, w := height/k, width/k
			out := make([]float32, batch*h*w*channels)
			cot := make([]float32, len(out))
			grad := make([]float32, len(x))
			for b := 0; b < batch; b++ {
				for y := 0; y < h; y++ {
					for col := 0; col < w; col++ {
						for c := 0; c < channels; c++ {
							best := ((b*height+y*k)*width+col*k)*channels + c
							for ky := 0; ky < k; ky++ {
								for kx := 0; kx < k; kx++ {
									idx := ((b*height+y*k+ky)*width+col*k+kx)*channels + c
									if x[idx] > x[best] {
										best = idx
									}
								}
							}
							i := ((b*h+y)*w+col)*channels + c
							out[i], cot[i] = x[best], float32(i%7-3)/2
							grad[best] = 2 * out[i] * cot[i] * cot[i] / float32(len(out))
						}
					}
				}
			}
			inShape, outShape := []int{batch, height, width, channels}, []int{batch, h, w, channels}
			cases = append(cases, gridReferenceCase{
				Name: fmt.Sprintf("pool_cpu_k%d_ties_%t", k, tied), Op: "max_pool2d", Kernel: k, Stride: k,
				Weights:   []gridReferenceTensor{{Shape: inShape, Data: x}},
				Grads:     []gridReferenceTensor{{Shape: inShape, Data: grad}},
				Output:    gridReferenceTensor{Shape: outShape, Data: out},
				Cotangent: gridReferenceTensor{Shape: outShape, Data: cot},
			})
		}
	}
	return cases
}
