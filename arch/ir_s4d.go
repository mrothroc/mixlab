package arch

// S4D emits a diagonal LTI SSM. mode=0 uses FFT convolution and mode=1 uses
// the equivalent recurrent update for reference/parity tests.
func (p *Program) S4D(
	x, logDT, logAReal, aImag, cReal, cImag, direct, output, kernel string,
	B, T, D, stateSize, mode int,
) {
	p.s4d(
		[]string{x, logDT, logAReal, aImag, cReal, cImag, direct},
		output, kernel, B, T, D, stateSize, mode, false,
	)
}

// S4DSobolev emits the FFT S4D path with a learned per-feature Sobolev
// exponent. The direct D*x contribution remains outside the spectral filter.
func (p *Program) S4DSobolev(
	x, logDT, logAReal, aImag, cReal, cImag, direct, beta, output, kernel string,
	B, T, D, stateSize int,
) {
	p.s4d(
		[]string{x, logDT, logAReal, aImag, cReal, cImag, direct, beta},
		output, kernel, B, T, D, stateSize, 0, true,
	)
}

func (p *Program) s4d(
	inputs []string,
	output, kernel string,
	B, T, D, stateSize, mode int,
	sobolev bool,
) {
	params := []int{B, T, D, stateSize, mode}
	if sobolev {
		params = append(params, 1)
	}
	p.AddOp(OpS4D, inputs, []string{output, kernel}, nil, params)
}

// S4DAdvanced emits the reference-compatible S4D path with grouped A/B,
// optional trainable B, bilinear discretization, and bidirectional kernels.
func (p *Program) S4DAdvanced(
	inputs []string,
	output, kernel string,
	B, T, D, stateSize, nSSM int,
	bidirectional bool,
	discretization string,
	trainableB bool,
) {
	p.s4dAdvanced(inputs, output, kernel, B, T, D, stateSize, nSSM, bidirectional, discretization, trainableB, false)
}

// S4DAdvancedSobolev emits the advanced S4D FFT path with the final input
// interpreted as a learned per-feature Sobolev exponent.
func (p *Program) S4DAdvancedSobolev(
	inputs []string,
	output, kernel string,
	B, T, D, stateSize, nSSM int,
	bidirectional bool,
	discretization string,
	trainableB bool,
) {
	p.s4dAdvanced(inputs, output, kernel, B, T, D, stateSize, nSSM, bidirectional, discretization, trainableB, true)
}

func (p *Program) s4dAdvanced(
	inputs []string,
	output, kernel string,
	B, T, D, stateSize, nSSM int,
	bidirectional bool,
	discretization string,
	trainableB bool,
	sobolev bool,
) {
	flags := 0
	if bidirectional {
		flags |= 1
	}
	if discretization == S4DDiscretizationBilinear {
		flags |= 2
	}
	if trainableB {
		flags |= 4
	}
	if sobolev {
		flags |= 8
	}
	p.AddOp(
		OpS4D,
		inputs,
		[]string{output, kernel},
		nil,
		[]int{B, T, D, stateSize, 0, nSSM, flags},
	)
}
