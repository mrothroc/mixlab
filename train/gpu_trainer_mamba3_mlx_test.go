//go:build mlx

package train

import (
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestIRProgramHasCanonicalMamba3(t *testing.T) {
	tests := []struct {
		name string
		prog *arch.Program
		want bool
	}{
		{name: "nil", prog: nil, want: false},
		{name: "ordinary", prog: programWithOp(arch.OpMatMul), want: false},
		{name: "scan", prog: programWithOp(arch.OpMamba3SelectiveScan), want: true},
		{name: "canonical block", prog: programWithOp(arch.OpMamba3CanonicalBlock), want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := irProgramHasCanonicalMamba3(tt.prog); got != tt.want {
				t.Fatalf("irProgramHasCanonicalMamba3()=%v, want %v", got, tt.want)
			}
		})
	}
}

func programWithOp(code int) *arch.Program {
	prog := arch.NewProgram(0)
	prog.AddOp(code, nil, nil, nil, nil)
	return prog
}
