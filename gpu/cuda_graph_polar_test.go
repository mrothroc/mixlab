package gpu

import (
	"os"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestPolarCUDAGraphLimitsCompose(t *testing.T) {
	for _, polar := range []int{arch.OpCartesianToPolar, arch.OpPolarToCartesian} {
		for _, other := range []int{arch.OpAdd, arch.OpGatedDeltaScan, arch.OpMamba3CanonicalBlock, arch.OpTTTMLPScan, arch.OpTTTMLPStatefulScan} {
			for _, reverse := range []bool{false, true} {
				p := &arch.Program{}
				for i := 0; i < 100; i++ {
					p.Ops = append(p.Ops, arch.Op{Code: arch.OpAdd})
				}
				a, b := polar, other
				if reverse {
					a, b = b, a
				}
				p.Ops = append(p.Ops, arch.Op{Code: a}, arch.Op{Code: b})
				if got := tuneCUDAGraphLimitsForOS(p, "linux"); got != (CUDAGraphLimits{20, 100, 64}) {
					t.Fatalf("polar=%d other=%d reverse=%v: %+v", polar, other, reverse, got)
				}
			}
		}
	}
	if got := TuneCUDAGraphLimits(nil); got != (CUDAGraphLimits{}) {
		t.Fatal(got)
	}
	p := &arch.Program{Ops: []arch.Op{{Code: arch.OpPolarToCartesian}}}
	if got := tuneCUDAGraphLimitsForOS(p, "linux"); got != (CUDAGraphLimits{3, 100, 64}) {
		t.Fatal(got)
	}
	if got := tuneCUDAGraphLimitsForOS(p, "darwin"); got != (CUDAGraphLimits{3, 0, 0}) {
		t.Fatalf("Metal policy changed: %+v", got)
	}
}

func TestPolarCUDAGraphLimitsKeepExplicitOverrides(t *testing.T) {
	for _, key := range []string{MLXMaxOpsPerBufferEnv, MLXMaxMBPerBufferEnv, MLXCUDAGraphCacheEnv} {
		t.Setenv(key, "123")
	}
	p := &arch.Program{Ops: []arch.Op{{Code: arch.OpPolarToCartesian}}}
	ApplyCUDAGraphLimits(tuneCUDAGraphLimitsForOS(p, "linux"))
	for _, key := range []string{MLXMaxOpsPerBufferEnv, MLXMaxMBPerBufferEnv, MLXCUDAGraphCacheEnv} {
		if os.Getenv(key) != "123" {
			t.Fatalf("overrode %s", key)
		}
	}
}
