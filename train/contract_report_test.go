package train

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestInspectContractCPUReports(t *testing.T) {
	for _, tc := range []struct{ path, profile, require string }{
		{"../arch/testdata/execution_contracts/ttt.json", "native-ttt-stateful", "streaming"},
		{"../arch/testdata/execution_contracts/recurrent.json", "native-full", "complete"},
		{"../examples/grid_two_stage_2.json", "native-full", "complete"},
		{"../examples/plain_3L.json", "native-full", ""},
	} {
		t.Run(tc.path, func(t *testing.T) {
			var out bytes.Buffer
			if err := RunInspectContract(tc.path, tc.profile, tc.require, &out); err != nil {
				t.Fatal(err)
			}
			var c arch.ModelContract
			if err := json.Unmarshal(out.Bytes(), &c); err != nil {
				t.Fatal(err)
			}
			if err := arch.ValidateModelContract(&c); err != nil {
				t.Fatal(err)
			}
			var again bytes.Buffer
			if err := RunInspectContract(tc.path, tc.profile, tc.require, &again); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(out.Bytes(), again.Bytes()) {
				t.Fatal("nondeterministic report")
			}
		})
	}
}

func TestInspectContractErrors(t *testing.T) {
	for _, tc := range []struct{ path, profile, require string }{
		{"", "native-full", ""},
		{"missing-config.json", "native-full", ""},
		{"../arch/testdata/execution_contracts/recurrent.json", "native-ttt-stateful", ""},
		{"../arch/testdata/execution_contracts/ttt.json", "native-full", "streaming"},
		{"../arch/testdata/execution_contracts/ttt.json", "native-full", "nonsense"},
		{"../arch/testdata/execution_contracts/ttt.json", "wrong-profile", ""},
	} {
		var out, diagnostic bytes.Buffer
		err := RunInspectContract(tc.path, tc.profile, tc.require, &out)
		if err == nil {
			t.Fatal("accepted", tc)
		}
		if out.Len() != 0 {
			t.Fatal("partial stdout on error")
		}
		if err := WriteContractDiagnostic(&diagnostic, err); err != nil {
			t.Fatal(err)
		}
		var d arch.ContractError
		if err := json.Unmarshal(diagnostic.Bytes(), &d); err != nil || d.Code == "" {
			t.Fatal("not a structured diagnostic", err)
		}
	}
}

func TestTTTContractInitialPackingUsesLayout(t *testing.T) {
	cfg, err := arch.LoadArchConfigQuiet("../arch/testdata/execution_contracts/ttt.json")
	if err != nil {
		t.Fatal(err)
	}
	_, layouts, err := arch.BuildTTTMLPStatefulInferenceIRProgram(cfg, 1, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	l := layouts[0]
	weights := make([][]float32, l.InitialWeightIndices[3]+1)
	for j, part := range l.InitialStateSlices() {
		weights[l.InitialWeightIndices[j]] = make([]float32, part.Length)
		for i := range weights[l.InitialWeightIndices[j]] {
			weights[l.InitialWeightIndices[j]][i] = float32(j + 1)
		}
	}
	packed, err := buildTTTMLPInitialStateData(layouts, weights)
	if err != nil {
		t.Fatal(err)
	}
	for j, part := range l.InitialStateSlices() {
		for _, v := range packed[0][part.DestinationStart : part.DestinationStart+part.Length] {
			if v != float32(j+1) {
				t.Fatal("packing mismatch")
			}
		}
	}
	weights[l.InitialWeightIndices[0]] = weights[l.InitialWeightIndices[0]][1:]
	if _, err := buildTTTMLPInitialStateData(layouts, weights); err == nil || !strings.Contains(err.Error(), "shape mismatch") {
		t.Fatal(err)
	}
}
