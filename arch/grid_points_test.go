package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/internal/gridpoints"
)

func TestGridVectorPointsConfig(t *testing.T) {
	base, _ := json.Marshal(polarTestConfig(t))
	for _, value := range []string{`[]`, `null`, `[[0]]`, `[[0,1,2]]`, `[[0,0]]`, `[[0,1],[1,2]]`, `[[0,2]]`, `[[-1,1]]`, `[[0,0.5]]`, `[[null,1]]`, `[[0,1]]`} {
		var cfg map[string]json.RawMessage
		_ = json.Unmarshal(base, &cfg)
		var adapter map[string]json.RawMessage
		_ = json.Unmarshal(cfg["input_adapter"], &adapter)
		adapter["vector_points"] = json.RawMessage(value)
		cfg["input_adapter"], _ = json.Marshal(adapter)
		raw, _ := json.Marshal(cfg)
		got, err := ParseArchConfig(raw, "points")
		if value != `[[0,1]]` {
			if err == nil || !strings.Contains(err.Error(), "vector_points") {
				t.Fatalf("%s: %v", value, err)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		roundtrip, _ := json.Marshal(got)
		again, err := ParseArchConfig(roundtrip, "points")
		if err != nil || !reflect.DeepEqual(got, again) {
			t.Fatal("roundtrip", err)
		}
		got.InputAdapter.VectorDim = 0
		raw, _ = json.Marshal(got)
		if _, err = ParseArchConfig(raw, "no-vectors"); err == nil {
			t.Fatal("missing vector_dim accepted")
		}
	}
	nongrid := &ArchConfig{InputAdapter: &InputAdapterSpec{Kind: InputAdapterTokenEmbedding, VectorPoints: gridpoints.Columns{{0, 1}}}}
	if err := validateInputAdapter(nongrid, "nongrid"); err == nil {
		t.Fatal("nongrid points accepted")
	}
}

func TestGridVectorPointsPolarAdmissionAndGraphParity(t *testing.T) {
	c := polarTestConfig(t)
	before, err := BuildGridIRProgram(c, true)
	if err != nil {
		t.Fatal(err)
	}
	c.InputAdapter.VectorPoints = gridpoints.Columns{{0, 1}}
	c.Training.GridAugmentation = &GridAugmentationSpec{Dihedral: true}
	after, err := BuildGridIRProgram(c, true)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("points changed graph", err)
	}
	for _, index := range []int{0, 6} {
		c := polarTestConfig(t)
		c.InputAdapter.VectorPoints = gridpoints.Columns{{0, 1}}
		c.Training.GridAugmentation = &GridAugmentationSpec{Dihedral: true}
		c.Blocks[0].Ops[index].Params["tx"] = []any{1, 0}
		_, err := BuildGridIRProgram(c, true)
		if err == nil || !strings.Contains(err.Error(), c.Blocks[0].Ops[index].Output) || !strings.Contains(err.Error(), "tx=[1,0]") {
			t.Fatal(err)
		}
	}
	// Ray-set equivariance is a model choice, not an augmentation restriction.
	c.Blocks[0].Ops[0].Params["n_rays"] = 18
	c.Blocks[0].Ops[6].Params["n_rays"] = 18
	if _, err := BuildGridIRProgram(c, true); err != nil {
		t.Fatal(err)
	}
}
