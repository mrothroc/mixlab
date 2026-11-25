package gridpoints

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestColumnsJSON(t *testing.T) {
	for _, raw := range []string{`null`, `[]`, `[null]`, `[[0]]`, `[[0,1,2]]`, `[[null,1]]`, `[[0,1.5]]`, `[[0,1.0]]`, `[[0,"1"]]`, `[[true,1]]`, `{}`} {
		var c Columns
		if err := json.Unmarshal([]byte(raw), &c); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	var c Columns
	if err := json.Unmarshal([]byte(`[[2,0],[4,1]]`), &c); err != nil || c.Validate(5) != nil {
		t.Fatal(c, err)
	}
	raw, _ := json.Marshal(c)
	var again Columns
	if err := json.Unmarshal(raw, &again); err != nil || !reflect.DeepEqual(c, again) || !c.Contains(2, 0) || c.Contains(0, 2) {
		t.Fatal("roundtrip/membership", c, again, err)
	}
}

func TestColumnsValidation(t *testing.T) {
	for _, c := range []Columns{{}, {{0, 0}}, {{0, 1}, {2, 0}}, {{-1, 0}}, {{0, 5}}} {
		if err := c.Validate(5); err == nil {
			t.Fatal("accepted", c)
		}
	}
	if (Columns{{0, 1}}).Validate(0) == nil || Columns(nil).Validate(0) != nil {
		t.Fatal("omission/width validation")
	}
}
