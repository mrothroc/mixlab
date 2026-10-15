package precision

import (
	"encoding/json"
	"testing"
)

func TestPolicyWire(t *testing.T) {
	p := Policy{"float32", "metal", true}
	b, e := json.Marshal(p)
	if e != nil {
		t.Fatal(e)
	}
	if string(b) != `{"compute_dtype":"float32","backend":"metal","tf32_allowed":true}` {
		t.Fatal(string(b))
	}
	h, e := p.Digest()
	if e != nil {
		t.Fatal(e)
	}
	var restored Policy
	if e = json.Unmarshal(b, &restored); e != nil {
		t.Fatal(e)
	}
	h2, e := restored.Digest()
	if e != nil || h != h2 || p != restored {
		t.Fatal("policy changed")
	}
	restored.TF32Allowed = false
	h2, _ = restored.Digest()
	if h == h2 {
		t.Fatal("policy change not reflected")
	}
	for _, p := range []Policy{{"bf16", "metal", false}, {"float32", "cuda", true}, {"float32", "unavailable", false}} {
		if e := p.Validate(); e != nil {
			t.Fatal(e)
		}
	}
	for _, p := range []Policy{{"float16", "metal", false}, {"float32", "unknown", false}, {"float32", "unavailable", true}, {"float32", "cpu", false}} {
		if p.Validate() == nil {
			t.Fatal(p)
		}
	}
}
