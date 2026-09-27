package strictjson

import "testing"

func TestValidate(t *testing.T) {
	for _, s := range []string{`{"a":1,"a":2}`, `{"a":[{"b":1,"b":2}]}`, `{} {}`, `[1,]`, "\xff", `[[[0]]]`} {
		if Validate([]byte(s), 2) == nil {
			t.Fatalf("accepted %q", s)
		}
	}
	for _, s := range []string{`{"a":[{"b":1}]}`, `[]`, `null`, `9007199254740993`} {
		if e := Validate([]byte(s), 8); e != nil {
			t.Fatal(s, e)
		}
	}
}
