package workerjob

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/distributed"
)

func fixture(t *testing.T) Assignment {
	t.Helper()
	m, err := distributed.NewDDPGroupMembership("run", "group", 1, "ring", []distributed.DDPGroupMember{{MemberID: "a", Rank: 0}, {MemberID: "b", Rank: 1}})
	if err != nil {
		t.Fatal(err)
	}
	v, err := distributed.NewLocalGroupView(m, "a", 0, "attempt")
	if err != nil {
		t.Fatal(err)
	}
	return Assignment{Version: Version, JobID: "job", AttemptID: "attempt", BuildID: strings.Repeat("a", 64),
		View: v, Config: json.RawMessage(`{"model_dim":16}`), DatasetSelector: "toy", TrainPattern: "/data/train*.bin",
		DatasetSHA256: strings.Repeat("b", 64), ProgramSHA256: strings.Repeat("c", 64), WeightLayoutSHA256: strings.Repeat("d", 64), OptimizerSHA256: strings.Repeat("e", 64), RuntimeSeconds: 60,
		RingAddresses: [][]string{{"127.0.0.1:31000"}, {"127.0.0.1:31001"}}}
}

func TestAssignmentBinding(t *testing.T) {
	a := fixture(t)
	b, err := a.Binding()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(raw, b); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{
		bytes.Replace(raw, []byte(`"job_id":"job"`), []byte(`"job_id":"other"`), 1),
		bytes.Replace(raw, []byte(`"job_id":"job"`), []byte(`"job_id":"job","job_id":"job"`), 1),
		bytes.Replace(raw, []byte(`"job_id"`), []byte(`"Job_ID"`), 1),
		bytes.Replace(raw, []byte(`"runtime_seconds":60`), []byte(`"runtime_seconds":null`), 1),
		append([]byte(`{"argv":[],`), raw[1:]...),
	} {
		if _, err := Decode(bad, b); err == nil {
			t.Fatalf("accepted %s", bad)
		}
	}
	changed := b
	changed.BuildID = strings.Repeat("f", 64)
	if _, err := Decode(raw, changed); err == nil {
		t.Fatal("accepted changed build")
	}
}

func TestAssignmentRejectsInvalidLaunch(t *testing.T) {
	for name, mutate := range map[string]func(*Assignment){
		"remote":    func(a *Assignment) { a.RingAddresses[1][0] = "10.0.0.2:31001" },
		"wildcard":  func(a *Assignment) { a.RingAddresses[1][0] = "0.0.0.0:31001" },
		"duplicate": func(a *Assignment) { a.RingAddresses[1][0] = a.RingAddresses[0][0] },
		"rank":      func(a *Assignment) { a.View.LocalRank = 2 },
		"member":    func(a *Assignment) { a.View.LocalMemberID = "b" },
		"attempt":   func(a *Assignment) { a.AttemptID = "other" },
		"digest":    func(a *Assignment) { a.DatasetSHA256 = "bad" },
		"unbounded": func(a *Assignment) { a.RuntimeSeconds = 0 },
		"relative":  func(a *Assignment) { a.TrainPattern = "train*.bin" },
		"version":   func(a *Assignment) { a.Version = "next" },
	} {
		t.Run(name, func(t *testing.T) {
			a := fixture(t)
			mutate(&a)
			if err := a.Validate(); err == nil {
				t.Fatal("accepted invalid assignment")
			}
		})
	}
}
