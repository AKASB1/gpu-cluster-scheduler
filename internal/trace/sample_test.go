package trace

import (
	"reflect"
	"testing"
)

// The committed sample trace (configs/traces/sample-200.csv, below 100 KB)
// loads through its manifest check, and the generator still produces it from
// the committed base workload with seed 1 and 200 jobs on the 128-GPU
// reference cluster (`go run ./cmd/scheduler -jobs 200 -seed 1 -save-trace`).
func TestCommittedSampleTrace(t *testing.T) {
	jobs, man, err := LoadVerified("../../configs/traces/sample-200.csv")
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 200 || man.Seed != 1 {
		t.Fatalf("sample: %d jobs, seed %d", len(jobs), man.Seed)
	}
	c := baseConfig(t)
	c.Jobs = 200
	if again := gen(t, c, 1); !reflect.DeepEqual(again, jobs) {
		t.Fatal("the generator no longer reproduces the committed sample trace")
	}
}
