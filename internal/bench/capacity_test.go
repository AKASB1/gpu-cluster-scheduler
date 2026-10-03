package bench

import (
	"context"
	"encoding/csv"
	"os"
	"path/filepath"
	"testing"
)

// A small capacity sweep runs end to end and reports, per load and policy,
// the smallest cluster size meeting the target.
func TestCapacitySweepSmoke(t *testing.T) {
	env := testEnv(t, 4)
	exp := &Experiment{Capacity: &CapacityConfig{Workload: "configs/workloads/base.json", Loads: []float64{0.9}, Policies: []string{"fifo+first_fit+easy"},
		TunedFrom: "x", TargetP95S: 1800, Seeds: []uint64{1, 2}, MinNodes: 8, MaxNodes: 20}}
	tuned := &Tuned{Scenarios: map[string]*ScenarioTuning{"x": {}}}
	dir := t.TempDir()
	if err := env.runCapacity(context.Background(), exp, tuned, EvalOptions{}, dir); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(dir, "capacity_needed.csv"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	recs, _ := csv.NewReader(f).ReadAll()
	if len(recs) != 2 || recs[1][3] == "" {
		t.Fatalf("capacity_needed.csv: %v", recs)
	}
	t.Logf("load 0.9 (vs 128 GPUs), 150 jobs: target met from %s GPUs", recs[1][4])
}
