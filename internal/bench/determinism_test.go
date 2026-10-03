package bench

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
)

func testEnv(t *testing.T, workers int) *Env {
	t.Helper()
	root, err := extpolicy.RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	return &Env{Root: root, Outputs: t.TempDir(), Workers: workers, JobsOverride: 150}
}

var testScenario = Scenario{ID: "det", Group: "T", Kind: KindSim, Cluster: "configs/clusters/heterogeneous.json",
	Workload: "configs/workloads/base.json", Override: []byte(`{"arrival": {"load": 0.95}, "estimate": {"model": "lognormal", "sigma": 0.5},
	"class_constraints": [{"class": "v100", "frac": 0.2}]}`)}

func runSmall(t *testing.T, env *Env, policies []string, seeds []uint64, out string) []byte {
	t.Helper()
	sc := testScenario
	ts, err := env.Traces(&sc, seeds)
	if err != nil {
		t.Fatal(err)
	}
	var tasks []Task
	for _, p := range policies {
		for _, s := range seeds {
			tasks = append(tasks, Task{Scenario: sc.ID, Policy: p, Seed: s, Trace: ts[s], Ref: Reference{Rs: 2, Rw: 100},
				Weights: Weights{1, 1, 1, 1, 1}})
		}
	}
	rows, err := env.RunTasks(context.Background(), tasks)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteRows(out, rows); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(out, "runs.csv"))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

var detPolicies = []string{"fifo+first_fit+none", "priority+best_fit+easy+preempt", "drf+topology_aware+conservative",
	"shortest_estimate+least_fragmentation+easy_predicted", "fifo+first_fit+easy_oracle"}

// Determinism (acceptance): the same seeds and configuration give
// byte-identical runs.csv (wall_ columns live in wall.csv) even with
// parallel workers in a different order; different seeds differ; adding or
// removing a policy does not change the traces. This is also the runner's
// concurrency test (run it with -count=20).
func TestRunnerDeterminism(t *testing.T) {
	seeds := []uint64{3, 4}
	a := runSmall(t, testEnv(t, 4), detPolicies, seeds, t.TempDir())
	b := runSmall(t, testEnv(t, 1), detPolicies, seeds, t.TempDir())
	if !bytes.Equal(a, b) {
		t.Fatal("runs.csv differs between two identical runs (4 workers vs 1)")
	}
	c := runSmall(t, testEnv(t, 4), detPolicies, []uint64{5, 6}, t.TempDir())
	if bytes.Equal(a, c) {
		t.Fatal("different seeds gave identical results")
	}
	// traces do not depend on the policy list
	e1, e2 := testEnv(t, 2), testEnv(t, 2)
	runSmall(t, e1, detPolicies[:1], seeds, t.TempDir())
	runSmall(t, e2, detPolicies, seeds, t.TempDir())
	for _, s := range []string{"seed-3.csv", "seed-4.csv"} {
		x, _ := os.ReadFile(filepath.Join(e1.Outputs, "traces", "det", s))
		y, _ := os.ReadFile(filepath.Join(e2.Outputs, "traces", "det", s))
		if len(x) == 0 || !bytes.Equal(x, y) {
			t.Fatalf("trace %s depends on the policy list", s)
		}
	}
}
