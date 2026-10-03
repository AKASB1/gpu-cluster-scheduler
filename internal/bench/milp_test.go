package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

func needPython(t *testing.T) {
	t.Helper()
	root, err := extpolicy.RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(extpolicy.Python(), "-c", "import harness.milp.solve")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		msg := fmt.Sprintf("Python harness with SciPy not available (%v %s); activate the virtual environment", err, out)
		if os.Getenv("GCS_REQUIRE_PYTHON") == "1" {
			t.Fatal(msg)
		}
		t.Skip(msg)
	}
}

func sumWFlow(rt *metrics.RunTrace) float64 {
	s := 0.0
	for i := range rt.Jobs {
		if !rt.Jobs[i].Infeasible {
			s += metrics.TimesOf(rt, &rt.Jobs[i]).WFlow
		}
	}
	return s
}

// Check 7 (Go side): on small whole-slot instances (S8 configuration, test
// seeds 901..906), the node-model MILP schedule replayed in the simulator as
// a fixed plan reproduces the MILP objective (relative 1e-9), and no Go
// baseline beats the optimum.
func TestMILPPlanReplayAndNoHeuristicBeatsOptimum(t *testing.T) {
	needPython(t)
	dir := t.TempDir()
	clPath := "../../configs/clusters/s8-small.json"
	cl, err := cluster.Load(clPath)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../../configs/workloads/s8-small.json")
	if err != nil {
		t.Fatal(err)
	}
	var gc trace.GenConfig
	if err := json.Unmarshal(data, &gc); err != nil {
		t.Fatal(err)
	}
	gc.WorkCapacity = cl.WorkCapacity()
	var tasks []MILPTask
	traces := map[string][]api.Job{}
	for seed := uint64(901); seed <= 906; seed++ {
		jobs, err := trace.Generate(gc, seed)
		if err != nil {
			t.Fatal(err)
		}
		id := fmt.Sprintf("seed%d", seed)
		p := filepath.Join(dir, id+".csv")
		if _, err := trace.SaveWithManifest(p, jobs, gc.Info(), seed); err != nil {
			t.Fatal(err)
		}
		abs, _ := filepath.Abs(clPath)
		tasks = append(tasks, MILPTask{ID: id, Trace: p, Cluster: abs, Model: "node", SlotS: 300, TimeLimitS: 120})
		traces[id] = jobs
	}
	res, err := SolveOffline(context.Background(), tasks, dir, filepath.Join(dir, "solve.log"))
	if err != nil {
		t.Fatal(err)
	}
	baselines := []string{"fifo+first_fit+none", "fifo+best_fit+easy", "shortest_estimate+best_fit+easy", "priority+first_fit+conservative",
		"drf+least_fragmentation+easy", "fifo+topology_aware+easy_predicted", "shortest_estimate+first_fit+none"}
	for _, r := range res {
		if r.Status != "optimal" || !r.WholeSlot || r.Objective == nil {
			t.Fatalf("%s: status %s whole_slot %v", r.ID, r.Status, r.WholeSlot)
		}
		jobs := traces[r.ID]
		plan, err := policy.NewPlan(r.Starts)
		if err != nil {
			t.Fatal(err)
		}
		rt, err := simulator.Run(simulator.Config{Cluster: cl, Jobs: jobs, Policy: plan})
		if err != nil {
			t.Fatalf("%s: replay: %v", r.ID, err)
		}
		got := sumWFlow(rt)
		if math.Abs(got-*r.Objective) > 1e-9*math.Max(1, *r.Objective) {
			t.Fatalf("%s: replayed objective %.9f, MILP %.9f", r.ID, got, *r.Objective)
		}
		for _, b := range baselines {
			p, err := policy.New(policy.Spec{Name: b}, policy.Env{})
			if err != nil {
				t.Fatal(err)
			}
			brt, err := simulator.Run(simulator.Config{Cluster: cl, Jobs: jobs, Policy: p})
			if err != nil {
				t.Fatal(err)
			}
			if w := sumWFlow(brt); w < *r.Objective-1e-9*math.Max(1, *r.Objective) {
				t.Fatalf("%s: %s reaches %.6f below the optimum %.6f", r.ID, b, w, *r.Objective)
			}
		}
		t.Logf("%s: optimum %.4f reproduced by replay (%d jobs, %d variables, %.2fs)", r.ID, *r.Objective, r.Jobs, r.Variables, r.WallSolveS)
	}
}
