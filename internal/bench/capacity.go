package bench

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

// CapacityConfig is the capacity-planning sweep (Tier 2): for each demand
// level and policy, the smallest homogeneous cluster (nodes of 8 GPUs, racks
// of 4 nodes) whose mean P95 wait over the seeds stays at or below the
// target. Demand levels are offered loads relative to the 128-GPU reference
// cluster, so the arrival rate of a level is the same for every cluster size
// (identical traces across sizes and policies).
type CapacityConfig struct {
	Workload   string    `json:"workload"`
	Loads      []float64 `json:"loads"`
	Policies   []string  `json:"policies"`
	TunedFrom  string    `json:"tuned_from"`
	TargetP95S float64   `json:"target_p95_wait_s"`
	Seeds      []uint64  `json:"seeds"`
	MinNodes   int       `json:"min_nodes"`
	MaxNodes   int       `json:"max_nodes"`
}

func capacityCluster(nodes int) (*cluster.Cluster, error) {
	c := cluster.Config{SchemaVersion: 1, Name: fmt.Sprintf("homogeneous-%d", nodes), Classes: []api.GPUClass{{Name: "a100", Speed: 1}},
		CrossNodeFactor: 1.1, CrossRackFactor: 1.25, RestartOverheadS: 120}
	for i := 0; i < nodes; i += 4 {
		c.NodeGroups = append(c.NodeGroups, cluster.NodeGroup{Count: min(4, nodes-i), Prefix: fmt.Sprintf("r%d-n", i/4), Rack: fmt.Sprintf("r%d", i/4),
			Class: "a100", GPUs: 8, CPUs: 128, MemGB: 1024})
	}
	return cluster.Build(c)
}

// runCapacity runs the sweep and writes capacity.csv (every point) and
// capacity_needed.csv (the smallest size meeting the target).
func (env *Env) runCapacity(ctx context.Context, exp *Experiment, tuned *Tuned, opts EvalOptions, dir string) error {
	cc := exp.Capacity
	if cc == nil {
		return nil
	}
	seeds := cc.Seeds
	if opts.Seeds != nil {
		seeds = opts.Seeds
	}
	st := tuned.Scenarios[cc.TunedFrom]
	if st == nil {
		return fmt.Errorf("capacity sweep: no tuning entry %q", cc.TunedFrom)
	}
	pols := Expand(cc.Policies, tuned.Defaults)
	type key struct {
		load  float64
		pol   string
		nodes int
	}
	var tasks []Task
	var keys []key
	for _, load := range cc.Loads {
		sc := &Scenario{ID: fmt.Sprintf("capacity-load%03.0f", load*100), Kind: KindSim, Cluster: "configs/clusters/homogeneous.json",
			Workload: cc.Workload, Override: []byte(fmt.Sprintf(`{"arrival": {"load": %g}}`, load))}
		// traces are generated once, against the work capacity of the 128-GPU
		// reference cluster, and replayed on every cluster size
		ts, err := env.Traces(sc, seeds)
		if err != nil {
			return err
		}
		for nodes := cc.MinNodes; nodes <= cc.MaxNodes; nodes++ {
			cl, err := capacityCluster(nodes)
			if err != nil {
				return err
			}
			for _, pol := range pols {
				for _, seed := range seeds {
					tr := *ts[seed]
					tr.Cluster = cl
					tasks = append(tasks, Task{Scenario: sc.ID, Policy: pol, Params: st.Params[pol], Seed: seed, Trace: &tr,
						Ref: Reference{1, 1}, Tag: fmt.Sprintf("@%dnodes", nodes)})
					keys = append(keys, key{load, pol, nodes})
				}
			}
		}
	}
	env.logf("capacity sweep: %d runs", len(tasks))
	rows, err := env.RunTasks(ctx, tasks)
	if err != nil {
		return err
	}
	vals := map[key][]float64{}
	for i, r := range rows {
		vals[keys[i]] = append(vals[keys[i]], r.Get("wait_p95"))
	}
	all := [][]string{{"load", "policy", "nodes", "gpus", "seeds", "wait_p95_mean_s", "wait_p95_ci95_half_s"}}
	need := [][]string{{"load", "policy", "target_p95_wait_s", "min_nodes_meeting_target", "min_gpus_meeting_target"}}
	for _, load := range cc.Loads {
		for _, pol := range pols {
			found := -1
			for nodes := cc.MinNodes; nodes <= cc.MaxNodes; nodes++ {
				ci := metrics.MeanCI(vals[key{load, pol, nodes}])
				all = append(all, []string{metrics.Format(load), pol, strconv.Itoa(nodes), strconv.Itoa(8 * nodes), strconv.Itoa(ci.N),
					metrics.Format(ci.Mean), metrics.Format(ci.Half)})
				if found < 0 && ci.Mean <= cc.TargetP95S {
					found = nodes
				}
			}
			n, g := "", ""
			if found >= 0 {
				n, g = strconv.Itoa(found), strconv.Itoa(8*found)
			}
			need = append(need, []string{metrics.Format(load), pol, metrics.Format(cc.TargetP95S), n, g})
		}
	}
	if err := writeCSV(filepath.Join(dir, "capacity.csv"), all); err != nil {
		return err
	}
	return writeCSV(filepath.Join(dir, "capacity_needed.csv"), need)
}
