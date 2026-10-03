package policy_test

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

const smallCluster = `{"schema_version":1,"name":"small","classes":[{"name":"a100","speed":1},{"name":"v100","speed":0.4}],
"node_groups":[{"count":3,"prefix":"r0-n","rack":"r0","class":"a100","gpus":8,"cpus":128,"mem_gb":1024},
{"count":3,"prefix":"r1-n","rack":"r1","class":"a100","gpus":8,"cpus":128,"mem_gb":1024},
{"count":2,"prefix":"r2-n","rack":"r2","class":"v100","gpus":8,"cpus":128,"mem_gb":1024}],
"cross_node_factor":1.1,"cross_rack_factor":1.25,"restart_overhead_s":60,"preempt_grace_s":0}`

func workload(t *testing.T, cl *cluster.Cluster, jobs int, load float64, est trace.EstimateConfig) trace.GenConfig {
	t.Helper()
	data, err := os.ReadFile("../../configs/workloads/base.json")
	if err != nil {
		t.Fatal(err)
	}
	var c trace.GenConfig
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatal(err)
	}
	c.Jobs, c.WorkCapacity, c.Estimate = jobs, cl.WorkCapacity(), est
	c.Arrival.Load = load
	c.ClassConstraints = []trace.ClassFraction{{Class: "v100", Frac: 0.1}}
	return c
}

func simulate(t *testing.T, cl *cluster.Cluster, jobs []api.Job, name, params string) *metrics.RunTrace {
	t.Helper()
	truth := map[string]time.Duration{}
	for _, j := range jobs {
		truth[j.ID] = j.Runtime
	}
	spec := policy.Spec{Name: name}
	if params != "" {
		spec.Params = json.RawMessage(params)
	}
	p, err := policy.New(spec, policy.Env{Oracle: func(id string) (time.Duration, bool) { r, ok := truth[id]; return r, ok }})
	if err != nil {
		t.Fatal(err)
	}
	rt, err := simulator.Run(simulator.Config{Cluster: cl, Jobs: jobs, Policy: p, Log: true})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := metrics.Verify(rt, cl.EmptyNodes()); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := metrics.CheckIdentities(rt); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return rt
}

// Check 5: with estimate_s >= runtime_s for every job, no job starts later
// than the shadow time computed when it first became the head of the queue
// (EASY with FIFO order; every placement; seeds 1..8, 300 jobs, load 0.9).
// With underestimates the same run counts the violations: a result, not a
// failure.
func TestEASYReservationProperty(t *testing.T) {
	cl, err := cluster.Parse([]byte(smallCluster))
	if err != nil {
		t.Fatal(err)
	}
	over := trace.EstimateConfig{Model: "user", PExact: 0.2, PUnder: 0, AccMin: 0.1,
		Popular: []float64{900, 1800, 3600, 7200, 14400, 28800, 43200, 86400, 172800}}
	under := trace.EstimateConfig{Model: "lognormal", Sigma: 0.5}
	checked, broken := 0, 0
	for _, pl := range []string{"first_fit", "best_fit", "least_fragmentation", "topology_aware"} {
		for seed := uint64(1); seed <= 8; seed++ {
			for _, est := range []trace.EstimateConfig{{Model: "exact"}, over} {
				jobs, err := trace.Generate(workload(t, cl, 300, 0.9, est), seed)
				if err != nil {
					t.Fatal(err)
				}
				for _, j := range jobs {
					if j.Estimate < j.Runtime {
						t.Fatal("estimate below run time in an exact/over model")
					}
				}
				for _, bf := range []string{"easy", "easy_oracle"} {
					rt := simulate(t, cl, jobs, "fifo+"+pl+"+"+bf, "")
					for _, r := range rt.Jobs {
						if r.HasShadow {
							checked++
							if r.FirstStart > r.Shadow {
								t.Fatalf("%s %s seed %d: job %s started at %v after its shadow %v", pl, bf, seed, r.Job.ID, r.FirstStart, r.Shadow)
							}
						}
					}
				}
			}
			jobs, err := trace.Generate(workload(t, cl, 300, 0.9, under), seed)
			if err != nil {
				t.Fatal(err)
			}
			rt := simulate(t, cl, jobs, "fifo+"+pl+"+easy", "")
			for _, r := range rt.Jobs {
				if r.HasShadow && r.FirstStart > r.Shadow {
					broken++
				}
			}
		}
	}
	if checked < 100 {
		t.Fatalf("only %d reservations checked", checked)
	}
	t.Logf("reservations checked with estimates >= run times: %d, all held; broken under lognormal underestimates: %d", checked, broken)
}

// Check 6 for every policy: invariants, identities, and byte-identical
// reruns on random traces with class constraints, gangs, topology, and
// (for the +preempt variants) preemption.
func TestInvariantsForEveryPolicy(t *testing.T) {
	cl, err := cluster.Parse([]byte(smallCluster))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, o := range []string{"fifo", "priority", "shortest_estimate", "drf"} {
		for _, pl := range []string{"first_fit", "best_fit", "least_fragmentation", "topology_aware", "hetero_ect"} {
			for _, bf := range []string{"none", "easy", "conservative", "easy_predicted", "easy_oracle", "risk"} {
				names = append(names, o+"+"+pl+"+"+bf)
			}
			names = append(names, o+"+"+pl+"+none+preempt", o+"+"+pl+"+easy+preempt")
		}
	}
	est := trace.EstimateConfig{Model: "lognormal", Sigma: 0.5}
	for seed := uint64(1); seed <= 2; seed++ {
		wl := workload(t, cl, 150, 0.95, est)
		wl.CheckpointIntervalS = 600
		jobs, err := trace.Generate(wl, seed)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range names {
			a := simulate(t, cl, jobs, name, "")
			b := simulate(t, cl, jobs, name, "")
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("%s seed %d: reruns differ", name, seed)
			}
		}
		// quotas with borrowing, strict quotas, and reclaim (base tenants t0..t3)
		shares := `{"quota_shares": {"t0": 0.25, "t1": 0.25, "t2": 0.25, "t3": 0.25}}`
		for _, name := range []string{"fifo+first_fit+easy+quota", "fifo+best_fit+none+quota_strict", "drf+topology_aware+easy+quota+reclaim",
			"priority+first_fit+easy_predicted+preempt+quota+reclaim", "shortest_estimate+hetero_ect+conservative+quota_strict+reclaim"} {
			a := simulate(t, cl, jobs, name, shares)
			b := simulate(t, cl, jobs, name, shares)
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("%s seed %d: reruns differ", name, seed)
			}
		}
	}
}

// Priority preemption end to end: the lost work in the simulator equals the
// hand computation. One node of 8 GPUs; L1 (priority 1, 4 GPUs, checkpoint
// 600 s) starts at 0, L2 (priority 1, 4 GPUs, no checkpoint) at 500; H
// (priority 8, 4 GPUs) arrives at 1000: L1 would lose 1000 - 600 = 400 s x 4
// = 1600, L2 500 s x 4 = 2000 -> L1 is the victim, losing 1600 GPU-s.
func TestPreemptionLostWorkEndToEnd(t *testing.T) {
	cl, err := cluster.Parse([]byte(`{"schema_version":1,"name":"one","classes":[{"name":"a","speed":1}],
"nodes":[{"name":"n0","rack":"r0","class":"a","gpus":8,"cpus":64,"mem_gb":64}],
"cross_node_factor":1,"cross_rack_factor":1,"restart_overhead_s":100,"preempt_grace_s":0}`))
	if err != nil {
		t.Fatal(err)
	}
	mk := func(id string, submit float64, prio int, ckpt float64) api.Job {
		return api.Job{ID: id, Submit: time.Duration(submit) * time.Second, Tenant: "t", User: "u", Priority: prio, GPUs: 4, Workers: 1,
			Topology: api.TopoAny, Runtime: 5000 * time.Second, Estimate: 5000 * time.Second, Preemptible: prio == 1,
			CheckpointInterval: time.Duration(ckpt) * time.Second}
	}
	h := mk("H", 1000, 8, 0)
	h.Runtime, h.Estimate = 100*time.Second, 100*time.Second
	rt := simulate(t, cl, []api.Job{mk("L1", 0, 1, 600), mk("L2", 500, 1, 0), h}, "priority+first_fit+none+preempt", "")
	var l1 *metrics.JobRecord
	for i := range rt.Jobs {
		if rt.Jobs[i].Job.ID == "L1" {
			l1 = &rt.Jobs[i]
		}
	}
	if l1.Preemptions != 1 || l1.LostW != 1600 || l1.Segments[0].Retained != 600 {
		t.Fatalf("L1: %d preemptions, lost %v, retained %v", l1.Preemptions, l1.LostW, l1.Segments[0].Retained)
	}
	// L1 restarts when H finishes (1100), pays 100 s overhead, needs 4400 s more.
	if l1.Completion != 5600*time.Second {
		t.Fatalf("L1 completion %v", l1.Completion)
	}
}
