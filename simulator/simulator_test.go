package simulator

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
)

// TestCapacity keeps the scenario of the scaffold's original test: two
// one-GPU jobs and one one-GPU node. Only "a" is placed at first; "b" waits
// until "a" finishes (the simulator now has time).
func TestCapacity(t *testing.T) {
	c := mustCluster(t, oneNode(1))
	rt := run(t, c, []api.Job{job("a", 0, 1, 1, 10), job("b", 0, 1, 1, 10)}, testFIFO{blocking: true})
	a, b := rec(rt, "a"), rec(rt, "b")
	if a.FirstStart != 0 || b.FirstStart != sec(10) || b.Completion != sec(20) {
		t.Fatalf("a start %v, b start %v end %v", a.FirstStart, b.FirstStart, b.Completion)
	}
	if rt.Log[0].JobID != "a" || rt.Log[0].Placement[0].Node != "n0" {
		t.Fatalf("log %+v", rt.Log)
	}
}

const twoGPU = `{"schema_version":1,"name":"p","classes":[{"name":"g","speed":1}],
"nodes":[{"name":"n0","rack":"r0","class":"g","gpus":2,"cpus":8,"mem_gb":8}],
"cross_node_factor":1,"cross_rack_factor":1,"restart_overhead_s":10,"preempt_grace_s":GRACE}`

func startOn0(id string) api.Action {
	return api.Action{Op: api.OpStart, JobID: id, Placement: api.Placement{{Node: 0, Workers: 1}}}
}

// Preemption with checkpoint loss and restart overhead, hand-computed:
// A (2 GPUs, 100 s, checkpoint every 30 s) runs from 0; at 50 it is
// preempted (work 50, retained 30, lost 20 s x 2 GPUs = 40 GPU-s) for B
// (2 GPUs, 20 s). A restarts at 70, pays 10 s overhead, needs 70 s more:
// it completes at 150. Consumed 100 + 160 = 260 = useful 200 + lost 40 +
// overhead 20.
func TestPreemptionCheckpointAndOverhead(t *testing.T) {
	c := mustCluster(t, strings.Replace(twoGPU, "GRACE", "0", 1))
	a := job("A", 0, 2, 1, 100)
	a.Preemptible, a.CheckpointInterval = true, sec(30)
	b := job("B", 50, 2, 1, 20)
	p := &scripted{at: map[time.Duration][]api.Action{
		0:       {startOn0("A")},
		sec(50): {{Op: api.OpPreempt, JobID: "A"}, startOn0("B")},
		sec(70): {startOn0("A")},
	}}
	rt := run(t, c, []api.Job{a, b}, p)
	ra := rec(rt, "A")
	if ra.Completion != sec(150) || ra.Preemptions != 1 || ra.QueueTotal != sec(20) {
		t.Fatalf("A: completion %v preemptions %d queue %v", ra.Completion, ra.Preemptions, ra.QueueTotal)
	}
	if ra.Segments[0].Retained != 30 || ra.LostW != 40 || ra.OverheadW != 20 || ra.ConsumedW != 260 || ra.UsefulW != 200 {
		t.Fatalf("A accounting %+v", ra)
	}
	// No checkpoint: everything is lost; A restarts at 70 and needs 10 + 100 s.
	a.CheckpointInterval = 0
	rt = run(t, c, []api.Job{a, b}, p)
	if ra = rec(rt, "A"); ra.LostW != 100 || ra.Completion != sec(180) {
		t.Fatalf("no checkpoint: lost %v completion %v", ra.LostW, ra.Completion)
	}
}

// With a 5 s grace period the preempted job holds its GPUs until 55; a start
// of B at 50 is invalid, B starts at 55 (requested wake-up).
func TestPreemptGrace(t *testing.T) {
	c := mustCluster(t, strings.Replace(twoGPU, "GRACE", "5", 1))
	a := job("A", 0, 2, 1, 100)
	a.Preemptible, a.CheckpointInterval = true, sec(30)
	b := job("B", 50, 2, 1, 20)
	bad := &scripted{at: map[time.Duration][]api.Action{
		0:       {startOn0("A")},
		sec(50): {{Op: api.OpPreempt, JobID: "A"}, startOn0("B")},
	}}
	_, err := Run(Config{Cluster: c, Jobs: []api.Job{a, b}, Policy: bad})
	var ae *ActionError
	if !errors.As(err, &ae) || ae.Index != 1 || !strings.Contains(err.Error(), "capacity") {
		t.Fatalf("want capacity error, got %v", err)
	}
	good := &scripted{at: map[time.Duration][]api.Action{
		0:       {startOn0("A")},
		sec(50): {{Op: api.OpPreempt, JobID: "A"}},
		sec(55): {startOn0("B")},
		sec(75): {startOn0("A")},
	}, wake: map[time.Duration]time.Duration{sec(50): sec(55)}}
	rt := run(t, c, []api.Job{a, b}, good)
	ra := rec(rt, "A")
	// 100 (first run) + 10 (grace) + 160 (10 s overhead + 70 s) = 270
	if ra.GraceW != 10 || ra.Completion != sec(155) || ra.ConsumedW != 270 {
		t.Fatalf("grace accounting %+v", ra)
	}
}

const topo = `{"schema_version":1,"name":"t","classes":[{"name":"fast","speed":1},{"name":"slow","speed":0.5}],
"nodes":[{"name":"a0","rack":"ra","class":"fast","gpus":8,"cpus":64,"mem_gb":64},
{"name":"a1","rack":"ra","class":"fast","gpus":8,"cpus":64,"mem_gb":64},
{"name":"b0","rack":"rb","class":"fast","gpus":8,"cpus":64,"mem_gb":64},
{"name":"s0","rack":"rb","class":"slow","gpus":8,"cpus":64,"mem_gb":64}],
"cross_node_factor":1.1,"cross_rack_factor":1.25,"restart_overhead_s":0,"preempt_grace_s":0}`

// Topology factor and mixed-speed slack, hand-computed. Node order:
// a0, a1 (rack ra), b0, s0 (rack rb).
func TestTopologyFactorAndMixedSpeed(t *testing.T) {
	c := mustCluster(t, topo)
	cases := []struct {
		name     string
		topo     api.Topology
		gpus, w  int
		p        api.Placement
		end      float64
		topoW    float64
		slackW   float64
		consumed float64
	}{
		// rack-sensitive over two racks: f = 1.25, 100 s -> 125 s; topo = 16*125*(1-1/1.25) = 400
		{"rack over racks", api.TopoRack, 8, 2, api.Placement{{Node: 0, Workers: 1}, {Node: 2, Workers: 1}}, 125, 400, 0, 2000},
		// node-sensitive over two nodes of one rack: f = 1.1 -> 110 s
		{"node over nodes", api.TopoNode, 4, 2, api.Placement{{Node: 0, Workers: 1}, {Node: 1, Workers: 1}}, 110, 8 * 110 * (1 - 1/1.1), 0, 880},
		// rack-sensitive within one rack: f = 1
		{"rack within rack", api.TopoRack, 8, 2, api.Placement{{Node: 0, Workers: 1}, {Node: 1, Workers: 1}}, 100, 0, 0, 1600},
		// any over a fast and a slow node: rate 0.5 -> 200 s; slack = 1 GPU *
		// (1 - 0.5) * 200 = 100; consumed 200 + 100 = 300
		{"mixed speed", api.TopoAny, 1, 2, api.Placement{{Node: 2, Workers: 1}, {Node: 3, Workers: 1}}, 200, 0, 100, 300},
	}
	for _, k := range cases {
		j := job("J", 0, k.gpus, k.w, 100)
		j.Topology = k.topo
		start := api.Action{Op: api.OpStart, JobID: "J", Placement: k.p}
		rt := run(t, c, []api.Job{j}, &scripted{at: map[time.Duration][]api.Action{0: {start}}})
		r := rec(rt, "J")
		if r.Completion != sec(k.end) || math.Abs(r.TopoW-k.topoW) > 1e-6 || math.Abs(r.SlackW-k.slackW) > 1e-9 ||
			math.Abs(r.ConsumedW-k.consumed) > 1e-6 {
			t.Errorf("%s: end %v topo %.4f slack %.4f consumed %.4f", k.name, r.Completion, r.TopoW, r.SlackW, r.ConsumedW)
		}
	}
}

func TestInvalidActionsAbortTheRun(t *testing.T) {
	c := mustCluster(t, topo)
	j := job("J", 0, 8, 2, 100)
	k := job("K", 0, 4, 1, 100)
	k.GPUClass = "slow"
	q := job("Q", 0, 4, 1, 100)
	q.Preemptible = true
	pl := func(nw ...int) api.Placement {
		var p api.Placement
		for i := 0; i < len(nw); i += 2 {
			p = append(p, api.NodeWorkers{Node: nw[i], Workers: nw[i+1]})
		}
		return p
	}
	st := func(id string, p api.Placement) api.Action {
		return api.Action{Op: api.OpStart, JobID: id, Placement: p}
	}
	cases := map[string][]api.Action{
		"unknown job":     {st("X", pl(0, 2))},
		"gang":            {st("J", pl(0, 1))},
		"capacity":        {st("J", pl(0, 2))},
		"class":           {st("K", pl(0, 1))},
		"twice":           {st("J", pl(0, 1, 1, 1)), st("J", pl(2, 1, 3, 1))},
		"not preemptible": {st("J", pl(0, 1, 1, 1)), {Op: api.OpPreempt, JobID: "J"}},
		"not running":     {{Op: api.OpPreempt, JobID: "J"}},
		"bad node":        {st("J", pl(9, 2))},
		"duplicate node":  {st("J", pl(0, 1, 0, 1))},
		"zero workers":    {st("J", pl(0, 2, 1, 0))},
		"unknown op":      {{Op: "kill", JobID: "J"}},
		"start+preempt":   {st("Q", pl(0, 1)), {Op: api.OpPreempt, JobID: "Q"}},
	}
	for name, acts := range cases {
		_, err := Run(Config{Cluster: c, Jobs: []api.Job{j, k, q}, Policy: &scripted{at: map[time.Duration][]api.Action{0: acts}}})
		var ae *ActionError
		if !errors.As(err, &ae) || ae.Policy != "scripted" {
			t.Errorf("%s: want ActionError naming the policy, got %v", name, err)
		}
	}
}

type idle struct{}

func (idle) Name() string                             { return "idle" }
func (idle) Schedule(*api.View) (api.Decision, error) { return api.Decision{}, nil }

func TestDeadlockIsAnError(t *testing.T) {
	c := mustCluster(t, oneNode(1))
	_, err := Run(Config{Cluster: c, Jobs: []api.Job{job("a", 0, 1, 1, 10)}, Policy: idle{}})
	var de *DeadlockError
	if !errors.As(err, &de) || de.Policy != "idle" {
		t.Fatalf("want deadlock naming the policy, got %v", err)
	}
}

func TestInfeasibleJobsAreRecordedAndExcluded(t *testing.T) {
	c := mustCluster(t, topo)
	big := job("big", 0, 8, 5, 10) // 40 GPUs > 32
	slow := job("slow", 0, 8, 2, 10)
	slow.GPUClass = "slow" // only one slow node with 8 GPUs
	ok := job("ok", 1, 1, 1, 10)
	rt := run(t, c, []api.Job{big, slow, ok}, testFIFO{})
	if !rec(rt, "big").Infeasible || !rec(rt, "slow").Infeasible || rec(rt, "ok").Infeasible {
		t.Fatal("infeasible classification")
	}
	if rt.NIntegral != 10000 {
		t.Fatalf("infeasible jobs must not count in the system: %d", rt.NIntegral)
	}
}

// Property test (checks 4 and 6) on random traces with random preemptions,
// a heterogeneous cluster with mixed classes and topology factors, grace on
// and off, and periodic ticks on and off. Seeds 1..5.
func TestInvariantsAndIdentitiesOnRandomTraces(t *testing.T) {
	var cfg trace.GenConfig
	cfg.Jobs = 400
	cfg.Arrival = trace.ArrivalConfig{Process: "poisson", Load: 0.9}
	cfg.Sizes = []trace.SizeClass{{Weight: 0.5, GPUs: 1, Workers: 1}, {Weight: 0.2, GPUs: 2, Workers: 2, Topology: api.TopoNode, TopoFrac: 1},
		{Weight: 0.2, GPUs: 4, Workers: 1}, {Weight: 0.1, GPUs: 8, Workers: 2, Topology: api.TopoRack, TopoFrac: 0.5}}
	cfg.Runtime = trace.RuntimeConfig{UserMedianS: 300, UserSigma: 0.5, JobSigma: 0.8, MinS: 10, MaxS: 7200}
	cfg.Tenants = []trace.TenantSpec{{Name: "a", Weight: 1, Users: 3}, {Name: "b", Weight: 1, Users: 3}}
	cfg.Priorities = []trace.PriorityClass{{Priority: 1, Weight: 1, PreemptibleFrac: 0.8}}
	cfg.CheckpointIntervalS = 120
	cfg.Resources = trace.ResourceConfig{CPUsPerGPU: 8, MemPerGPU: 7.5}
	cfg.ClassConstraints = []trace.ClassFraction{{Class: "slow", Frac: 0.1}}
	cfg.Estimate = trace.EstimateConfig{Model: "exact"}
	for _, grace := range []string{"0", "3"} {
		js := strings.Replace(topo, `"restart_overhead_s":0`, `"restart_overhead_s":30`, 1)
		cl := mustCluster(t, strings.Replace(js, `"preempt_grace_s":0`, `"preempt_grace_s":`+grace, 1))
		cfg.WorkCapacity = cl.WorkCapacity()
		for seed := uint64(1); seed <= 5; seed++ {
			jobs, err := trace.Generate(cfg, seed)
			if err != nil {
				t.Fatal(err)
			}
			for _, tick := range []time.Duration{0, sec(60)} {
				mk := func() Config {
					cfg := Config{Cluster: cl, Jobs: jobs, Policy: &testChaos{r: rng.New(seed, "chaos"), p: 0.2}, ScheduleInterval: tick, Log: true}
					if grace == "0" { // random node failures (failures with a grace period are not supported)
						fr := rng.New(seed, "failures")
						for k := 0; k < 12; k++ {
							cfg.Failures = append(cfg.Failures, Failure{Node: cl.Nodes[fr.IntN(len(cl.Nodes))].Name,
								At: time.Duration(fr.IntN(20000)) * time.Second, Repair: time.Duration(1+fr.IntN(3000)) * time.Second})
						}
					}
					return cfg
				}
				a, err := Run(mk())
				if err != nil {
					t.Fatal(err)
				}
				if err := metrics.Verify(a, cl.EmptyNodes()); err != nil {
					t.Fatalf("grace %s seed %d: %v", grace, seed, err)
				}
				if err := metrics.CheckIdentities(a); err != nil {
					t.Fatalf("grace %s seed %d: %v", grace, seed, err)
				}
				pre := 0
				for i := range a.Jobs {
					pre += a.Jobs[i].Preemptions
				}
				if pre == 0 {
					t.Fatal("the chaos policy must preempt")
				}
				b, err := Run(mk())
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(a, b) {
					t.Fatalf("grace %s seed %d: two runs differ", grace, seed)
				}
			}
		}
	}
}

// Node failure, hand-computed: A (8 GPUs, 100 s, checkpoint every 30 s, NOT
// preemptible) runs on n0 from 0. n0 fails at 50 for 20 s: A is killed
// (work 50, retained 30, lost 20 s x 8 GPUs = 160), n0 has no capacity until
// 70, A restarts at 70 with 10 s overhead and needs 70 s more: done at 150.
// The pending A while nothing runs and n0 is down is not a deadlock.
func TestNodeFailureKillsAndRepairs(t *testing.T) {
	c := mustCluster(t, `{"schema_version":1,"name":"f","classes":[{"name":"g","speed":1}],
"nodes":[{"name":"n0","rack":"r0","class":"g","gpus":8,"cpus":8,"mem_gb":8},{"name":"n1","rack":"r0","class":"g","gpus":4,"cpus":8,"mem_gb":8}],
"cross_node_factor":1,"cross_rack_factor":1,"restart_overhead_s":10,"preempt_grace_s":0}`)
	a := job("A", 0, 8, 1, 100)
	a.CheckpointInterval = sec(30)
	rt, err := Run(Config{Cluster: c, Jobs: []api.Job{a}, Policy: testFIFO{blocking: true}, Log: true,
		Failures: []Failure{{Node: "n0", At: sec(50), Repair: sec(20)}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := metrics.Verify(rt, c.EmptyNodes()); err != nil {
		t.Fatal(err)
	}
	if err := metrics.CheckIdentities(rt); err != nil {
		t.Fatal(err)
	}
	r := rec(rt, "A")
	if r.Completion != sec(150) || r.FailureKills != 1 || r.LostW != 160 || r.Segments[0].Retained != 30 || !r.Segments[0].Killed {
		t.Fatalf("A: %+v", r)
	}
	if len(rt.Downtime) != 1 || rt.Downtime[0].From != sec(50) || rt.Downtime[0].To != sec(70) {
		t.Fatalf("downtime %+v", rt.Downtime)
	}
	if rt.Log[1].Op != api.OpKill || rt.Log[2].T != sec(70) {
		t.Fatalf("log %+v", rt.Log)
	}
	// A failure of a node that is already down is absorbed; a failure after
	// the run ended is ignored; an unknown node is an error.
	rt, err = Run(Config{Cluster: c, Jobs: []api.Job{a}, Policy: testFIFO{blocking: true},
		Failures: []Failure{{Node: "n0", At: sec(50), Repair: sec(20)}, {Node: "n0", At: sec(60), Repair: sec(100)}, {Node: "n1", At: sec(9999), Repair: sec(1)}}})
	if err != nil || rec(rt, "A").Completion != sec(150) || len(rt.Downtime) != 1 {
		t.Fatalf("absorbed failure: %v %+v", err, rt.Downtime)
	}
	if _, err := Run(Config{Cluster: c, Jobs: []api.Job{a}, Policy: testFIFO{}, Failures: []Failure{{Node: "x", At: 0, Repair: sec(1)}}}); err == nil {
		t.Fatal("unknown node accepted")
	}
}

// The waste decomposition alone holds by construction (the rounding term is
// the remainder), so Verify recomputes every run from its placement. Each
// corruption below keeps CheckIdentities satisfied and must fail Verify.
func TestVerifyRecomputesTheAccounting(t *testing.T) {
	c := mustCluster(t, strings.Replace(twoGPU, "GRACE", "0", 1))
	a := job("A", 0, 2, 1, 100)
	a.Preemptible, a.CheckpointInterval = true, sec(30)
	b := job("B", 50, 2, 1, 20)
	p := &scripted{at: map[time.Duration][]api.Action{
		0:       {startOn0("A")},
		sec(50): {{Op: api.OpPreempt, JobID: "A"}, startOn0("B")},
		sec(70): {startOn0("A")},
	}}
	cases := map[string]func(rt *metrics.RunTrace, r *metrics.JobRecord){
		// the completing run lasts 1 s too long (as if the overhead were
		// counted twice); the extra time is booked as rounding
		"duration": func(rt *metrics.RunTrace, r *metrics.JobRecord) {
			s := &r.Segments[len(r.Segments)-1]
			s.End += time.Second
			r.Completion += time.Second
			r.ConsumedW += 2
			r.RoundingW += 2
			r.AllocGPUms += 2000
			rt.NIntegral += 1000
			rt.AllocIntMs += 2000
		},
		// a wrong slowest speed shifts progress time into slack
		"s_min": func(_ *metrics.RunTrace, r *metrics.JobRecord) {
			r.Segments[0].SMin = 0.5
			r.SlackW += 10
			r.TopoW -= 10
		},
		// retained work that is not a multiple of the checkpoint interval
		"checkpoint": func(_ *metrics.RunTrace, r *metrics.JobRecord) {
			r.Segments[0].Retained = 35
			r.LostW -= 10
			r.RoundingW += 10
		},
	}
	for name, corrupt := range cases {
		rt, err := Run(Config{Cluster: c, Jobs: []api.Job{a, b}, Policy: p})
		if err != nil {
			t.Fatal(err)
		}
		corrupt(rt, rec(rt, "A"))
		if err := metrics.CheckIdentities(rt); err != nil {
			t.Fatalf("%s: corruption should keep the identities: %v", name, err)
		}
		if err := metrics.Verify(rt, c.EmptyNodes()); err == nil {
			t.Errorf("%s: Verify accepted a corrupted record", name)
		}
	}
}
