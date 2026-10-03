package policy

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/placement"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/preemption"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/queue"
)

func s(x float64) time.Duration { return time.Duration(x * float64(time.Second)) }

// node builds a node state; racks group nodes in index order.
func node(i int, rack string, gpus, free int) api.NodeState {
	return api.NodeState{Index: i, Name: fmt.Sprintf("n%d", i), Rack: rack, Class: "a", Speed: 1, GPUs: gpus, CPUs: 1000,
		MemMB: 1 << 30, FreeGPUs: free, FreeCPUs: 1000, FreeMemMB: 1 << 30}
}

func pend(id string, submit float64, prio, gpus, workers int, est float64) api.PendingJob {
	return api.PendingJob{JobID: id, Tenant: "t", User: "u", Submit: s(submit), Priority: prio, GPUs: gpus, Workers: workers,
		Topology: api.TopoAny, Estimate: s(est)}
}

func running(id string, prio, gpus, workers int, pl api.Placement, end float64) api.RunningJob {
	return api.RunningJob{JobID: id, Tenant: "t", User: "u", Priority: prio, GPUs: gpus, Workers: workers, Topology: api.TopoAny,
		Placement: pl, Rate: 1, EstEnd: s(end), Estimate: s(end)}
}

func view(nodes []api.NodeState, run []api.RunningJob, pending ...api.PendingJob) *api.View {
	c := &api.ClusterInfo{Classes: []api.GPUClass{{Name: "a", Speed: 1}}, CrossNode: 1, CrossRack: 1}
	for _, n := range nodes {
		c.TotalGPUs += n.GPUs
		c.TotalCPUs += n.CPUs
		c.TotalMemMB += n.MemMB
	}
	return &api.View{Cluster: c, Nodes: nodes, Running: run, Pending: pending}
}

func mustNew(t *testing.T, name, params string) Policy {
	t.Helper()
	spec := Spec{Name: name}
	if params != "" {
		spec.Params = json.RawMessage(params)
	}
	p, err := New(spec, Env{Oracle: func(string) (time.Duration, bool) { return 0, false }})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func ids(v *api.View, idx []int) string {
	out := ""
	for _, i := range idx {
		out += v.Pending[i].JobID
	}
	return out
}

func acts(d api.Decision) string {
	out := ""
	for _, a := range d.Actions {
		out += fmt.Sprintf("%s:%s%v ", a.Op, a.JobID, a.Placement)
	}
	return out
}

func TestOrders(t *testing.T) {
	v := view([]api.NodeState{node(0, "r", 8, 8)}, nil,
		pend("a", 5, 1, 1, 1, 50), pend("c", 1, 4, 1, 1, 10), pend("b", 1, 1, 1, 1, 30))
	v.Pending[0].Wait = s(10 * 3600)
	o, _ := queue.New("fifo", queue.Params{})
	if got := ids(v, o.Sort(v)); got != "bca" {
		t.Errorf("fifo %s", got)
	}
	o, _ = queue.New("priority", queue.Params{})
	if got := ids(v, o.Sort(v)); got != "cba" {
		t.Errorf("priority %s", got)
	}
	// aging 0.5/h: a waited 10 h -> effective 6 > 4
	o, _ = queue.New("priority", queue.Params{AgingPerHour: 0.5})
	if got := ids(v, o.Sort(v)); got != "acb" {
		t.Errorf("priority with aging %s", got)
	}
	v.Pending[1].Retained = 5 // c: 10 - 5 = 5 remaining
	o, _ = queue.New("shortest_estimate", queue.Params{})
	if got := ids(v, o.Sort(v)); got != "cba" {
		t.Errorf("shortest_estimate %s", got)
	}
}

// DRF by hand: cluster of 32 GPUs, 1000 CPUs; tenant A runs 8 GPUs (share
// 0.25), B nothing. B gets its jobs first: b1 (4 GPUs) -> 0.125, b2 (8) ->
// 0.375 > 0.25, then A: a1 (4) -> 0.375 = B's, tie broken by name (A), so
// a1, a2, then b3.
func TestDRFOrderByHand(t *testing.T) {
	nodes := []api.NodeState{node(0, "r", 16, 8), node(1, "r", 16, 16)}
	v := view(nodes, nil)
	add := func(id, ten string, submit float64, gpus int) {
		p := pend(id, submit, 1, gpus, 1, 10)
		p.Tenant = ten
		v.Pending = append(v.Pending, p)
	}
	add("a1", "A", 1, 4)
	add("a2", "A", 2, 4)
	add("b1", "B", 3, 4)
	add("b2", "B", 4, 8)
	add("b3", "B", 5, 4)
	v.Tenants = []api.TenantUsage{{Tenant: "A", RunningGPUs: 8}, {Tenant: "B"}}
	o, _ := queue.New("drf", queue.Params{})
	if got := ids(v, o.Sort(v)); got != "b1b2a1a2b3" {
		t.Fatalf("drf order %s", got)
	}
}

func TestPlacements(t *testing.T) {
	nodes := []api.NodeState{node(0, "r0", 8, 6), node(1, "r0", 8, 4), node(2, "r1", 8, 8)}
	free := placement.FreeOf(nodes)
	req := func(g, w int) placement.Request { return placement.Request{GPUs: g, Workers: w, Topology: api.TopoAny} }
	check := func(pl placement.Placer, r placement.Request, want string) {
		t.Helper()
		got, ok := pl.Place(r, nodes, free)
		if fmt.Sprint(got, ok) != want {
			t.Errorf("%s %+v: got %v %v, want %s", pl.Name(), r, got, ok, want)
		}
	}
	check(placement.FirstFit{}, req(4, 2), "[{0 1} {1 1}] true")
	check(placement.FirstFit{}, req(8, 1), "[{2 1}] true")
	check(placement.FirstFit{}, req(8, 2), "[] false")
	check(placement.BestFit{}, req(4, 1), "[{1 1}] true")
	check(placement.BestFit{}, req(2, 2), "[{1 2}] true")
	check(placement.TopologyAware{}, req(4, 2), "[{2 2}] true")
	check(placement.TopologyAware{}, req(2, 5), "[{0 3} {1 2}] true")
	check(placement.TopologyAware{}, req(4, 3), "[{0 1} {2 2}] true")
	// least fragmentation: p(1) = p(4) = 0.5; free 5 and 4. A 1-GPU worker on
	// the 5-free node reduces stranded GPUs (0.5 -> 0); on the 4-free node it
	// adds 1.5. Best fit would take the 4-free node.
	d := &metrics.SizeDist{Sizes: []int{1, 4}, Probs: []float64{0.5, 0.5}}
	n2 := []api.NodeState{node(0, "r", 8, 5), node(1, "r", 8, 4)}
	lf := placement.LeastFrag{Dist: func() *metrics.SizeDist { return d }}
	if got, _ := lf.Place(req(1, 1), n2, placement.FreeOf(n2)); fmt.Sprint(got) != "[{0 1}]" {
		t.Errorf("least_fragmentation %v", got)
	}
	if got, _ := (placement.BestFit{}).Place(req(1, 1), n2, placement.FreeOf(n2)); fmt.Sprint(got) != "[{1 1}]" {
		t.Errorf("best_fit %v", got)
	}
	// class constraint
	nodes[2].Class = "b"
	r := req(8, 1)
	r.Class = "a"
	check(placement.FirstFit{}, r, "[] false")
}

func TestNoBackfillBlocksAtTheHead(t *testing.T) {
	v := view([]api.NodeState{node(0, "r", 8, 4)}, nil, pend("a", 1, 1, 2, 1, 10), pend("b", 2, 1, 8, 1, 10), pend("c", 3, 1, 1, 1, 10))
	d, _ := mustNew(t, "fifo+first_fit+none", "").Schedule(v)
	if acts(d) != "start:a[{0 1}] " {
		t.Fatalf("got %s", acts(d))
	}
}

// EASY by hand. Nodes n0 (8 free), n1 (running R, 8 GPUs, est end 100), n2
// (running S, 4 of 8 GPUs, est end 200). Head H = 8x2 does not fit now;
// shadow = 100 (after R) with the reservation on n0, n1; extra = 4 GPUs on
// n2. C (4 GPUs, est 500) ends after the shadow but fits into extra: start
// on n2. D (8 GPUs, est 50) ends before the shadow: start on n0. E (8 GPUs,
// est 500): no.
func TestEASYByHand(t *testing.T) {
	nodes := []api.NodeState{node(0, "r", 8, 8), node(1, "r", 8, 0), node(2, "r", 8, 4)}
	run := []api.RunningJob{running("R", 1, 8, 1, api.Placement{{Node: 1, Workers: 1}}, 100),
		running("S", 1, 4, 1, api.Placement{{Node: 2, Workers: 1}}, 200)}
	v := view(nodes, run, pend("H", 1, 1, 8, 2, 10), pend("C", 2, 1, 4, 1, 500), pend("D", 3, 1, 8, 1, 50), pend("E", 4, 1, 8, 1, 500))
	d, _ := mustNew(t, "fifo+first_fit+easy", "").Schedule(v)
	if got := acts(d); got != "start:C[{2 1}] start:D[{0 1}] " {
		t.Fatalf("got %s", got)
	}
	if len(d.Reservations) != 1 || d.Reservations[0].JobID != "H" || d.Reservations[0].Shadow != s(100) {
		t.Fatalf("reservations %+v", d.Reservations)
	}
	// Without the extra capacity (S uses all of n2), C must wait.
	v.Nodes[2].FreeGPUs = 0
	v.Running[1].GPUs = 8
	d, _ = mustNew(t, "fifo+first_fit+easy", "").Schedule(v)
	if got := acts(d); got != "start:D[{0 1}] " {
		t.Fatalf("got %s", got)
	}
}

// Conservative by hand: one node of 12 GPUs, R runs 8 GPUs until 100.
// A (8 GPUs, est 50) is reserved at [100, 150); B (4 GPUs, est 30) fits now
// for its whole duration: it starts. C (4 GPUs, est 200) would overlap A's
// reservation at 100 when started now (4 free now, 12-8-4 = 0 left during A
// with B gone by then: 12-8 = 4 >= 4... so it fits [0,200)?) -- no: during
// [0,100) R holds 8 and B 4 until 30, C needs 4 -> 12-8-4-4 < 0 at [0,30).
// C is reserved at 30? During [30,100): 12-8 = 4 free -> C fits; during
// [100,150): A takes 8, R gone -> 4 free -> fits; so C's reservation is 30.
func TestConservativeByHand(t *testing.T) {
	nodes := []api.NodeState{node(0, "r", 12, 4)}
	run := []api.RunningJob{running("R", 1, 8, 1, api.Placement{{Node: 0, Workers: 1}}, 100)}
	v := view(nodes, run, pend("A", 1, 1, 8, 1, 50), pend("B", 2, 1, 4, 1, 30), pend("C", 3, 1, 4, 1, 200))
	d, _ := mustNew(t, "fifo+first_fit+conservative", "").Schedule(v)
	if got := acts(d); got != "start:B[{0 1}] " {
		t.Fatalf("got %s", got)
	}
	if fmt.Sprint(d.Reservations) != "[{A 1m40s} {C 30s}]" {
		t.Fatalf("reservations %v", d.Reservations)
	}
}

func TestPredictorRule(t *testing.T) {
	p := &predictor{k: 2, last: map[string][]float64{}}
	p.update([]api.CompletedJob{{User: "u", Runtime: s(100)}, {User: "u", Runtime: s(200)}, {User: "u", Runtime: s(300)}, {User: "w", Runtime: s(7)}})
	if got := p.predict("u", 1000); got != 250 {
		t.Fatalf("mean of the last two: %v", got)
	}
	if got := p.predict("u", 240); got != 240 {
		t.Fatalf("never above the estimate: %v", got)
	}
	if got := p.predict("x", 99); got != 99 {
		t.Fatalf("no history -> estimate: %v", got)
	}
	// A running job that outlived its prediction falls back to the estimate.
	c := mustNew(t, "fifo+first_fit+easy_predicted", "").(*Composite)
	c.pred = p
	v := view([]api.NodeState{node(0, "r", 8, 0)}, nil)
	v.Now = s(400)
	r := running("R", 1, 8, 1, api.Placement{{Node: 0, Workers: 1}}, 1000)
	r.Estimate = s(1000)
	if got := c.runningEnd(v, &r); got != s(1000) {
		t.Fatalf("outlived prediction: end %v", got)
	}
	v.Now = s(100)
	if got := c.runningEnd(v, &r); got != s(250) {
		t.Fatalf("predicted end %v", got)
	}
}

// Priority preemption by hand: one node of 8 GPUs fully used by L1 and L2
// (priority 1, 4 GPUs each). L1 did 1000 s of work with checkpoints every
// 600 s: it would lose 400 s x 4 GPUs = 1600. L2 did 500 s without
// checkpoints: it would lose 2000. H (priority 8, 4 GPUs) preempts L1.
func TestPriorityPreemptionByHand(t *testing.T) {
	nodes := []api.NodeState{node(0, "r", 8, 0)}
	l1 := running("L1", 1, 4, 1, api.Placement{{Node: 0, Workers: 1}}, 5000)
	l1.Preemptible, l1.WorkDone, l1.CheckpointInterval = true, 1000, s(600)
	l2 := running("L2", 1, 4, 1, api.Placement{{Node: 0, Workers: 1}}, 5000)
	l2.Preemptible, l2.WorkDone = true, 500
	if preemption.Lost(&l1) != 1600 || preemption.Lost(&l2) != 2000 {
		t.Fatalf("lost %v %v", preemption.Lost(&l1), preemption.Lost(&l2))
	}
	v := view(nodes, []api.RunningJob{l1, l2}, pend("H", 1, 8, 4, 1, 10))
	d, _ := mustNew(t, "priority+first_fit+none+preempt", "").Schedule(v)
	if got := acts(d); got != "preempt:L1[] start:H[{0 1}] " {
		t.Fatalf("got %s", got)
	}
	// An 8-GPU job needs both; min_gap 8 forbids any victim.
	v.Pending[0].GPUs = 8
	d, _ = mustNew(t, "priority+first_fit+none+preempt", "").Schedule(v)
	if got := acts(d); got != "preempt:L1[] preempt:L2[] start:H[{0 1}] " {
		t.Fatalf("got %s", got)
	}
	d, _ = mustNew(t, "priority+first_fit+none+preempt", `{"min_gap": 8}`).Schedule(v)
	if len(d.Actions) != 0 {
		t.Fatalf("min_gap: %s", acts(d))
	}
}

func TestPlanReplay(t *testing.T) {
	p, err := NewPlan([]PlannedStart{{JobID: "b", StartS: 20, Placement: []NodeWorkers{{"n0", 1}}}, {JobID: "a", StartS: 0, Placement: []NodeWorkers{{"n0", 1}}}})
	if err != nil {
		t.Fatal(err)
	}
	v := view([]api.NodeState{node(0, "r", 8, 8)}, nil)
	d, _ := p.Schedule(v)
	if acts(d) != "start:a[{0 1}] " || d.WakeAt != s(20) {
		t.Fatalf("got %s wake %v", acts(d), d.WakeAt)
	}
	v.Now = s(25)
	if _, err := p.Schedule(v); err == nil {
		t.Fatal("a missed planned start must be an error")
	}
}

func TestFactory(t *testing.T) {
	for _, name := range []string{"fifo+first_fit+none", "priority+best_fit+easy+preempt", "drf+least_fragmentation+conservative",
		"shortest_estimate+topology_aware+easy_predicted", "fifo+first_fit+easy_oracle", "fifo+best_fit+risk"} {
		if p := mustNew(t, name, ""); p.Name() != name {
			t.Errorf("name %s", p.Name())
		}
	}
	bad := []Spec{{Name: "fifo+first_fit"}, {Name: "lifo+first_fit+none"}, {Name: "fifo+worst_fit+none"}, {Name: "fifo+first_fit+aggressive"},
		{Name: "fifo+first_fit+none+kill"}, {Name: "fifo+first_fit+none", Params: json.RawMessage(`{"bogus": 1}`)},
		{Name: "priority+first_fit+none", Params: json.RawMessage(`{"aging_per_hour": -1}`)}, {Name: "py:fifo"}}
	for _, b := range bad {
		if _, err := New(b, Env{}); err == nil {
			t.Errorf("%+v accepted", b)
		}
	}
	if _, err := New(Spec{Name: "fifo+first_fit+easy_oracle"}, Env{}); err == nil {
		t.Error("oracle without the true run times accepted")
	}
}

func TestScheduleDoesNotModifyTheView(t *testing.T) {
	nodes := []api.NodeState{node(0, "r", 8, 8), node(1, "r", 8, 0), node(2, "r", 8, 4)}
	run := []api.RunningJob{running("R", 1, 8, 1, api.Placement{{Node: 1, Workers: 1}}, 100)}
	v := view(nodes, run, pend("H", 1, 1, 8, 2, 10), pend("C", 2, 1, 4, 1, 500))
	before := fmt.Sprintf("%+v", *v)
	for _, name := range []string{"fifo+best_fit+easy", "drf+least_fragmentation+conservative", "priority+topology_aware+none+preempt"} {
		if _, err := mustNew(t, name, "").Schedule(v); err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%+v", *v) != before {
			t.Fatalf("%s modified the view", name)
		}
	}
	_ = reflect.DeepEqual
}
