package metrics

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// Verify checks the invariants of a finished run (check 6): no resource is
// ever over-allocated; every feasible job completes exactly once, after its
// submission; all workers of a run start at the same instant (one start per
// segment); segments are ordered in time; class and preemptibility
// constraints hold; retained work never exceeds the work done; nothing runs on
// a node under repair; and every run's accounting matches a recomputation
// from its placement (checkAccounting). nodes are the empty nodes of the
// cluster.
func Verify(rt *RunTrace, nodes []api.NodeState) error {
	type ev struct {
		t     time.Duration
		delta int // +1 allocate, -1 free (frees first at equal times)
		node  int
		g, c  int
		m     int64
	}
	var evs []ev
	for i := range rt.Jobs {
		r := &rt.Jobs[i]
		j := &r.Job
		if r.Infeasible {
			if len(r.Segments) > 0 {
				return fmt.Errorf("job %s: infeasible but ran", j.ID)
			}
			continue
		}
		if len(r.Segments) == 0 {
			return fmt.Errorf("job %s: never ran", j.ID)
		}
		prevEnd := j.Submit
		retained := 0.0
		for k, s := range r.Segments {
			last := k == len(r.Segments)-1
			if s.Start < prevEnd || s.End < s.Start {
				return fmt.Errorf("job %s: segment %d not ordered in time", j.ID, k)
			}
			if s.Preempted == last {
				return fmt.Errorf("job %s: must complete exactly once, in its last segment", j.ID)
			}
			if s.Preempted && !s.Killed && !j.Preemptible {
				return fmt.Errorf("job %s: preempted but not preemptible", j.ID)
			}
			if s.Placement.Workers() != j.Workers {
				return fmt.Errorf("job %s: segment %d places %d workers, gang needs %d", j.ID, k, s.Placement.Workers(), j.Workers)
			}
			done := retained + s.Work
			if s.Preempted {
				if s.Retained > done+1e-9 || s.Retained < retained-1e-9 {
					return fmt.Errorf("job %s: retained %.6f outside [%.6f, %.6f]", j.ID, s.Retained, retained, done)
				}
				retained = s.Retained
			}
			for _, nw := range s.Placement {
				n := &nodes[nw.Node]
				if j.GPUClass != "" && n.Class != j.GPUClass {
					return fmt.Errorf("job %s: ran on class %s, needs %s", j.ID, n.Class, j.GPUClass)
				}
				mem := j.MemMB() * int64(nw.Workers)
				evs = append(evs, ev{s.Start, 1, nw.Node, j.GPUs * nw.Workers, j.CPUs * nw.Workers, mem},
					ev{s.End + s.Grace, -1, nw.Node, j.GPUs * nw.Workers, j.CPUs * nw.Workers, mem})
			}
			prevEnd = s.End + s.Grace
		}
		if r.Completion != r.Segments[len(r.Segments)-1].End || r.FirstStart != r.Segments[0].Start {
			return fmt.Errorf("job %s: completion or first start inconsistent with segments", j.ID)
		}
		if err := checkAccounting(r, nodes, &rt.Info); err != nil {
			return fmt.Errorf("job %s: %w", j.ID, err)
		}
	}
	// a node under repair has no capacity: its downtime counts as a full allocation
	for _, d := range rt.Downtime {
		n := &nodes[d.Node]
		evs = append(evs, ev{d.From, 1, d.Node, n.GPUs, n.CPUs, n.MemMB}, ev{d.To, -1, d.Node, n.GPUs, n.CPUs, n.MemMB})
	}
	slices.SortFunc(evs, func(a, b ev) int {
		if c := cmp.Compare(a.t, b.t); c != 0 {
			return c
		}
		return cmp.Compare(a.delta, b.delta)
	})
	g := make([]int, len(nodes))
	c := make([]int, len(nodes))
	m := make([]int64, len(nodes))
	for _, e := range evs {
		g[e.node] += e.delta * e.g
		c[e.node] += e.delta * e.c
		m[e.node] += int64(e.delta) * e.m
		n := &nodes[e.node]
		if g[e.node] > n.GPUs || c[e.node] > n.CPUs || m[e.node] > n.MemMB {
			return fmt.Errorf("node %s over-allocated at t=%v", n.Name, e.t)
		}
	}
	return nil
}

// checkAccounting recomputes every run of a job from its placement, the node
// speeds and racks, and the cluster's topology factors, restart overhead, and
// grace period (docs/simulator.md §3, §4, §7), independently of the
// simulator's bookkeeping, and compares the result with the record: the
// slowest speed, factor, and rate of each run; the restart overhead (none on
// the first start); the work done; the retained work (checkpoint rule); the
// duration of the completing run (overhead plus the remaining work at its
// rate, rounded up to a millisecond), hence rounding in [0, G * rate * 0.001);
// and every waste term and the allocated GPU-milliseconds of the job.
func checkAccounting(r *JobRecord, nodes []api.NodeState, info *api.ClusterInfo) error {
	j := &r.Job
	G := float64(j.TotalGPUs())
	runtime, ck := clock.Sec(j.Runtime), clock.Sec(j.CheckpointInterval)
	near := func(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Max(math.Abs(a), math.Abs(b))) }
	var consumed, overhead, slack, topo, lost, grace, rounding float64
	var allocMs int64
	retained := 0.0
	for k, s := range r.Segments {
		sMin := math.Inf(1)
		racks := map[string]bool{}
		for _, nw := range s.Placement {
			sMin = math.Min(sMin, nodes[nw.Node].Speed)
			racks[nodes[nw.Node].Rack] = true
		}
		f := 1.0
		switch {
		case len(s.Placement) > 1 && len(racks) > 1 && j.Topology != api.TopoAny:
			f = info.CrossRack
		case len(s.Placement) > 1 && j.Topology == api.TopoNode:
			f = info.CrossNode
		}
		rate := sMin / f
		if s.SMin != sMin || s.F != f || !near(s.Rate, rate) {
			return fmt.Errorf("segment %d: s_min %g, f %g, rate %g; recomputed %g, %g, %g", k, s.SMin, s.F, s.Rate, sMin, f, rate)
		}
		wantO := time.Duration(0)
		if k > 0 {
			wantO = info.RestartOverhead
		}
		if s.Overhead != wantO {
			return fmt.Errorf("segment %d: restart overhead %v, want %v", k, s.Overhead, wantO)
		}
		D := s.End - s.Start
		O := min(D, wantO)
		Dp := clock.Sec(D - O)
		work := Dp * rate
		if !near(s.Work, work) {
			return fmt.Errorf("segment %d: work %.9f, recomputed %.9f", k, s.Work, work)
		}
		for _, nw := range s.Placement {
			g, sp := float64(j.GPUs*nw.Workers), nodes[nw.Node].Speed
			consumed += g * sp * (clock.Sec(D) + clock.Sec(s.Grace))
			overhead += g * sp * clock.Sec(O)
			slack += g * (sp - sMin) * Dp
			grace += g * sp * clock.Sec(s.Grace)
		}
		topo += G * sMin * Dp * (1 - 1/f)
		allocMs += int64(j.TotalGPUs()) * (D + s.Grace).Milliseconds()
		done := retained + work
		if s.Preempted {
			want := 0.0
			if ck > 0 {
				want = math.Min(math.Floor(done/ck+1e-9)*ck, done)
			}
			if !near(s.Retained, want) {
				return fmt.Errorf("segment %d: retained %.9f, checkpoint rule gives %.9f", k, s.Retained, want)
			}
			wantGrace := info.PreemptGrace
			if s.Killed {
				wantGrace = 0
			}
			if s.Grace != wantGrace {
				return fmt.Errorf("segment %d: grace %v, want %v", k, s.Grace, wantGrace)
			}
			lost += G * (done - s.Retained)
			retained = s.Retained
			continue
		}
		if s.Grace != 0 || D-O != clock.WorkToDuration(runtime-retained, rate) {
			return fmt.Errorf("segment %d: completing run lasts %v after overhead, the remaining work %.6f at rate %g needs %v",
				k, D-O, runtime-retained, rate, clock.WorkToDuration(runtime-retained, rate))
		}
		rounding = G * (done - runtime)
		if rounding < -1e-6*G || rounding > G*rate*0.001*(1+1e-9) {
			return fmt.Errorf("rounding %.9g outside [0, G * rate * 0.001)", rounding)
		}
	}
	for _, c := range []struct {
		name      string
		got, want float64
	}{{"consumed", r.ConsumedW, consumed}, {"overhead", r.OverheadW, overhead}, {"slack", r.SlackW, slack}, {"topology", r.TopoW, topo},
		{"lost", r.LostW, lost}, {"grace", r.GraceW, grace}, {"rounding", r.RoundingW, rounding}, {"useful", r.UsefulW, G * runtime}} {
		if !near(c.got, c.want) {
			return fmt.Errorf("%s %.9f, recomputed from the segments %.9f", c.name, c.got, c.want)
		}
	}
	if r.AllocGPUms != allocMs {
		return fmt.Errorf("allocated %d GPU-ms, recomputed %d", r.AllocGPUms, allocMs)
	}
	return nil
}

// CheckIdentities checks the accounting identities of check 4 on one run:
// the integral of jobs in the system equals the sum of JCTs (exact, ms); the
// integral of allocated GPUs equals the allocated GPU-seconds (exact, ms);
// and per job, consumed = useful + lost + overhead + topology + slack +
// rounding + grace (relative tolerance 1e-9, float sums).
func CheckIdentities(rt *RunTrace) error {
	var jct, alloc int64
	for i := range rt.Jobs {
		r := &rt.Jobs[i]
		if r.Infeasible {
			continue
		}
		jct += (r.Completion - r.Job.Submit).Milliseconds()
		alloc += r.AllocGPUms
		sum := r.UsefulW + r.LostW + r.OverheadW + r.TopoW + r.SlackW + r.RoundingW + r.GraceW
		if math.Abs(sum-r.ConsumedW) > 1e-9*math.Max(1, r.ConsumedW) {
			return fmt.Errorf("job %s: consumed %.9f != decomposition %.9f", r.Job.ID, r.ConsumedW, sum)
		}
		if r.RoundingW < -1e-6 || r.LostW < -1e-9 {
			return fmt.Errorf("job %s: negative waste term (rounding %.3g, lost %.3g)", r.Job.ID, r.RoundingW, r.LostW)
		}
	}
	if jct != rt.NIntegral {
		return fmt.Errorf("Little identity: integral of N = %d job-ms, sum of JCT = %d ms", rt.NIntegral, jct)
	}
	if alloc != rt.AllocIntMs {
		return fmt.Errorf("allocation identity: integral = %d GPU-ms, sum over jobs = %d GPU-ms", rt.AllocIntMs, alloc)
	}
	return nil
}
