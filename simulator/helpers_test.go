package simulator

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
)

// testFIFO is a minimal policy for simulator tests: pending jobs in view
// order, first-fit placement filling nodes in index order; blocking stops at
// the first job that does not fit.
type testFIFO struct{ blocking bool }

func (testFIFO) Name() string { return "test-fifo" }

func (p testFIFO) Schedule(v *api.View) (api.Decision, error) {
	free := freeOf(v)
	var d api.Decision
	for i := range v.Pending {
		pl, ok := firstFit(&v.Pending[i], v.Nodes, free)
		if !ok {
			if p.blocking {
				break
			}
			continue
		}
		d.Actions = append(d.Actions, api.Action{Op: api.OpStart, JobID: v.Pending[i].JobID, Placement: pl})
	}
	return d, nil
}

type capacity struct {
	g, c int
	m    int64
}

func freeOf(v *api.View) []capacity {
	out := make([]capacity, len(v.Nodes))
	for i, n := range v.Nodes {
		out[i] = capacity{n.FreeGPUs, n.FreeCPUs, n.FreeMemMB}
	}
	return out
}

func firstFit(j *api.PendingJob, nodes []api.NodeState, free []capacity) (api.Placement, bool) {
	need := j.Workers
	var pl api.Placement
	for i := range nodes {
		if j.GPUClass != "" && nodes[i].Class != j.GPUClass {
			continue
		}
		w := free[i].g / j.GPUs
		if j.CPUs > 0 {
			w = min(w, free[i].c/j.CPUs)
		}
		if j.MemMB > 0 {
			w = min(w, int(free[i].m/j.MemMB))
		}
		w = min(w, need)
		if w > 0 {
			pl = append(pl, api.NodeWorkers{Node: i, Workers: w})
			need -= w
		}
		if need == 0 {
			break
		}
	}
	if need > 0 {
		return nil, false
	}
	for _, nw := range pl {
		free[nw.Node].g -= j.GPUs * nw.Workers
		free[nw.Node].c -= j.CPUs * nw.Workers
		free[nw.Node].m -= j.MemMB * int64(nw.Workers)
	}
	return pl, true
}

// testChaos preempts a random preemptible running job with probability p at
// each invocation, then schedules like non-blocking FIFO.
type testChaos struct {
	r *rand.Rand
	p float64
}

func (*testChaos) Name() string { return "test-chaos" }

func (c *testChaos) Schedule(v *api.View) (api.Decision, error) {
	var d api.Decision
	free := freeOf(v)
	if len(v.Running) > 0 && c.r.Float64() < c.p {
		rj := v.Running[c.r.IntN(len(v.Running))]
		if rj.Preemptible {
			d.Actions = append(d.Actions, api.Action{Op: api.OpPreempt, JobID: rj.JobID})
			if v.Cluster.PreemptGrace == 0 {
				for _, nw := range rj.Placement {
					free[nw.Node].g += rj.GPUs * nw.Workers
					free[nw.Node].c += rj.CPUs * nw.Workers
					free[nw.Node].m += rj.MemMB * int64(nw.Workers)
				}
			}
		}
	}
	for i := range v.Pending {
		if pl, ok := firstFit(&v.Pending[i], v.Nodes, free); ok {
			d.Actions = append(d.Actions, api.Action{Op: api.OpStart, JobID: v.Pending[i].JobID, Placement: pl})
		}
	}
	return d, nil
}

// scripted returns fixed actions at fixed times.
type scripted struct {
	at   map[time.Duration][]api.Action
	wake map[time.Duration]time.Duration
}

func (*scripted) Name() string { return "scripted" }

func (s *scripted) Schedule(v *api.View) (api.Decision, error) {
	return api.Decision{Actions: s.at[v.Now], WakeAt: s.wake[v.Now]}, nil
}

func mustCluster(t testing.TB, js string) *cluster.Cluster {
	t.Helper()
	c, err := cluster.Parse([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func oneNode(gpus int) string {
	return fmt.Sprintf(`{"schema_version":1,"name":"one","classes":[{"name":"g","speed":1}],
"nodes":[{"name":"n0","rack":"r0","class":"g","gpus":%d,"cpus":1,"mem_gb":1}],
"cross_node_factor":1,"cross_rack_factor":1,"restart_overhead_s":0,"preempt_grace_s":0}`, gpus)
}

func sec(s float64) time.Duration { return time.Duration(s*1000) * time.Millisecond }

func job(id string, submit float64, gpus, workers int, runtime float64) api.Job {
	return api.Job{ID: id, Submit: sec(submit), Tenant: "t", User: "u", Priority: 1, GPUs: gpus, Workers: workers,
		Topology: api.TopoAny, Runtime: sec(runtime), Estimate: sec(runtime)}
}

func run(t testing.TB, c *cluster.Cluster, jobs []api.Job, p policy.Policy) *metrics.RunTrace {
	t.Helper()
	rt, err := Run(Config{Cluster: c, Jobs: jobs, Policy: p, Log: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := metrics.Verify(rt, c.EmptyNodes()); err != nil {
		t.Fatal(err)
	}
	if err := metrics.CheckIdentities(rt); err != nil {
		t.Fatal(err)
	}
	return rt
}

func rec(rt *metrics.RunTrace, id string) *metrics.JobRecord {
	for i := range rt.Jobs {
		if rt.Jobs[i].Job.ID == id {
			return &rt.Jobs[i]
		}
	}
	return nil
}
