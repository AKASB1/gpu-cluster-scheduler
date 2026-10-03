package policy

import (
	"fmt"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
)

// DecisionCostPolicies are the policies of the decision-cost microbenchmark.
var DecisionCostPolicies = []string{"fifo+first_fit+none", "fifo+first_fit+easy", "fifo+best_fit+none",
	"fifo+least_fragmentation+none", "drf+first_fit+none"}

// DecisionCostPending are the queue lengths of the microbenchmark.
var DecisionCostPending = []int{64, 512, 4096}

// DecisionCostView builds the fixed input of one scheduling invocation for
// the decision-cost microbenchmark: 128 nodes of 8 GPUs (16 racks), about
// half of the GPUs held by running jobs, and `pending` queued jobs drawn from
// a fixed stream (sizes 1-8 GPUs, a few gangs, four tenants).
func DecisionCostView(pending int) *api.View {
	r := rng.New(1, "decision-cost")
	c := &api.ClusterInfo{Classes: []api.GPUClass{{Name: "a100", Speed: 1}}, CrossNode: 1.1, CrossRack: 1.25,
		RestartOverhead: 120 * time.Second}
	v := &api.View{Now: 100000 * time.Second, Cluster: c}
	for i := 0; i < 128; i++ {
		rack := fmt.Sprintf("r%02d", i/8)
		c.Racks = append(c.Racks, rack)
		v.Nodes = append(v.Nodes, api.NodeState{Index: i, Name: fmt.Sprintf("%s-n%02d", rack, i%8), Rack: rack, Class: "a100", Speed: 1,
			GPUs: 8, CPUs: 128, MemMB: 1024000, FreeGPUs: 8, FreeCPUs: 128, FreeMemMB: 1024000})
		c.TotalGPUs, c.TotalCPUs, c.TotalMemMB = c.TotalGPUs+8, c.TotalCPUs+128, c.TotalMemMB+1024000
	}
	sizes := []int{1, 1, 1, 2, 2, 4, 4, 8}
	tenants := []string{"t0", "t1", "t2", "t3"}
	use := map[string]*api.TenantUsage{}
	for _, t := range tenants {
		use[t] = &api.TenantUsage{Tenant: t}
	}
	for k := 0; k < 256; k++ { // running jobs until about half of the GPUs are used
		n, g := r.IntN(128), sizes[r.IntN(len(sizes))]
		node := &v.Nodes[n]
		if node.FreeGPUs < g || c.TotalGPUs-totalFree(v) >= c.TotalGPUs/2 {
			continue
		}
		id := fmt.Sprintf("r%04d", k)
		t := tenants[r.IntN(4)]
		node.FreeGPUs -= g
		node.FreeCPUs -= 12 * g
		node.FreeMemMB -= 96000 * int64(g)
		node.Running = append(node.Running, api.NodeShare{JobID: id, Workers: 1})
		est := time.Duration(600+r.IntN(20000)) * time.Second
		v.Running = append(v.Running, api.RunningJob{JobID: id, Tenant: t, User: t + "-u0", Priority: 1, GPUs: g, Workers: 1,
			CPUs: 12 * g, MemMB: 96000 * int64(g), Topology: api.TopoAny, RunStart: v.Now - est/2, Placement: api.Placement{{Node: n, Workers: 1}},
			Rate: 1, Estimate: est, WorkDone: (est / 2).Seconds(), EstEnd: v.Now + est/2, Preemptible: true})
		use[t].RunningGPUs += g
		use[t].RunningCPUs += 12 * g
		use[t].RunningMem += 96000 * int64(g)
	}
	for k := 0; k < pending; k++ {
		g, w := sizes[r.IntN(len(sizes))], 1
		if r.IntN(20) == 0 {
			g, w = 8, 2+r.IntN(3)
		}
		t := tenants[r.IntN(4)]
		v.Pending = append(v.Pending, api.PendingJob{JobID: fmt.Sprintf("p%05d", k), Tenant: t, User: fmt.Sprintf("%s-u%d", t, r.IntN(8)),
			Submit: time.Duration(k) * 10 * time.Second, Priority: []int{1, 4, 8}[r.IntN(3)], GPUs: g, Workers: w, CPUs: 12 * g,
			MemMB: 96000 * int64(g), Topology: api.TopoAny, Estimate: time.Duration(300+r.IntN(30000)) * time.Second,
			Wait: v.Now - time.Duration(k)*10*time.Second})
	}
	slices.SortFunc(v.Pending, func(a, b api.PendingJob) int { return api.PendingLess(&a, &b) })
	for _, t := range tenants {
		v.Tenants = append(v.Tenants, *use[t])
	}
	return v
}

func totalFree(v *api.View) int {
	f := 0
	for i := range v.Nodes {
		f += v.Nodes[i].FreeGPUs
	}
	return f
}
