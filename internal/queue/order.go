// Package queue holds the order part of the Go baselines: the sequence in
// which a policy considers the pending jobs at one invocation.
package queue

import (
	"cmp"
	"fmt"
	"slices"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// Order returns the pending jobs of a view as indices into v.Pending, in the
// order the policy considers them. It is a total order (ties by job ID).
type Order interface {
	Name() string
	Sort(v *api.View) []int
}

// Params are the order parameters (only priority uses one).
type Params struct {
	// AgingPerHour raises the effective priority of a pending job by this
	// amount per hour of pending time (priority order).
	AgingPerHour float64 `json:"aging_per_hour,omitempty"`
}

// New returns the order of the given name.
func New(name string, p Params) (Order, error) {
	switch name {
	case "fifo":
		return FIFO{}, nil
	case "priority":
		if p.AgingPerHour < 0 {
			return nil, fmt.Errorf("aging_per_hour must be >= 0")
		}
		return Priority{AgingPerHour: p.AgingPerHour}, nil
	case "shortest_estimate":
		return ShortestEstimate{}, nil
	case "drf":
		return DRF{}, nil
	}
	return nil, fmt.Errorf("unknown order %q", name)
}

func indices(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
}

// FIFO orders by submit time, then job ID (priority is ignored; a preempted
// job keeps its original submit time).
type FIFO struct{}

// Name implements Order.
func (FIFO) Name() string { return "fifo" }

// Sort implements Order.
func (FIFO) Sort(v *api.View) []int {
	idx := indices(len(v.Pending))
	slices.SortFunc(idx, func(a, b int) int {
		pa, pb := &v.Pending[a], &v.Pending[b]
		if c := cmp.Compare(pa.Submit, pb.Submit); c != 0 {
			return c
		}
		return cmp.Compare(pa.JobID, pb.JobID)
	})
	return idx
}

// Priority orders by effective priority (high first) = priority +
// AgingPerHour * pending hours so far, then submit time, then job ID.
type Priority struct{ AgingPerHour float64 }

// Name implements Order.
func (Priority) Name() string { return "priority" }

// Sort implements Order.
func (p Priority) Sort(v *api.View) []int {
	eff := make([]float64, len(v.Pending))
	for i := range v.Pending {
		eff[i] = float64(v.Pending[i].Priority) + p.AgingPerHour*clock.Sec(v.Pending[i].Wait)/3600
	}
	idx := indices(len(v.Pending))
	slices.SortFunc(idx, func(a, b int) int {
		if c := cmp.Compare(eff[b], eff[a]); c != 0 {
			return c
		}
		pa, pb := &v.Pending[a], &v.Pending[b]
		if c := cmp.Compare(pa.Submit, pb.Submit); c != 0 {
			return c
		}
		return cmp.Compare(pa.JobID, pb.JobID)
	})
	return idx
}

// ShortestEstimate orders by remaining estimated work (estimate minus
// retained work), then submit time, then job ID.
type ShortestEstimate struct{}

// Name implements Order.
func (ShortestEstimate) Name() string { return "shortest_estimate" }

// Sort implements Order.
func (ShortestEstimate) Sort(v *api.View) []int {
	rem := make([]float64, len(v.Pending))
	for i := range v.Pending {
		rem[i] = max(0, clock.Sec(v.Pending[i].Estimate)-v.Pending[i].Retained)
	}
	idx := indices(len(v.Pending))
	slices.SortFunc(idx, func(a, b int) int {
		if c := cmp.Compare(rem[a], rem[b]); c != 0 {
			return c
		}
		pa, pb := &v.Pending[a], &v.Pending[b]
		if c := cmp.Compare(pa.Submit, pb.Submit); c != 0 {
			return c
		}
		return cmp.Compare(pa.JobID, pb.JobID)
	})
	return idx
}

// DRF is dominant resource fairness over tenants (Ghodsi et al.): a tenant's
// dominant share is the largest of its running GPU, CPU, and memory shares of
// the cluster. The order is built by progressive filling: repeatedly take the
// next job (priority, submit, ID) of the tenant with the lowest dominant
// share (ties: tenant name) and add that job's demand to the tenant's share,
// as if it started.
type DRF struct{}

// Name implements Order.
func (DRF) Name() string { return "drf" }

// Sort implements Order.
func (DRF) Sort(v *api.View) []int {
	c := v.Cluster
	type tenant struct {
		g, cpu float64
		mem    float64
		queue  []int
	}
	ts := map[string]*tenant{}
	var names []string
	for _, u := range v.Tenants {
		ts[u.Tenant] = &tenant{g: float64(u.RunningGPUs), cpu: float64(u.RunningCPUs), mem: float64(u.RunningMem)}
	}
	for i := range v.Pending { // v.Pending is already in (priority, submit, ID) order
		name := v.Pending[i].Tenant
		if ts[name] == nil {
			ts[name] = &tenant{}
		}
		if len(ts[name].queue) == 0 {
			names = append(names, name)
		}
		ts[name].queue = append(ts[name].queue, i)
	}
	slices.Sort(names)
	share := func(t *tenant) float64 {
		s := t.g / float64(c.TotalGPUs)
		if c.TotalCPUs > 0 {
			s = max(s, t.cpu/float64(c.TotalCPUs))
		}
		if c.TotalMemMB > 0 {
			s = max(s, t.mem/float64(c.TotalMemMB))
		}
		return s
	}
	out := make([]int, 0, len(v.Pending))
	for len(out) < len(v.Pending) {
		var best string
		bestShare := 0.0
		for _, name := range names {
			t := ts[name]
			if len(t.queue) == 0 {
				continue
			}
			if s := share(t); best == "" || s < bestShare {
				best, bestShare = name, s
			}
		}
		t := ts[best]
		i := t.queue[0]
		t.queue = t.queue[1:]
		p := &v.Pending[i]
		t.g += float64(p.TotalGPUs())
		t.cpu += float64(p.CPUs * p.Workers)
		t.mem += float64(p.MemMB * int64(p.Workers))
		out = append(out, i)
	}
	return out
}
