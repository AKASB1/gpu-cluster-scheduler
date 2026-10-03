package policy

import (
	"math"
	"slices"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/placement"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/preemption"
)

// Quotas (Tier 2), built from the concepts of the Kueue documentation
// (ClusterQueue nominal quota, cohort borrowing, reclaim by preemption) and
// not claimed to behave like Kueue:
//
//   - every tenant has a nominal GPU quota: floor(share * cluster GPUs);
//   - a job is admitted when its tenant's running GPUs plus the job's GPUs stay
//     within the nominal quota, or, with +quota (borrowing), at any level the
//     cluster can hold (the cohort is all tenants, so unused quota of others
//     is borrowed); +quota_strict never borrows, except that a job larger
//     than its tenant's whole quota is admitted when the tenant runs nothing
//     (otherwise it could never run); a job that is not admitted is skipped
//     (it does not block jobs of other tenants);
//   - jobs within their tenant's quota are considered before borrowing jobs
//     (stable partition of the base order at the start of the invocation);
//   - +reclaim: a pending job within its tenant's quota that does not fit may
//     preempt preemptible jobs of tenants that run above their nominal quota,
//     victims chosen to minimize lost work (package preemption, at most 8).

// quotaOrder initializes the usage and nominal quotas of the pass and puts
// the jobs within quota first.
func (p *pass) quotaOrder(order []int) []int {
	p.usage, p.nominal = map[string]int{}, map[string]int{}
	for _, t := range p.v.Tenants {
		p.usage[t.Tenant] = t.RunningGPUs
	}
	for t, s := range p.c.shares {
		p.nominal[t] = int(math.Floor(s * float64(p.v.Cluster.TotalGPUs)))
	}
	within := func(i int) bool {
		pj := &p.v.Pending[i]
		return p.usage[pj.Tenant]+pj.TotalGPUs() <= p.nominal[pj.Tenant]
	}
	out := make([]int, 0, len(order))
	for _, i := range order {
		if within(i) {
			out = append(out, i)
		}
	}
	for _, i := range order {
		if !within(i) {
			out = append(out, i)
		}
	}
	return out
}

// admit applies the quota admission rule (always true without quotas).
func (p *pass) admit(i int) bool {
	if !p.c.quota {
		return true
	}
	pj := &p.v.Pending[i]
	if p.usage[pj.Tenant]+pj.TotalGPUs() <= p.nominal[pj.Tenant] {
		return true
	}
	// strict quotas still admit a job larger than the tenant's whole quota
	// when the tenant runs nothing, so that every job can eventually run
	return p.c.borrow || p.usage[pj.Tenant] == 0
}

// reclaimPass lets jobs within their tenant's quota preempt jobs of
// borrowing tenants.
func (p *pass) reclaimPass(order []int) {
	exclude := map[string]bool{}
	for _, i := range order {
		pj := &p.v.Pending[i]
		if p.started[i] || p.usage[pj.Tenant]+pj.TotalGPUs() > p.nominal[pj.Tenant] {
			continue
		}
		r := placement.RequestOf(pj)
		if _, ok := p.c.placer.Place(r, p.v.Nodes, p.free); ok {
			continue // fits; the backfill rules held it back
		}
		borrowing := func(rj *api.RunningJob) bool { return p.usage[rj.Tenant] > p.nominal[rj.Tenant] }
		victims, pl, ok := preemption.SelectWith(r, borrowing, p.v.Running, exclude, p.v.Nodes, p.free, p.c.placer, 8)
		if !ok {
			continue
		}
		for _, vi := range victims {
			rj := &p.v.Running[vi]
			exclude[rj.JobID] = true
			preemption.Release(p.free, rj)
			p.usage[rj.Tenant] -= rj.GPUs * rj.Workers
			p.dec.Actions = append(p.dec.Actions, api.Action{Op: api.OpPreempt, JobID: rj.JobID})
		}
		p.start(i, pl)
	}
}

// parseSuffixes reads the optional parts after the backfill part.
func parseSuffixes(c *Composite, parts []string) bool {
	for k, s := range parts {
		if slices.Contains(parts[:k], s) {
			return false
		}
		switch s {
		case "preempt":
			c.preempt = true
		case "quota":
			c.quota, c.borrow = true, true
		case "quota_strict":
			c.quota = true
		case "reclaim":
			c.reclaim = true
		default:
			return false
		}
	}
	if slices.Contains(parts, "quota") && slices.Contains(parts, "quota_strict") {
		return false
	}
	return !c.reclaim || c.quota
}
