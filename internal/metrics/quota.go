package metrics

import (
	"cmp"
	"math"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// QuotaFields computes the quota metrics of a run (Tier 2) over the
// window's time span, for nominal quotas floor(share * total GPUs):
//
//   - quota_satisfaction: the share of entitled demand that was served,
//     sum over tenants of the integral of min(usage, demand, quota) divided by
//     the integral of min(demand, quota), where usage is the tenant's
//     allocated GPUs and demand its allocated plus pending GPUs;
//   - borrowed_gpu_hours: the integral of max(0, usage - quota), in GPU-hours;
//   - jain_weighted_bsld: the quota-weighted Jain index over the per-tenant
//     mean bounded slowdown b_i with weights w_i (the quota shares):
//     (sum w b)^2 / (sum w * sum w b^2), 1 when all tenants are equal.
func QuotaFields(rt *RunTrace, shares map[string]float64) []Field {
	win := Window(rt)
	a, b := span(rt, win)
	nominal := map[string]float64{}
	for t, s := range shares {
		nominal[t] = math.Floor(s * float64(rt.Info.TotalGPUs))
	}
	type ev struct {
		t              time.Duration
		tenant         string
		dUse, dPending int
	}
	var evs []ev
	for i := range rt.Jobs {
		r := &rt.Jobs[i]
		if r.Infeasible {
			continue
		}
		g, t := r.Job.TotalGPUs(), r.Job.Tenant
		pendingFrom := r.Job.Submit
		for _, s := range r.Segments {
			evs = append(evs, ev{pendingFrom, t, 0, g}, ev{s.Start, t, 0, -g}, ev{s.Start, t, g, 0}, ev{s.End + s.Grace, t, -g, 0})
			pendingFrom = s.End + s.Grace
		}
	}
	slices.SortStableFunc(evs, func(x, y ev) int { return cmp.Compare(x.t, y.t) })
	use, pend := map[string]int{}, map[string]int{}
	// float sums run over tenants in name order, never in map order
	quotaTenants := sortedKeys(nominal)
	allTenants := map[string]bool{}
	for _, e := range evs {
		allTenants[e.tenant] = true
	}
	usingTenants := sortedKeys(allTenants)
	var served, entitled, borrowed float64
	prev := a
	flush := func(to time.Duration) {
		lo, hi := max(prev, a), min(to, b)
		if hi > lo {
			dt := clock.Sec(hi - lo)
			for _, t := range quotaTenants {
				q := nominal[t]
				u, d := float64(use[t]), float64(use[t]+pend[t])
				served += math.Min(math.Min(u, d), q) * dt
				entitled += math.Min(d, q) * dt
			}
			for _, t := range usingTenants {
				borrowed += math.Max(0, float64(use[t])-nominal[t]) * dt
			}
		}
		prev = max(prev, to)
	}
	for _, e := range evs {
		if e.t > prev {
			flush(e.t)
		}
		use[e.tenant] += e.dUse
		pend[e.tenant] += e.dPending
	}
	flush(b)
	// quota-weighted Jain over per-tenant mean bounded slowdown
	sum := map[string]float64{}
	n := map[string]int{}
	for _, i := range win {
		r := &rt.Jobs[i]
		sum[r.Job.Tenant] += TimesOf(rt, r).BSld
		n[r.Job.Tenant]++
	}
	var sw, swb, swb2 float64
	for _, t := range sortedKeys(shares) {
		w := shares[t]
		if n[t] == 0 || w <= 0 {
			continue
		}
		bm := sum[t] / float64(n[t])
		sw += w
		swb += w * bm
		swb2 += w * bm * bm
	}
	jw := math.NaN()
	if sw > 0 && swb2 > 0 {
		jw = swb * swb / (sw * swb2)
	}
	return []Field{{"quota_satisfaction", ratio(served, entitled)}, {"borrowed_gpu_hours", borrowed / 3600}, {"jain_weighted_bsld", jw}}
}
