package policy

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/placement"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/preemption"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/queue"
)

// Backfill modes.
const (
	BackfillNone          = "none"
	BackfillEASY          = "easy"
	BackfillConservative  = "conservative"
	BackfillEASYPredicted = "easy_predicted"
	BackfillEASYOracle    = "easy_oracle"
	// BackfillRisk is risk_backfill (Tier 2): EASY in which the shadow time
	// and the backfill test use the q-quantile of the empirical run-time /
	// estimate ratio (the user's, else all users') times the estimate.
	BackfillRisk = "risk"
)

// Params are the parameters of a composed policy (all optional).
type Params struct {
	// AgingPerHour: priority order, effective priority gain per pending hour.
	AgingPerHour float64 `json:"aging_per_hour,omitempty"`
	// MinGap and MaxVictims: priority preemption (see package preemption).
	MinGap     int `json:"min_gap,omitempty"`
	MaxVictims int `json:"max_victims,omitempty"`
	// Depth bounds the number of queued jobs conservative backfilling
	// reserves for at one invocation (default 64).
	Depth int `json:"depth,omitempty"`
	// History is the number of a user's last completed jobs whose mean run
	// time predicts the next one (easy_predicted; default 2, after Tsafrir,
	// Etsion, and Feitelson).
	History int `json:"history,omitempty"`
	// Quantile q of risk backfilling (default 0.9) and the number of a user's
	// completed jobs needed before the user's own ratios are used (default 5).
	Quantile   float64 `json:"quantile,omitempty"`
	MinHistory int     `json:"min_history,omitempty"`
	// QuotaShares: nominal GPU quota per tenant as a share of the cluster's
	// GPUs (required by +quota, +quota_strict).
	QuotaShares map[string]float64 `json:"quota_shares,omitempty"`
}

// Composite is a Go baseline built from four parts: order, placement,
// backfill, and preemption.
type Composite struct {
	name     string
	order    queue.Order
	placer   placement.Placer
	backfill string
	preempt  bool
	pp       preemption.Params
	depth    int
	pred     *predictor
	oracle   func(string) (time.Duration, bool)
	risk     *ratioModel
	sizes    sizeObserver
	// quotas (Tier 2): nominal quota per tenant, cohort borrowing, reclaim
	quota   bool
	borrow  bool
	reclaim bool
	shares  map[string]float64
}

// Name implements Policy.
func (c *Composite) Name() string { return c.name }

// releaseEntry is a running (or just started) job's estimated end.
type releaseEntry struct {
	end time.Duration
	id  string
	req placement.Request
	pl  api.Placement
}

type pass struct {
	c       *Composite
	v       *api.View
	free    []placement.Cap
	dec     api.Decision
	started map[int]bool
	rel     []releaseEntry // releases for the shadow computation
	usage   map[string]int // running GPUs per tenant (quotas)
	nominal map[string]int // nominal GPU quota per tenant (quotas)
}

// Schedule implements Policy.
func (c *Composite) Schedule(v *api.View) (api.Decision, error) {
	c.sizes.observe(v)
	if va, ok := c.placer.(placement.ViewAware); ok {
		va.Prepare(v)
	}
	if c.risk != nil {
		c.risk.update(v.History)
	}
	if c.pred != nil {
		c.pred.update(v.History)
	}
	p := &pass{c: c, v: v, free: placement.FreeOf(v.Nodes), started: map[int]bool{}}
	for i := range v.Running {
		r := &v.Running[i]
		p.rel = append(p.rel, releaseEntry{end: c.runningEnd(v, r), id: r.JobID,
			req: placement.Request{GPUs: r.GPUs, CPUs: r.CPUs, MemMB: r.MemMB, Workers: r.Workers}, pl: r.Placement})
	}
	order := c.order.Sort(v)
	if c.quota {
		order = p.quotaOrder(order)
	}
	switch c.backfill {
	case BackfillNone:
		for _, i := range order {
			if !p.admit(i) {
				continue
			}
			if !p.tryStart(i) {
				break
			}
		}
	case BackfillEASY, BackfillEASYPredicted, BackfillEASYOracle, BackfillRisk:
		if err := p.easy(order); err != nil {
			return api.Decision{}, err
		}
	case BackfillConservative:
		p.conservative(order)
	}
	if c.reclaim {
		p.reclaimPass(order)
	}
	if c.preempt {
		if v.Cluster.PreemptGrace > 0 {
			return api.Decision{}, errors.New("priority preemption assumes preempt_grace_s = 0")
		}
		p.preemptPass(order)
	}
	return p.dec, nil
}

// start records a start action of pending job i with placement pl.
func (p *pass) start(i int, pl api.Placement) {
	pj := &p.v.Pending[i]
	r := placement.RequestOf(pj)
	placement.Take(p.free, r, pl)
	p.started[i] = true
	if p.usage != nil {
		p.usage[pj.Tenant] += pj.TotalGPUs()
	}
	p.dec.Actions = append(p.dec.Actions, api.Action{Op: api.OpStart, JobID: pj.JobID, Placement: pl})
	p.rel = append(p.rel, releaseEntry{end: p.v.Now + p.c.shadowDuration(p.v, pj, pl), id: pj.JobID, req: r, pl: pl})
}

func (p *pass) tryStart(i int) bool {
	pl, ok := p.c.placer.Place(placement.RequestOf(&p.v.Pending[i]), p.v.Nodes, p.free)
	if ok {
		p.start(i, pl)
	}
	return ok
}

// estimatedDuration is a pending job's expected duration with placement pl,
// from its user estimate (or the true run time for the oracle).
func (c *Composite) estimatedDuration(v *api.View, pj *api.PendingJob, pl api.Placement) time.Duration {
	work := clock.Sec(pj.Estimate)
	switch c.backfill {
	case BackfillEASYOracle:
		if rt, ok := c.oracle(pj.JobID); ok {
			work = clock.Sec(rt)
		}
	case BackfillRisk:
		work = c.risk.work(pj.User, work)
	}
	return duration(v, pj, pl, work-pj.Retained)
}

// shadowDuration is the duration used for release times in the shadow
// computation: the prediction for easy_predicted, else as estimatedDuration.
func (c *Composite) shadowDuration(v *api.View, pj *api.PendingJob, pl api.Placement) time.Duration {
	if c.backfill == BackfillEASYPredicted {
		return duration(v, pj, pl, c.pred.predict(pj.User, clock.Sec(pj.Estimate))-pj.Retained)
	}
	return c.estimatedDuration(v, pj, pl)
}

func duration(v *api.View, pj *api.PendingJob, pl api.Placement, work float64) time.Duration {
	rate, _, _ := api.RateOf(pj.Topology, pl, v.Nodes, v.Cluster)
	var overhead time.Duration
	if pj.Started {
		overhead = v.Cluster.RestartOverhead
	}
	return api.EstimatedDuration(max(work, 0), overhead, rate)
}

// runningEnd is a running job's end time as this policy estimates it.
func (c *Composite) runningEnd(v *api.View, r *api.RunningJob) time.Duration {
	end := func(work float64) time.Duration {
		return r.RunStart + api.EstimatedDuration(work-r.RetainedAtStart, r.Overhead, r.Rate)
	}
	switch c.backfill {
	case BackfillEASYOracle:
		if rt, ok := c.oracle(r.JobID); ok {
			return end(clock.Sec(rt))
		}
	case BackfillEASYPredicted:
		// The prediction; when the job has outlived it, the prediction is
		// raised to the user estimate (and the overrun rule applies beyond).
		if e := end(c.pred.predict(r.User, clock.Sec(r.Estimate))); e >= v.Now {
			return e
		}
	case BackfillRisk:
		// The quantile prediction; once outlived, the user estimate (and the
		// overrun rule beyond it), as for easy_predicted.
		if e := end(c.risk.work(r.User, clock.Sec(r.Estimate))); e >= v.Now {
			return e
		}
	}
	return r.EstEnd
}

// shadow computes the head's reservation: release jobs in order of estimated
// end, test the placement after each release. It returns the shadow time, the
// reserved placement, and the extra capacity (free at the shadow time minus
// the reservation).
func (p *pass) shadow(r placement.Request) (time.Duration, api.Placement, []placement.Cap, bool) {
	rel := slices.Clone(p.rel)
	slices.SortFunc(rel, func(a, b releaseEntry) int {
		if c := cmp.Compare(a.end, b.end); c != 0 {
			return c
		}
		return cmp.Compare(a.id, b.id)
	})
	capS := slices.Clone(p.free)
	for k := 0; k < len(rel); k++ {
		placement.Give(capS, rel[k].req, rel[k].pl)
		pl, ok := p.c.placer.Place(r, p.v.Nodes, capS)
		if !ok {
			continue
		}
		// everything that ends at the same instant is free at the shadow time
		for k+1 < len(rel) && rel[k+1].end == rel[k].end {
			k++
			placement.Give(capS, rel[k].req, rel[k].pl)
		}
		placement.Take(capS, r, pl)
		return rel[k].end, pl, capS, true
	}
	return 0, nil, nil, false
}

// easy is EASY backfilling: start jobs in order while they fit; the first
// job that does not fit (the head) gets a reservation at its shadow time; a
// later job starts now if its estimated end is not after the shadow time, or
// if it fits into the capacity the reservation leaves free (extra).
func (p *pass) easy(order []int) error {
	head := -1
	var shadow time.Duration
	var extra []placement.Cap
	for _, i := range order {
		if !p.admit(i) {
			continue
		}
		pj := &p.v.Pending[i]
		r := placement.RequestOf(pj)
		if head < 0 {
			if p.tryStart(i) {
				continue
			}
			head = i
			var ok bool
			shadow, _, extra, ok = p.shadow(r)
			if !ok {
				return nil // no reservation possible (grace-held GPUs): no backfilling
			}
			p.dec.Reservations = append(p.dec.Reservations, api.Reservation{JobID: pj.JobID, Shadow: shadow})
			continue
		}
		pl, ok := p.c.placer.Place(r, p.v.Nodes, p.free)
		if !ok {
			continue
		}
		if p.v.Now+p.c.estimatedDuration(p.v, pj, pl) <= shadow {
			p.start(i, pl)
			continue
		}
		capX := make([]placement.Cap, len(p.free))
		for n := range capX {
			capX[n] = placement.Cap{GPUs: min(p.free[n].GPUs, extra[n].GPUs), CPUs: min(p.free[n].CPUs, extra[n].CPUs),
				MemMB: min(p.free[n].MemMB, extra[n].MemMB)}
		}
		if pl2, ok := p.c.placer.Place(r, p.v.Nodes, capX); ok {
			p.start(i, pl2)
			placement.Take(extra, r, pl2)
		}
	}
	return nil
}

// segment of the conservative profile: capacity per node on [t, next t).
type segment struct {
	t   time.Duration
	cap []placement.Cap
}

// conservative is conservative backfilling: every queued job (up to Depth)
// gets a reservation at the earliest time its placement fits for its whole
// estimated duration, given the running jobs and the earlier reservations;
// jobs whose reservation is now start now.
func (p *pass) conservative(order []int) {
	// profile from the running jobs' estimated ends
	rel := slices.Clone(p.rel)
	slices.SortFunc(rel, func(a, b releaseEntry) int { return cmp.Compare(a.end, b.end) })
	prof := []segment{{t: p.v.Now, cap: slices.Clone(p.free)}}
	for _, e := range rel {
		// an overrun job (estimated end = now) still holds its resources now:
		// it is assumed to end one millisecond later
		end := max(e.end, p.v.Now+time.Millisecond)
		last := prof[len(prof)-1]
		if end > last.t {
			prof = append(prof, segment{t: end, cap: slices.Clone(last.cap)})
		}
		placement.Give(prof[len(prof)-1].cap, e.req, e.pl)
	}
	minCap := func(from int, end time.Duration) []placement.Cap {
		m := slices.Clone(prof[from].cap)
		for k := from + 1; k < len(prof) && prof[k].t < end; k++ {
			for n := range m {
				c := prof[k].cap[n]
				m[n] = placement.Cap{GPUs: min(m[n].GPUs, c.GPUs), CPUs: min(m[n].CPUs, c.CPUs), MemMB: min(m[n].MemMB, c.MemMB)}
			}
		}
		return m
	}
	for depth, i := range order {
		if depth >= p.c.depth {
			break
		}
		if !p.admit(i) {
			continue
		}
		pj := &p.v.Pending[i]
		r := placement.RequestOf(pj)
		for k := 0; k < len(prof); k++ {
			pl, ok := p.c.placer.Place(r, p.v.Nodes, prof[k].cap)
			if !ok {
				continue
			}
			end := prof[k].t + p.c.estimatedDuration(p.v, pj, pl)
			m := minCap(k, end)
			if !placement.Contains(m, r, pl) {
				// retry once on the capacity that is free for the whole interval
				if pl, ok = p.c.placer.Place(r, p.v.Nodes, m); !ok {
					continue
				}
				end = prof[k].t + p.c.estimatedDuration(p.v, pj, pl)
				if m = minCap(k, end); !placement.Contains(m, r, pl) {
					continue
				}
			}
			// reserve [prof[k].t, end): split at end, subtract from the covered segments
			j := k
			for j < len(prof) && prof[j].t < end {
				j++
			}
			if j == len(prof) || prof[j].t != end {
				prev := prof[j-1].cap
				prof = slices.Insert(prof, j, segment{t: end, cap: slices.Clone(prev)})
			}
			for q := k; q < j; q++ {
				placement.Take(prof[q].cap, r, pl)
			}
			if k == 0 {
				p.start(i, pl)
			} else {
				p.dec.Reservations = append(p.dec.Reservations, api.Reservation{JobID: pj.JobID, Shadow: prof[k].t})
			}
			break
		}
	}
}

// preemptPass lets each job that was not started and does not fit preempt
// lower-priority preemptible jobs (victims minimizing lost work).
func (p *pass) preemptPass(order []int) {
	exclude := map[string]bool{}
	for _, i := range order {
		if p.started[i] || !p.admit(i) {
			continue
		}
		pj := &p.v.Pending[i]
		r := placement.RequestOf(pj)
		if _, ok := p.c.placer.Place(r, p.v.Nodes, p.free); ok {
			continue // it fits but the backfill rules held it back
		}
		victims, pl, ok := preemption.Select(r, pj.Priority, p.v.Running, exclude, p.v.Nodes, p.free, p.c.placer, p.c.pp)
		if !ok {
			continue
		}
		for _, vi := range victims {
			rj := &p.v.Running[vi]
			exclude[rj.JobID] = true
			preemption.Release(p.free, rj)
			p.dec.Actions = append(p.dec.Actions, api.Action{Op: api.OpPreempt, JobID: rj.JobID})
		}
		p.start(i, pl)
	}
}

// predictor is the rule of Tsafrir, Etsion, and Feitelson: the predicted run
// time of a job is the mean of the last k completed run times of the same
// user, never above the user estimate, and the estimate without history.
type predictor struct {
	k    int
	last map[string][]float64
	seen int
}

func (p *predictor) update(h []api.CompletedJob) {
	for ; p.seen < len(h); p.seen++ {
		c := h[p.seen]
		l := append(p.last[c.User], clock.Sec(c.Runtime))
		if len(l) > p.k {
			l = l[len(l)-p.k:]
		}
		p.last[c.User] = l
	}
}

func (p *predictor) predict(user string, estimate float64) float64 {
	l := p.last[user]
	if len(l) == 0 {
		return estimate
	}
	s := 0.0
	for _, x := range l {
		s += x
	}
	return min(s/float64(len(l)), estimate)
}

// sizeObserver keeps the per-worker GPU-size distribution of the jobs the
// policy has seen (for least_fragmentation).
type sizeObserver struct {
	seen   map[string]bool
	counts map[int]float64
	dist   metrics.SizeDist
	dirty  bool
}

func (s *sizeObserver) observe(v *api.View) {
	if s.seen == nil {
		s.seen, s.counts = map[string]bool{}, map[int]float64{}
	}
	for i := range v.Pending {
		pj := &v.Pending[i]
		if !s.seen[pj.JobID] {
			s.seen[pj.JobID] = true
			s.counts[pj.GPUs] += float64(pj.Workers)
			s.dirty = true
		}
	}
}

func (s *sizeObserver) get() *metrics.SizeDist {
	if s.dirty {
		var d metrics.SizeDist
		tot := 0.0
		for g, c := range s.counts {
			d.Sizes = append(d.Sizes, g)
			tot += c
		}
		slices.Sort(d.Sizes)
		for _, g := range d.Sizes {
			d.Probs = append(d.Probs, s.counts[g]/tot)
		}
		s.dist, s.dirty = d, false
	}
	return &s.dist
}

// newComposite builds a composed policy from "order+placement+backfill"
// with optional suffixes "+preempt", "+quota" or "+quota_strict", "+reclaim".
func newComposite(name string, prm Params, oracle func(string) (time.Duration, bool)) (*Composite, error) {
	parts := splitName(name)
	c := &Composite{name: name}
	if len(parts) < 3 || !parseSuffixes(c, parts[3:]) {
		return nil, fmt.Errorf("policy name %q: want order+placement+backfill[+preempt][+quota|+quota_strict][+reclaim]", name)
	}
	c.backfill = parts[2]
	if c.quota {
		if len(prm.QuotaShares) == 0 {
			return nil, fmt.Errorf("policy %q: quotas need quota_shares", name)
		}
		sum := 0.0
		for t, sh := range prm.QuotaShares {
			if t == "" || !(sh >= 0) {
				return nil, fmt.Errorf("policy %q: bad quota share %q=%v", name, t, sh)
			}
			sum += sh
		}
		if sum > 1+1e-9 {
			return nil, fmt.Errorf("policy %q: quota shares sum to %v > 1", name, sum)
		}
		c.shares = prm.QuotaShares
	}
	var err error
	if c.order, err = queue.New(parts[0], queue.Params{AgingPerHour: prm.AgingPerHour}); err != nil {
		return nil, err
	}
	if c.placer, err = placement.New(parts[1], c.sizes.get); err != nil {
		return nil, err
	}
	switch c.backfill {
	case BackfillNone, BackfillEASY:
	case BackfillConservative:
		c.depth = prm.Depth
		if c.depth == 0 {
			c.depth = 64
		}
		if c.depth < 1 {
			return nil, errors.New("depth must be >= 1")
		}
	case BackfillEASYPredicted:
		k := prm.History
		if k == 0 {
			k = 2
		}
		if k < 1 {
			return nil, errors.New("history must be >= 1")
		}
		c.pred = &predictor{k: k, last: map[string][]float64{}}
	case BackfillEASYOracle:
		if oracle == nil {
			return nil, errors.New("easy_oracle needs the true run times (an oracle runs only in the benchmark)")
		}
		c.oracle = oracle
	case BackfillRisk:
		q, mh := prm.Quantile, prm.MinHistory
		if q == 0 {
			q = 0.9
		}
		if mh == 0 {
			mh = 5
		}
		if !(q > 0 && q <= 1) || mh < 1 {
			return nil, errors.New("risk backfilling needs 0 < quantile <= 1 and min_history >= 1")
		}
		c.risk = &ratioModel{q: q, minHistory: mh, user: map[string][]float64{}}
	default:
		return nil, fmt.Errorf("unknown backfill %q", c.backfill)
	}
	if c.preempt {
		if c.pp, err = (preemption.Params{MinGap: prm.MinGap, MaxVictims: prm.MaxVictims}).Defaults(); err != nil {
			return nil, err
		}
	}
	return c, nil
}

func splitName(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '+' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}
