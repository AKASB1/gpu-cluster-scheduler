package metrics

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strconv"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// Tau is the bounded-slowdown threshold in seconds.
const Tau = 10.0

// WindowFraction is the share of jobs dropped at each end (by submit order).
const WindowFraction = 0.10

// LargeGPUs is the gang size from which a job counts as large (S3).
const LargeGPUs = 8

// Field is one named value of a result row. NaN means "not defined" and is
// written as an empty CSV field.
type Field struct {
	Name  string
	Value float64
}

// Run is the metric set of one run, as an ordered list of fields.
type Run struct {
	Fields []Field
	index  map[string]int
}

// Get returns a field value (NaN if absent).
func (r *Run) Get(name string) float64 {
	if i, ok := r.index[name]; ok {
		return r.Fields[i].Value
	}
	return math.NaN()
}

func (r *Run) add(name string, v float64) {
	if r.index == nil {
		r.index = map[string]int{}
	}
	if _, dup := r.index[name]; dup {
		panic("metrics: duplicate field " + name)
	}
	r.index[name] = len(r.Fields)
	r.Fields = append(r.Fields, Field{name, v})
}

// Percentile is the nearest-rank percentile: the ceil(p*n)-th smallest value
// of x (x need not be sorted; it is not modified). NaN for an empty sample.
func Percentile(x []float64, p float64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	s := slices.Clone(x)
	slices.Sort(s)
	k := int(math.Ceil(p*float64(len(s)))) - 1
	return s[min(max(k, 0), len(s)-1)]
}

// Mean of x (NaN when empty).
func Mean(x []float64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s / float64(len(x))
}

// Jain is Jain's fairness index (sum x)^2 / (n sum x^2) (NaN when empty).
func Jain(x []float64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	s, s2 := 0.0, 0.0
	for _, v := range x {
		s += v
		s2 += v * v
	}
	if s2 == 0 {
		return 1
	}
	return s * s / (float64(len(x)) * s2)
}

// Window returns the indices (into rt.Jobs) of the measurement window: the
// feasible jobs in trace order without the first and the last
// floor(n/10) jobs.
func Window(rt *RunTrace) []int {
	var feas []int
	for i := range rt.Jobs {
		if !rt.Jobs[i].Infeasible {
			feas = append(feas, i)
		}
	}
	k := int(float64(len(feas)) * WindowFraction)
	return feas[k : len(feas)-k]
}

// JobTimes are the per-job time metrics in seconds.
type JobTimes struct {
	Wait, JCT, Queue, Ideal, BSld, WFlow float64
}

// TimesOf computes the per-job metrics of one record.
func TimesOf(rt *RunTrace, r *JobRecord) JobTimes {
	j := &r.Job
	speed := rt.FastestAny
	if j.GPUClass != "" {
		speed = rt.ClassSpeed[j.GPUClass]
	}
	t := JobTimes{Wait: clock.Sec(r.FirstStart - j.Submit), JCT: clock.Sec(r.Completion - j.Submit),
		Queue: clock.Sec(r.QueueTotal), Ideal: clock.Sec(j.Runtime) / speed}
	d := math.Max(t.Ideal, Tau)
	t.WFlow = t.JCT / d
	t.BSld = math.Max(1, t.WFlow)
	return t
}

// span returns the window's time span: from the submit time of the first to
// that of the last window job; when that is empty (batch scenarios), from the
// first submission to the last completion.
func span(rt *RunTrace, win []int) (time.Duration, time.Duration) {
	if len(win) == 0 {
		return 0, 0
	}
	a, b := rt.Jobs[win[0]].Job.Submit, rt.Jobs[win[len(win)-1]].Job.Submit
	if b <= a {
		b = rt.End
	}
	return a, b
}

// integrate applies f to every sample interval clipped to [a, b] and returns
// the sum of f(sample) * overlap seconds.
func integrate(rt *RunTrace, a, b time.Duration, f func(*Sample) float64) float64 {
	sum := 0.0
	for i := range rt.Samples {
		s := &rt.Samples[i]
		end := rt.End
		if i+1 < len(rt.Samples) {
			end = rt.Samples[i+1].T
		}
		lo, hi := max(s.T, a), min(end, b)
		if hi > lo {
			sum += f(s) * clock.Sec(hi-lo)
		}
	}
	return sum
}

type stat struct{ mean, p50, p95, p99 float64 }

func stats(x []float64) stat {
	return stat{Mean(x), Percentile(x, 0.5), Percentile(x, 0.95), Percentile(x, 0.99)}
}

// Compute returns the metrics of one run (docs/contracts.md §5).
func Compute(rt *RunTrace) *Run {
	r := &Run{}
	win := Window(rt)
	feasible, infeasible := 0, 0
	for i := range rt.Jobs {
		if rt.Jobs[i].Infeasible {
			infeasible++
		} else {
			feasible++
		}
	}
	r.add("jobs", float64(feasible))
	r.add("window_jobs", float64(len(win)))
	r.add("infeasible_jobs", float64(infeasible))

	var wait, jct, bsld, wflow, largeWait []float64
	tenants := map[string][]JobTimes{}
	prios := map[int][]JobTimes{}
	var useful, consumed, lost, overhead, topo, slack, grace float64
	preempt, slo, sloN, resv, broken := 0, 0, 0, 0, 0
	for _, i := range win {
		rec := &rt.Jobs[i]
		t := TimesOf(rt, rec)
		wait, jct, bsld, wflow = append(wait, t.Wait), append(jct, t.JCT), append(bsld, t.BSld), append(wflow, t.WFlow)
		if rec.Job.TotalGPUs() >= LargeGPUs {
			largeWait = append(largeWait, t.Wait)
		}
		tenants[rec.Job.Tenant] = append(tenants[rec.Job.Tenant], t)
		prios[rec.Job.Priority] = append(prios[rec.Job.Priority], t)
		useful += rec.UsefulW
		consumed += rec.ConsumedW
		lost += rec.LostW
		overhead += rec.OverheadW
		topo += rec.TopoW
		slack += rec.SlackW + rec.RoundingW
		grace += rec.GraceW
		preempt += rec.Preemptions
		if rec.Job.HasMaxWait {
			sloN++
			if rec.FirstStart-rec.Job.Submit <= rec.Job.MaxWait {
				slo++
			}
		}
		if rec.HasShadow {
			resv++
			// the first start at or after the report (a job may report a
			// shadow time only after it was preempted)
			for _, sg := range rec.Segments {
				if sg.Start >= rec.ShadowAt {
					if sg.Start > rec.Shadow {
						broken++
					}
					break
				}
			}
		}
	}
	for _, m := range []struct {
		name string
		x    []float64
	}{{"wait", wait}, {"jct", jct}, {"bsld", bsld}} {
		s := stats(m.x)
		r.add(m.name+"_mean", s.mean)
		r.add(m.name+"_p50", s.p50)
		r.add(m.name+"_p95", s.p95)
		r.add(m.name+"_p99", s.p99)
	}
	r.add("wflow_mean", Mean(wflow))
	all := 0.0
	for i := range rt.Jobs {
		if !rt.Jobs[i].Infeasible {
			all += TimesOf(rt, &rt.Jobs[i]).WFlow
		}
	}
	r.add("wflow_sum_all", all)
	r.add("large_jobs", float64(len(largeWait)))
	r.add("large_wait_mean", Mean(largeWait))
	r.add("large_wait_p95", Percentile(largeWait, 0.95))

	a, b := span(rt, win)
	if b > a {
		capacity := float64(rt.Info.TotalGPUs) * clock.Sec(b-a)
		r.add("utilization", integrate(rt, a, b, func(s *Sample) float64 { return float64(s.Alloc) })/capacity)
		r.add("frag_blocked_frac", integrate(rt, a, b, func(s *Sample) float64 { return b2f(s.Blocked) })/clock.Sec(b-a))
		free := integrate(rt, a, b, func(s *Sample) float64 { return float64(s.Free) })
		str := integrate(rt, a, b, func(s *Sample) float64 { return s.Stranded })
		r.add("stranded_frac", ratio(str, free))
	} else {
		r.add("utilization", math.NaN())
		r.add("frag_blocked_frac", math.NaN())
		r.add("stranded_frac", math.NaN())
	}
	r.add("goodput_ratio", ratio(useful, consumed))
	r.add("wasted_gpu_hours", (consumed-useful)/3600)
	r.add("lost_gpu_hours", lost/3600)
	r.add("overhead_gpu_hours", overhead/3600)
	r.add("topology_gpu_hours", topo/3600)
	r.add("slack_gpu_hours", slack/3600)
	r.add("grace_gpu_hours", grace/3600)
	r.add("preemptions", float64(preempt))
	kills := 0
	for _, i := range win {
		kills += rt.Jobs[i].FailureKills
	}
	r.add("failure_kills", float64(kills))
	down := 0.0
	for _, d := range rt.Downtime {
		if lo, hi := max(d.From, a), min(d.To, b); hi > lo {
			down += float64(d.GPUs) * clock.Sec(hi-lo)
		}
	}
	r.add("node_downtime_gpu_hours", down/3600)

	// fairness over per-tenant mean bounded slowdown
	tnames := sortedKeys(tenants)
	var tmeans []float64
	for _, name := range tnames {
		tmeans = append(tmeans, Mean(field(tenants[name], func(t JobTimes) float64 { return t.BSld })))
	}
	r.add("jain_bsld", Jain(tmeans))
	if len(tmeans) > 0 {
		r.add("tenant_worst_best_ratio", slices.Max(tmeans)/slices.Min(tmeans))
	} else {
		r.add("tenant_worst_best_ratio", math.NaN())
	}
	if sloN > 0 {
		r.add("slo_attainment", float64(slo)/float64(sloN))
		r.add("slo_violation_rate", 1-float64(slo)/float64(sloN))
	} else {
		r.add("slo_attainment", math.NaN())
		r.add("slo_violation_rate", math.NaN())
	}
	r.add("slo_jobs", float64(sloN))
	first, last := time.Duration(math.MaxInt64), time.Duration(0)
	for i := range rt.Jobs {
		if !rt.Jobs[i].Infeasible {
			first = min(first, rt.Jobs[i].Job.Submit)
			last = max(last, rt.Jobs[i].Completion)
		}
	}
	if feasible > 0 {
		r.add("makespan", clock.Sec(last-first))
	} else {
		r.add("makespan", math.NaN())
	}
	r.add("reservations", float64(resv))
	r.add("broken_reservations", float64(broken))
	r.add("invocations", float64(rt.Invocations))
	for _, name := range tnames {
		ts := tenants[name]
		r.add("tenant_"+name+"_jobs", float64(len(ts)))
		r.addDist("tenant_"+name+"_", ts)
	}
	pnames := make([]int, 0, len(prios))
	for p := range prios {
		pnames = append(pnames, p)
	}
	slices.Sort(pnames)
	for _, p := range pnames {
		ts := prios[p]
		pre := "prio_" + strconv.Itoa(p) + "_"
		r.add(pre+"jobs", float64(len(ts)))
		r.addDist(pre, ts)
	}
	return r
}

// addDist adds wait, JCT, and bounded slowdown (mean, P50, P95, P99) of a
// group of window jobs, with the given column prefix.
func (r *Run) addDist(prefix string, ts []JobTimes) {
	for _, m := range []struct {
		name string
		f    func(JobTimes) float64
	}{{"wait", func(t JobTimes) float64 { return t.Wait }}, {"jct", func(t JobTimes) float64 { return t.JCT }},
		{"bsld", func(t JobTimes) float64 { return t.BSld }}} {
		s := stats(field(ts, m.f))
		r.add(prefix+m.name+"_mean", s.mean)
		r.add(prefix+m.name+"_p50", s.p50)
		r.add(prefix+m.name+"_p95", s.p95)
		r.add(prefix+m.name+"_p99", s.p99)
	}
}

// Wall returns the non-deterministic wall-clock fields of a run.
func Wall(rt *RunTrace) *Run {
	r := &Run{}
	r.add("wall_decide_mean_s", Mean(rt.WallDecide))
	r.add("wall_decide_p95_s", Percentile(rt.WallDecide, 0.95))
	if len(rt.WallDecide) > 0 {
		r.add("wall_decide_max_s", slices.Max(rt.WallDecide))
	} else {
		r.add("wall_decide_max_s", math.NaN())
	}
	if s := rt.Solver; s != nil {
		r.add("wall_solver_calls", float64(s.Calls))
		r.add("wall_solve_mean_s", Mean(s.WallSolve))
		r.add("wall_solve_p95_s", Percentile(s.WallSolve, 0.95))
		r.add("wall_solve_max_s", slices.Max(s.WallSolve))
		r.add("wall_solve_cpu_mean_s", Mean(s.CPUSolve))
		r.add("wall_solve_cpu_p95_s", Percentile(s.CPUSolve, 0.95))
		r.add("wall_solve_cpu_max_s", slices.Max(s.CPUSolve))
		r.add("wall_solver_capped", float64(s.Capped))
		r.add("wall_solver_not_optimal", float64(s.NotOptimal))
		r.add("wall_solver_max_gap", s.MaxGap)
	}
	return r
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func ratio(a, b float64) float64 {
	if b == 0 {
		return math.NaN()
	}
	return a / b
}

func field(ts []JobTimes, f func(JobTimes) float64) []float64 {
	out := make([]float64, len(ts))
	for i, t := range ts {
		out[i] = f(t)
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, cmp.Compare[string])
	return keys
}

// Format writes a value for CSV: empty for NaN, otherwise the shortest
// decimal with 10 significant digits (deterministic).
func Format(v float64) string {
	if math.IsNaN(v) {
		return ""
	}
	if math.IsInf(v, 0) {
		return fmt.Sprint(v)
	}
	return strconv.FormatFloat(v, 'g', 10, 64)
}
