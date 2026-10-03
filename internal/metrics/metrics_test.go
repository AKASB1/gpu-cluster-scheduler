package metrics

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

func TestPercentileNearestRank(t *testing.T) {
	x := []float64{5, 1, 4, 2, 3}
	for p, want := range map[float64]float64{0.5: 3, 0.95: 5, 0.2: 1, 0.21: 2, 0.99: 5, 0: 1} {
		if got := Percentile(x, p); got != want {
			t.Errorf("P%v = %v, want %v", p, got, want)
		}
	}
	if x[0] != 5 {
		t.Fatal("input modified")
	}
	if !math.IsNaN(Percentile(nil, 0.5)) {
		t.Fatal("empty sample")
	}
}

func TestJain(t *testing.T) {
	if Jain([]float64{2, 2, 2}) != 1 || !near(Jain([]float64{1, 3}), 0.8) {
		t.Fatal("jain")
	}
}

func TestStudentT(t *testing.T) {
	// Reference quantiles of Student's t (two-sided 95%).
	for df, want := range map[float64]float64{1: 12.7062, 4: 2.7764, 9: 2.2622, 19: 2.0930, 100: 1.9840} {
		if got := TQuantile(0.975, df); math.Abs(got-want) > 1e-3 {
			t.Errorf("t(0.975, %v) = %.4f, want %.4f", df, got, want)
		}
	}
	c := MeanCI([]float64{1, 2, 3, 4, math.NaN()})
	// mean 2.5, sd 1.29099, half = 3.1824 * 1.29099 / 2 = 2.0542
	if c.N != 4 || c.Mean != 2.5 || math.Abs(c.Half-2.0542) > 1e-3 {
		t.Fatalf("CI %+v", c)
	}
	w, ti, l := WTL([]float64{1, 2, 3, 4}, []float64{2, 2, 1, 5}, true)
	if w != 2 || ti != 1 || l != 1 {
		t.Fatalf("WTL %d %d %d", w, ti, l)
	}
}

func TestSizeDistAndStranded(t *testing.T) {
	jobs := []api.Job{{GPUs: 1, Workers: 1}, {GPUs: 2, Workers: 1}, {GPUs: 4, Workers: 2}}
	d := SizeDistOf(jobs)
	// p(1) = 0.25, p(2) = 0.25, p(4) = 0.5 (weighted by workers)
	if fmt.Sprint(d.Sizes) != "[1 2 4]" || !near(d.Probs[2], 0.5) {
		t.Fatalf("dist %+v", d)
	}
	for free, want := range map[int]float64{3: 0.25*1 + 0.5*3, 8: 0, 6: 0.5 * 2, 1: 0.25 + 0.5} {
		if got := d.Stranded(free); !near(got, want) {
			t.Errorf("stranded(%d) = %v, want %v", free, got, want)
		}
	}
}

func TestBlockedClassification(t *testing.T) {
	nodes := []api.NodeState{{Name: "a", Class: "x", FreeGPUs: 4, FreeCPUs: 64, FreeMemMB: 1000},
		{Name: "b", Class: "x", FreeGPUs: 4, FreeCPUs: 8, FreeMemMB: 1000}}
	cases := []struct {
		s         Shape
		fit, ok   bool
		blockedBy bool
	}{
		{Shape{GPUs: 8, Workers: 1}, true, false, true},                      // totals fit, no node has 8
		{Shape{GPUs: 4, Workers: 2}, true, true, false},                      // one per node
		{Shape{GPUs: 2, Workers: 4, CPUs: 8}, true, false, true},             // node b only fits 1 worker by CPU
		{Shape{GPUs: 1, Workers: 9}, false, false, false},                    // totals do not fit: not blocked
		{Shape{GPUs: 1, Workers: 1, Class: "y"}, false, false, false},        // no eligible node
		{Shape{GPUs: 1, Workers: 2, MemMB: 600}, true, true, false},          // one per node by memory
		{Shape{GPUs: 1, Workers: 3, MemMB: 600}, true, false, true},          // 1800 <= 2000 MB in total, but 1 per node by memory
		{Shape{GPUs: 2, Workers: 3, MemMB: 400, CPUs: 4}, true, true, false}, // a: 2, b: 2 by CPU -> 3 fit
	}
	for i, c := range cases {
		fit, ok := Classify(c.s, nodes)
		if fit != c.fit || ok != c.ok || Blocked([]Shape{c.s}, nodes) != c.blockedBy {
			t.Errorf("case %d: fit %v ok %v", i, fit, ok)
		}
	}
}

// A hand-made run: 10 jobs (window drops 1 at each end), cluster of 8 GPUs.
func handRun() *RunTrace {
	rt := &RunTrace{Info: api.ClusterInfo{TotalGPUs: 8}, FastestAny: 2, ClassSpeed: map[string]float64{"fast": 2, "slow": 1}}
	s := func(x float64) time.Duration { return time.Duration(x * float64(time.Second)) }
	for i := 0; i < 10; i++ {
		j := api.Job{ID: fmt.Sprintf("j%d", i), Submit: s(float64(i)), Tenant: "a", Priority: 1, GPUs: 1, Workers: 1, Runtime: s(100)}
		rt.Jobs = append(rt.Jobs, JobRecord{Job: j, FirstStart: j.Submit, Completion: j.Submit + s(50), UsefulW: 100, ConsumedW: 100})
	}
	// job 1: tenant b, runtime 5 s (ideal 2.5 < tau), waited 20 s, JCT 30 -> bsld 3
	rt.Jobs[1].Job.Tenant, rt.Jobs[1].Job.Runtime = "b", s(5)
	rt.Jobs[1].FirstStart, rt.Jobs[1].Completion = s(21), s(31)
	// job 2: class slow (ideal 100 / 1 = 100), JCT 50 -> bsld 1, wflow 0.5;
	// wasted 50 GPU-s (lost to preemption), max_wait 10 s violated (wait 20)
	rt.Jobs[2].Job.GPUClass = "slow"
	rt.Jobs[2].FirstStart, rt.Jobs[2].ConsumedW, rt.Jobs[2].LostW = s(22), 150, 50
	rt.Jobs[2].Job.HasMaxWait, rt.Jobs[2].Job.MaxWait = true, s(10)
	rt.Jobs[3].Job.HasMaxWait, rt.Jobs[3].Job.MaxWait = true, s(10) // met (wait 0)
	// job 0 is outside the window: a huge value must not count
	rt.Jobs[0].Completion = s(10000)
	rt.End = s(10000)
	// samples: [0,4): alloc 4, free 4, blocked; [4,8): alloc 8, free 0; [8,End): alloc 2, free 6
	rt.Samples = []Sample{{T: 0, Alloc: 4, Free: 4, Stranded: 1, Blocked: true}, {T: s(4), Alloc: 8, Free: 0}, {T: s(8), Alloc: 2, Free: 6, Stranded: 3}}
	return rt
}

func TestComputeHandMadeRun(t *testing.T) {
	rt := handRun()
	r := Compute(rt)
	// window: jobs 1..8, span [1 s, 8 s]
	// utilization: alloc 4 on [1,4) = 12, 8 on [4,8) = 32 -> 44 / (8 * 7)
	want := map[string]float64{
		"window_jobs":       8,
		"utilization":       44.0 / 56,
		"frag_blocked_frac": 3.0 / 7,
		// stranded / free over the span: (1*3 + 0*4) / (4*3 + 0*4) = 3/12
		"stranded_frac":      0.25,
		"goodput_ratio":      800.0 / 850,
		"lost_gpu_hours":     50.0 / 3600,
		"slo_attainment":     0.5,
		"slo_jobs":           2,
		"wait_p50":           0,
		"wait_p95":           21 - 1,
		"tenant_b_jobs":      1,
		"tenant_a_jobs":      7,
		"tenant_b_bsld_mean": 3,
		// tenant a: jobs 3..8 have JCT 50, ideal 50 -> bsld 1; job 2 -> 1 => mean 1
		"tenant_a_bsld_mean":      1,
		"jain_bsld":               16.0 / (2 * 10),
		"tenant_worst_best_ratio": 3,
		"makespan":                10000,
	}
	for k, v := range want {
		if got := r.Get(k); !near(got, v) {
			t.Errorf("%s = %v, want %v", k, got, v)
		}
	}
	// wflow: job 2 has JCT 50 / max(100, 10) = 0.5; bounded slowdown clamps to 1
	if got := TimesOf(rt, &rt.Jobs[2]); !near(got.WFlow, 0.5) || got.BSld != 1 {
		t.Fatalf("job 2 times %+v", got)
	}
	// job 1: JCT 30 / max(2.5, 10) = 3
	if got := TimesOf(rt, &rt.Jobs[1]); !near(got.BSld, 3) || !near(got.Ideal, 2.5) {
		t.Fatalf("job 1 times %+v", got)
	}
}

func TestWindowSkipsInfeasibleAndDropsTenPercent(t *testing.T) {
	rt := &RunTrace{}
	for i := 0; i < 23; i++ {
		rt.Jobs = append(rt.Jobs, JobRecord{Infeasible: i == 5})
	}
	w := Window(rt) // 22 feasible -> drop 2 at each end
	if len(w) != 18 || w[0] != 2 || w[len(w)-1] != 20 {
		t.Fatalf("window %v", w)
	}
	for _, i := range w {
		if i == 5 {
			t.Fatal("infeasible job in window")
		}
	}
}
