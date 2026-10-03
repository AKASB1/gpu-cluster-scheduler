package simulator

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
)

// queueTrace builds n single-GPU jobs with Poisson arrivals (rate lambda)
// and service times from svc, rounded to whole milliseconds.
func queueTrace(seed uint64, n int, lambda float64, svc func(*rand.Rand) float64) []api.Job {
	ra, rs := rng.New(seed, "arrivals"), rng.New(seed, "service")
	jobs := make([]api.Job, n)
	t := 0.0
	for i := range jobs {
		t += ra.ExpFloat64() / lambda
		rt := max(clock.RoundMs(svc(rs)), time.Millisecond)
		jobs[i] = api.Job{ID: fmt.Sprintf("j%07d", i), Submit: clock.RoundMs(t), Tenant: "t", User: "u", Priority: 1,
			GPUs: 1, Workers: 1, Topology: api.TopoAny, Runtime: rt, Estimate: rt}
	}
	return jobs
}

func meanWait(t *testing.T, gpus int, jobs []api.Job) float64 {
	rt, err := Run(Config{Cluster: mustCluster(t, oneNode(gpus)), Jobs: jobs, Policy: testFIFO{blocking: true}})
	if err != nil {
		t.Fatal(err)
	}
	sum := 0.0
	for i := range rt.Jobs {
		sum += clock.Sec(rt.Jobs[i].FirstStart - rt.Jobs[i].Job.Submit)
	}
	return sum / float64(len(rt.Jobs))
}

// Check 1 (exact): one node with one GPU, FIFO; the simulated waits equal the
// Lindley recursion W[n+1] = max(0, W[n] + S[n] - A[n+1]) on the realised,
// millisecond-rounded inter-arrival and service times. Seeds 1..3.
func TestLindleyRecursionExact(t *testing.T) {
	for seed := uint64(1); seed <= 3; seed++ {
		jobs := queueTrace(seed, 20000, 1.0/100, func(r *rand.Rand) float64 { return 90 * math.Exp(0.8*r.NormFloat64()-0.32) })
		rt, err := Run(Config{Cluster: mustCluster(t, oneNode(1)), Jobs: jobs, Policy: testFIFO{blocking: true}})
		if err != nil {
			t.Fatal(err)
		}
		var w time.Duration
		for n := range jobs {
			if n > 0 {
				w = max(0, w+jobs[n-1].Runtime-(jobs[n].Submit-jobs[n-1].Submit))
			}
			if got := rt.Jobs[n].FirstStart - jobs[n].Submit; got != w {
				t.Fatalf("seed %d job %d: simulated wait %v, Lindley %v", seed, n, got, w)
			}
		}
	}
}

// Check 2 (statistical): M/G/1 at rho = 0.7 with exponential, deterministic,
// and lognormal service (mean 100 s). The mean wait over 10 seeds x 50,000
// jobs matches Pollaczek-Khinchine lambda*E[S^2]/(2(1-rho)) within the stated
// relative tolerance (fixed before the first run; the standard error of the
// 10-seed mean is logged next to the error).
func TestMG1PollaczekKhinchine(t *testing.T) {
	if testing.Short() {
		t.Skip("long statistical test")
	}
	const mean, rho = 100.0, 0.7
	lambda := rho / mean
	sigma := 1.0 // lognormal shape; E[S^2] = mean^2 * exp(sigma^2)
	cases := []struct {
		name string
		svc  func(*rand.Rand) float64
		es2  float64
		tol  float64
	}{
		{"exponential", func(r *rand.Rand) float64 { return mean * r.ExpFloat64() }, 2 * mean * mean, 0.05},
		{"deterministic", func(*rand.Rand) float64 { return mean }, mean * mean, 0.05},
		{"lognormal", func(r *rand.Rand) float64 { return mean * math.Exp(sigma*r.NormFloat64()-sigma*sigma/2) }, mean * mean * math.Exp(sigma*sigma), 0.08},
	}
	for _, c := range cases {
		want := lambda * c.es2 / (2 * (1 - rho))
		var per []float64
		for seed := uint64(1); seed <= 10; seed++ {
			per = append(per, meanWait(t, 1, queueTrace(seed, 50000, lambda, c.svc)))
		}
		got, se := meanSE(per)
		t.Logf("M/G/1 %s: simulated %.2f s (SE %.2f), P-K %.2f s, rel. error %+.3f", c.name, got, se, want, got/want-1)
		if math.Abs(got/want-1) > c.tol {
			t.Errorf("M/G/1 %s: simulated %.2f s, P-K %.2f s (tolerance %.0f%%)", c.name, got, want, 100*c.tol)
		}
	}
}

// meanSE returns the mean of x and its standard error.
func meanSE(x []float64) (float64, float64) {
	m := 0.0
	for _, v := range x {
		m += v
	}
	m /= float64(len(x))
	ss := 0.0
	for _, v := range x {
		ss += (v - m) * (v - m)
	}
	return m, math.Sqrt(ss / float64(len(x)-1) / float64(len(x)))
}

// erlangC is the mean wait of M/M/c with arrival rate lambda and service rate mu.
func erlangC(c int, lambda, mu float64) float64 {
	a := lambda / mu
	rho := a / float64(c)
	term, sum := 1.0, 0.0
	for k := 0; k < c; k++ {
		sum += term
		term *= a / float64(k+1)
	}
	top := term / (1 - rho) // a^c / c! / (1 - rho)
	pw := top / (sum + top)
	return pw / (float64(c)*mu - lambda)
}

// Check 3 (statistical): one node with c = 4 and c = 16 GPUs, single-GPU
// jobs, exponential service (mean 100 s), FIFO, rho = 0.7. The mean wait over
// 10 seeds matches Erlang C within the stated relative tolerance (fixed
// before the first run; the standard error is logged).
func TestMMcErlangC(t *testing.T) {
	if testing.Short() {
		t.Skip("long statistical test")
	}
	const mean, rho = 100.0, 0.7
	for _, k := range []struct {
		c   int
		n   int
		tol float64
	}{{4, 50000, 0.06}, {16, 100000, 0.10}} {
		lambda := rho * float64(k.c) / mean
		want := erlangC(k.c, lambda, 1/mean)
		var per []float64
		for seed := uint64(1); seed <= 10; seed++ {
			per = append(per, meanWait(t, k.c, queueTrace(seed, k.n, lambda, func(r *rand.Rand) float64 { return mean * r.ExpFloat64() })))
		}
		got, se := meanSE(per)
		t.Logf("M/M/%d: simulated %.3f s (SE %.3f), Erlang C %.3f s, rel. error %+.3f", k.c, got, se, want, got/want-1)
		if math.Abs(got/want-1) > k.tol {
			t.Errorf("M/M/%d: simulated %.3f s, Erlang C %.3f s (tolerance %.0f%%)", k.c, got, want, 100*k.tol)
		}
	}
}
