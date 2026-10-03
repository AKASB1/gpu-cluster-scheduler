package trace

import (
	"bytes"
	"encoding/json"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"math"
	"os"
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

func baseConfig(t *testing.T) GenConfig {
	t.Helper()
	data, err := os.ReadFile("../../configs/workloads/base.json")
	if err != nil {
		t.Fatal(err)
	}
	var c GenConfig
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatal(err)
	}
	c.WorkCapacity = 128
	return c
}

func gen(t *testing.T, c GenConfig, seed uint64) []api.Job {
	t.Helper()
	jobs, err := Generate(c, seed)
	if err != nil {
		t.Fatal(err)
	}
	return jobs
}

func interarrivals(jobs []api.Job) []float64 {
	out := make([]float64, 0, len(jobs))
	for i := 1; i < len(jobs); i++ {
		out = append(out, clock.Sec(jobs[i].Submit-jobs[i-1].Submit))
	}
	return out
}

func meanSD(x []float64) (float64, float64) {
	m := 0.0
	for _, v := range x {
		m += v
	}
	m /= float64(len(x))
	s := 0.0
	for _, v := range x {
		s += (v - m) * (v - m)
	}
	return m, math.Sqrt(s / float64(len(x)-1))
}

func TestGenerateDeterministicAndSeedSensitive(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 500
	a, b, d := Encode(gen(t, c, 7)), Encode(gen(t, c, 7)), Encode(gen(t, c, 8))
	if !bytes.Equal(a, b) {
		t.Fatal("same seed gave different traces")
	}
	if bytes.Equal(a, d) {
		t.Fatal("different seeds gave the same trace")
	}
	if _, err := Parse(a); err != nil {
		t.Fatalf("generated trace does not validate: %v", err)
	}
}

// Tolerances: rates and shares are checked at about 4 standard errors on
// fixed seeds, so the tests are deterministic and not fragile.
func TestPoissonRateAndCV(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 20000
	jobs := gen(t, c, 11)
	rate := c.ArrivalRate(11)
	m, sd := meanSD(interarrivals(jobs))
	if math.Abs(m*rate-1) > 0.03 {
		t.Fatalf("mean interarrival %.2f s, target %.2f s", m, 1/rate)
	}
	if cv := sd / m; math.Abs(cv-1) > 0.04 {
		t.Fatalf("poisson CV %.3f", cv)
	}
}

func TestBurstyProcessesHaveCVAboveOne(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 50000
	c.Arrival = ArrivalConfig{Process: "gamma", Load: 0.8, CV: 3}
	rate := c.ArrivalRate(12)
	m, sd := meanSD(interarrivals(gen(t, c, 12)))
	if cv := sd / m; math.Abs(cv-3) > 0.3 || math.Abs(m*rate-1) > 0.05 {
		t.Fatalf("gamma: CV %.3f (target 3), mean*rate %.3f", cv, m*rate)
	}
	c.Jobs = 20000
	c.Arrival = ArrivalConfig{Process: "mmpp", Load: 0.8, MMPPRates: []float64{0.4, 4}, MMPPSojourn: []float64{21600, 3600}}
	m, sd = meanSD(interarrivals(gen(t, c, 13)))
	rate = c.ArrivalRate(13)
	if cv := sd / m; cv <= 1.2 || math.Abs(m*rate-1) > 0.1 {
		t.Fatalf("mmpp: CV %.3f, mean*rate %.3f", cv, m*rate)
	}
}

func TestDiurnalThinning(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 20000
	c.Arrival = ArrivalConfig{Process: "diurnal", Load: 0.8, Amplitude: 0.6, PeriodS: 86400}
	jobs := gen(t, c, 14)
	// Count arrivals in the rising half (sin > 0) and the falling half.
	var hi, lo float64
	for _, j := range jobs {
		if math.Sin(2*math.Pi*clock.Sec(j.Submit)/86400) > 0 {
			hi++
		} else {
			lo++
		}
	}
	// Expected ratio (1 + 2A/pi) / (1 - 2A/pi) for a sinusoidal rate.
	want := (1 + 2*0.6/math.Pi) / (1 - 2*0.6/math.Pi)
	if r := hi / lo; math.Abs(r/want-1) > 0.06 {
		t.Fatalf("diurnal ratio %.3f, want %.3f", r, want)
	}
}

func TestSizeAndPriorityMix(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 20000
	jobs := gen(t, c, 15)
	sizes := map[[2]int]float64{}
	pri := map[int]float64{}
	var targets, preempt float64
	for _, j := range jobs {
		sizes[[2]int{j.GPUs, j.Workers}]++
		pri[j.Priority]++
		if j.HasMaxWait {
			targets++
		}
		if j.Preemptible {
			preempt++
		}
	}
	n := float64(len(jobs))
	for _, s := range c.Sizes {
		if got := sizes[[2]int{s.GPUs, s.Workers}] / n; math.Abs(got-s.Weight) > 4*math.Sqrt(s.Weight*(1-s.Weight)/n) {
			t.Errorf("size %dx%d share %.4f, want %.4f", s.GPUs, s.Workers, got, s.Weight)
		}
	}
	for _, p := range c.Priorities {
		if got := pri[p.Priority] / n; math.Abs(got-p.Weight) > 0.015 {
			t.Errorf("priority %d share %.4f, want %.4f", p.Priority, got, p.Weight)
		}
	}
	// targets on priorities 4 and 8 (0.5 of jobs); preemptible 0.5*1 + 0.4*0.5 = 0.7
	if math.Abs(targets/n-0.5) > 0.015 || math.Abs(preempt/n-0.7) > 0.015 {
		t.Errorf("targets %.3f preemptible %.3f", targets/n, preempt/n)
	}
}

func TestEstimateModels(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 20000
	c.Estimate = EstimateConfig{Model: "lognormal", Mu: 0, Sigma: 0.5}
	var logs []float64
	for _, j := range gen(t, c, 16) {
		logs = append(logs, math.Log(clock.Sec(j.Estimate)/clock.Sec(j.Runtime)))
	}
	m, sd := meanSD(logs)
	if math.Abs(m) > 0.02 || math.Abs(sd-0.5) > 0.02 {
		t.Fatalf("lognormal estimate error: mean %.4f sd %.4f", m, sd)
	}
	pop := []float64{900, 1800, 3600, 7200, 14400, 28800, 43200, 86400, 172800}
	c.Estimate = EstimateConfig{Model: "user", PExact: 0.15, PUnder: 0.1, UnderMin: 0.5, UnderMax: 0.95, AccMin: 0.05, Popular: pop}
	var under, exact, onPopular, over float64
	isPop := map[float64]bool{}
	for _, p := range pop {
		isPop[p] = true
	}
	jobs := gen(t, c, 17)
	for _, j := range jobs {
		switch {
		case j.Estimate < j.Runtime:
			under++
		case j.Estimate == j.Runtime:
			exact++
		default:
			over++
			if isPop[clock.Sec(j.Estimate)] {
				onPopular++
			}
		}
	}
	n := float64(len(jobs))
	if math.Abs(under/n-0.1) > 0.01 || exact/n < 0.14 || onPopular/over < 0.95 {
		t.Fatalf("user model: under %.3f exact %.3f popular share of over %.3f", under/n, exact/n, onPopular/over)
	}
}

func TestOfferedLoadCalibration(t *testing.T) {
	// The arrival rate is calibrated per seed on the seed's user population,
	// so every seed is close to the target, not only the mean over seeds.
	// Tolerances: 6% on the mean of 20 seeds; 25% for any single seed
	// (heavy-tailed run times of individual jobs remain).
	c := baseConfig(t)
	c.Jobs = 2000
	loads := seedLoads(t, c, 20)
	mean, worst := 0.0, 0.0
	for _, l := range loads {
		mean += l / float64(len(loads))
		worst = math.Max(worst, math.Abs(l/c.Arrival.Load-1))
	}
	t.Logf("realised loads: mean %.3f, worst relative deviation %.3f", mean, worst)
	if math.Abs(mean/c.Arrival.Load-1) > 0.06 || worst > 0.25 {
		t.Fatalf("realised loads %v (mean %.3f, worst deviation %.3f), target %.3f", loads, mean, worst, c.Arrival.Load)
	}
}

func seedLoads(t *testing.T, c GenConfig, seeds int) []float64 {
	var out []float64
	for s := 1; s <= seeds; s++ {
		jobs := gen(t, c, uint64(s))
		work := 0.0
		for _, j := range jobs {
			work += float64(j.TotalGPUs()) * clock.Sec(j.Runtime)
		}
		out = append(out, work/(clock.Sec(jobs[len(jobs)-1].Submit)*c.WorkCapacity))
	}
	return out
}

func TestBatchBurstAndSlotGrid(t *testing.T) {
	c := baseConfig(t)
	c.Jobs = 100
	c.Arrival = ArrivalConfig{Process: "batch"}
	for _, j := range gen(t, c, 1) {
		if j.Submit != 0 {
			t.Fatal("batch job not at 0")
		}
	}
	c = baseConfig(t)
	c.Jobs = 3000
	mw := 600.0
	c.Burst = &BurstConfig{MeanGapS: 20000, Jobs: 20, SpreadS: 600, Priority: 9, MaxWaitS: &mw}
	jobs := gen(t, c, 2)
	if _, err := Parse(Encode(jobs)); err != nil {
		t.Fatal(err)
	}
	burst := 0
	for _, j := range jobs {
		if j.Priority == 9 {
			burst++
			if j.Preemptible || !j.HasMaxWait {
				t.Fatal("burst job attributes")
			}
		}
	}
	if burst == 0 || burst%20 != 0 {
		t.Fatalf("burst jobs %d", burst)
	}
	c = baseConfig(t)
	c.Jobs = 300
	c.SlotGrid = &SlotGridConfig{SlotS: 300, MinSlots: 1, MaxSlots: 6}
	for _, j := range gen(t, c, 3) {
		if j.Submit%(300e3*1e6) != 0 || j.Runtime%(300e3*1e6) != 0 || j.Runtime > 1800e9 {
			t.Fatalf("not on the slot grid: %+v", j)
		}
	}
}
