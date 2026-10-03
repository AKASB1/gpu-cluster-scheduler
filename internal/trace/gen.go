package trace

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
)

// GeneratorName and GeneratorVersion identify this generator in manifests.
const (
	GeneratorName    = "gcs-gen"
	GeneratorVersion = 1
)

// GenConfig is the committed configuration of one synthetic workload. Every
// distribution and parameter is an assumption documented in docs/simulator.md.
type GenConfig struct {
	Jobs int `json:"jobs"`
	// WorkCapacity is the cluster's reference-GPU work rate (sum of GPUs x
	// speed); the arrival rate is load x WorkCapacity / E[GPUs x runtime].
	WorkCapacity float64         `json:"work_capacity"`
	Arrival      ArrivalConfig   `json:"arrival"`
	Sizes        []SizeClass     `json:"sizes"`
	Runtime      RuntimeConfig   `json:"runtime"`
	Tenants      []TenantSpec    `json:"tenants"`
	Priorities   []PriorityClass `json:"priorities"`
	// CheckpointIntervalS applies to every job (0 = no checkpoint).
	CheckpointIntervalS float64         `json:"checkpoint_interval_s"`
	Resources           ResourceConfig  `json:"resources"`
	ClassConstraints    []ClassFraction `json:"class_constraints,omitempty"`
	Estimate            EstimateConfig  `json:"estimate"`
	Burst               *BurstConfig    `json:"burst,omitempty"`
	SlotGrid            *SlotGridConfig `json:"slot_grid,omitempty"`
}

// ArrivalConfig selects the arrival process.
type ArrivalConfig struct {
	Process string  `json:"process"` // poisson, gamma, mmpp, diurnal, batch
	Load    float64 `json:"load"`    // offered load (not used by batch)
	CV      float64 `json:"cv,omitempty"`
	// MMPP: relative rates and mean sojourn times per state; the relative
	// rates are normalised so that the time-average rate is the target rate.
	MMPPRates   []float64 `json:"mmpp_rates,omitempty"`
	MMPPSojourn []float64 `json:"mmpp_sojourn_s,omitempty"`
	// Diurnal: rate(t) = mean * (1 + amplitude * sin(2 pi t / period)).
	Amplitude float64 `json:"amplitude,omitempty"`
	PeriodS   float64 `json:"period_s,omitempty"`
}

// SizeClass is a job shape with its probability.
type SizeClass struct {
	Weight  float64 `json:"weight"`
	GPUs    int     `json:"gpus"`
	Workers int     `json:"workers"`
	// Topology is the sensitivity given with probability TopoFrac (else any).
	Topology api.Topology `json:"topology,omitempty"`
	TopoFrac float64      `json:"topo_frac,omitempty"`
}

// RuntimeConfig: each user has a typical run time drawn once from
// lognormal(ln(UserMedianS), UserSigma); a job's run time is the typical
// value times lognormal(0, JobSigma), truncated to [MinS, MaxS].
type RuntimeConfig struct {
	UserMedianS float64 `json:"user_median_s"`
	UserSigma   float64 `json:"user_sigma"`
	JobSigma    float64 `json:"job_sigma"`
	MinS        float64 `json:"min_s"`
	MaxS        float64 `json:"max_s"`
}

// TenantSpec is a tenant's share of submissions and its number of users.
type TenantSpec struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
	Users  int     `json:"users"`
}

// PriorityClass is a priority level with its share and job attributes.
type PriorityClass struct {
	Priority        int      `json:"priority"`
	Weight          float64  `json:"weight"`
	PreemptibleFrac float64  `json:"preemptible_frac"`
	MaxWaitS        *float64 `json:"max_wait_s,omitempty"`
}

// ResourceConfig sets CPUs and memory per worker in proportion to its GPUs.
type ResourceConfig struct {
	CPUsPerGPU     int     `json:"cpus_per_gpu"`
	MemPerGPU      float64 `json:"mem_gb_per_gpu"`
	CPUHeavyFrac   float64 `json:"cpu_heavy_frac,omitempty"`
	CPUHeavyPerGPU int     `json:"cpu_heavy_per_gpu,omitempty"`
	MemHeavyFrac   float64 `json:"mem_heavy_frac,omitempty"`
	MemHeavyPerGPU float64 `json:"mem_heavy_gb_per_gpu,omitempty"`
}

// ClassFraction constrains a fraction of jobs to one GPU class.
type ClassFraction struct {
	Class string  `json:"class"`
	Frac  float64 `json:"frac"`
}

// EstimateConfig is the estimate model estimate = runtime * m.
type EstimateConfig struct {
	Model string `json:"model"` // exact, lognormal, user
	// lognormal: m = exp(N(Mu, Sigma)).
	Mu    float64 `json:"mu,omitempty"`
	Sigma float64 `json:"sigma,omitempty"`
	// user (after Mu'alem and Feitelson, plus underestimates): with PExact the
	// estimate is exact; with PUnder m ~ U(UnderMin, UnderMax); otherwise the
	// accuracy runtime/estimate ~ U(AccMin, 1) and the estimate is rounded up
	// to the next popular value (kept as is above the largest one).
	PExact   float64   `json:"p_exact,omitempty"`
	PUnder   float64   `json:"p_under,omitempty"`
	UnderMin float64   `json:"under_min,omitempty"`
	UnderMax float64   `json:"under_max,omitempty"`
	AccMin   float64   `json:"acc_min,omitempty"`
	Popular  []float64 `json:"popular_s,omitempty"`
}

// BurstConfig adds bursts of high-priority jobs: burst starts form a Poisson
// process with mean gap MeanGapS; each burst has Jobs jobs submitted
// uniformly within SpreadS; burst jobs get Priority, MaxWaitS, and are not
// preemptible. Their sizes and run times follow the main mix.
type BurstConfig struct {
	MeanGapS float64  `json:"mean_gap_s"`
	Jobs     int      `json:"jobs"`
	SpreadS  float64  `json:"spread_s"`
	Priority int      `json:"priority"`
	MaxWaitS *float64 `json:"max_wait_s,omitempty"`
}

// SlotGridConfig makes every submit time and run time a whole number of
// slots (the MILP instances of S8): run times are drawn uniformly from
// [MinSlots, MaxSlots], submit times are rounded down to the slot grid.
type SlotGridConfig struct {
	SlotS    float64 `json:"slot_s"`
	MinSlots int     `json:"min_slots"`
	MaxSlots int     `json:"max_slots"`
}

// Generate builds the trace of cfg under seed. The result depends only on
// (cfg, seed). Job IDs are j000000, j000001, ... in submit order.
func Generate(cfg GenConfig, seed uint64) ([]api.Job, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	typical := cfg.userTypicals(seed)
	times := arrivals(cfg, seed, cfg.meanWorkGiven(typical))
	type arrival struct {
		t     float64
		burst bool
	}
	all := make([]arrival, 0, len(times))
	for _, t := range times {
		all = append(all, arrival{t: t})
	}
	if b := cfg.Burst; b != nil && len(times) > 0 {
		r := rng.New(seed, "bursts")
		horizon := times[len(times)-1]
		for start := r.ExpFloat64() * b.MeanGapS; start < horizon; start += r.ExpFloat64() * b.MeanGapS {
			for k := 0; k < b.Jobs; k++ {
				all = append(all, arrival{t: start + r.Float64()*b.SpreadS, burst: true})
			}
		}
	}
	// Round to ms first, then order by (time, main-before-burst, index) so the
	// order is a total order independent of float ties.
	type stamped struct {
		ms    time.Duration
		burst bool
		idx   int
	}
	st := make([]stamped, len(all))
	for i, a := range all {
		st[i] = stamped{clock.RoundMs(a.t), a.burst, i}
	}
	slices.SortFunc(st, func(a, b stamped) int {
		if c := cmp.Compare(a.ms, b.ms); c != 0 {
			return c
		}
		if a.burst != b.burst {
			if a.burst {
				return 1
			}
			return -1
		}
		return cmp.Compare(a.idx, b.idx)
	})

	sizeR := rng.New(seed, "sizes")
	runR := rng.New(seed, "runtimes")
	tenR := rng.New(seed, "tenants")
	priR := rng.New(seed, "priorities")
	resR := rng.New(seed, "resources")
	clsR := rng.New(seed, "classes")
	estR := rng.New(seed, "estimates")

	rc := cfg.Runtime
	jobs := make([]api.Job, len(st))
	for i, s := range st {
		j := &jobs[i]
		j.ID = fmt.Sprintf("j%06d", i)
		j.Submit = s.ms
		// tenant and user
		ti := pick(tenR, len(cfg.Tenants), func(k int) float64 { return cfg.Tenants[k].Weight })
		ten := cfg.Tenants[ti]
		j.Tenant = ten.Name
		j.User = fmt.Sprintf("%s-u%d", ten.Name, tenR.IntN(ten.Users))
		// size
		sc := cfg.Sizes[pick(sizeR, len(cfg.Sizes), func(k int) float64 { return cfg.Sizes[k].Weight })]
		j.GPUs, j.Workers, j.Topology = sc.GPUs, sc.Workers, api.TopoAny
		if sc.Topology != "" && sc.Topology != api.TopoAny && sizeR.Float64() < sc.TopoFrac {
			j.Topology = sc.Topology
		}
		// run time
		if g := cfg.SlotGrid; g != nil {
			slots := g.MinSlots + runR.IntN(g.MaxSlots-g.MinSlots+1)
			j.Runtime = clock.RoundMs(float64(slots) * g.SlotS)
			slot := clock.RoundMs(g.SlotS)
			j.Submit = j.Submit / slot * slot
		} else {
			rt := typical[j.User] * math.Exp(rc.JobSigma*runR.NormFloat64())
			rt = math.Min(math.Max(rt, rc.MinS), rc.MaxS)
			j.Runtime = max(clock.RoundMs(rt), time.Millisecond)
		}
		// priority class
		pc := cfg.Priorities[pick(priR, len(cfg.Priorities), func(k int) float64 { return cfg.Priorities[k].Weight })]
		j.Priority = pc.Priority
		j.Preemptible = priR.Float64() < pc.PreemptibleFrac
		maxWait := pc.MaxWaitS
		if s.burst {
			j.Priority, j.Preemptible, maxWait = cfg.Burst.Priority, false, cfg.Burst.MaxWaitS
		}
		if maxWait != nil {
			j.MaxWait, j.HasMaxWait = clock.RoundMs(*maxWait), true
		}
		j.CheckpointInterval = clock.RoundMs(cfg.CheckpointIntervalS)
		// resources per worker
		res := cfg.Resources
		cpg, mpg := res.CPUsPerGPU, res.MemPerGPU
		if resR.Float64() < res.CPUHeavyFrac {
			cpg = res.CPUHeavyPerGPU
		}
		if resR.Float64() < res.MemHeavyFrac {
			mpg = res.MemHeavyPerGPU
		}
		j.CPUs, j.MemGB = cpg*j.GPUs, mpg*float64(j.GPUs)
		// class constraint
		u := clsR.Float64()
		for _, cf := range cfg.ClassConstraints {
			if u < cf.Frac {
				j.GPUClass = cf.Class
				break
			}
			u -= cf.Frac
		}
		j.Estimate = estimate(cfg.Estimate, estR, j.Runtime)
	}
	// Burst rounding to the slot grid can reorder submit times only in the
	// slot-grid mode; keep the file sorted.
	if cfg.SlotGrid != nil {
		slices.SortStableFunc(jobs, func(a, b api.Job) int { return cmp.Compare(a.Submit, b.Submit) })
		for i := range jobs {
			jobs[i].ID = fmt.Sprintf("j%06d", i)
		}
	}
	return jobs, nil
}

// arrivals returns the main-stream arrival times in seconds.
func arrivals(cfg GenConfig, seed uint64, meanWork float64) []float64 {
	a := cfg.Arrival
	out := make([]float64, 0, cfg.Jobs)
	if a.Process == "batch" {
		for i := 0; i < cfg.Jobs; i++ {
			out = append(out, 0)
		}
		return out
	}
	rate := a.Load * cfg.WorkCapacity / meanWork // jobs per second
	r := rng.New(seed, "arrivals")
	t := 0.0
	switch a.Process {
	case "poisson":
		for len(out) < cfg.Jobs {
			t += r.ExpFloat64() / rate
			out = append(out, t)
		}
	case "gamma":
		shape := 1 / (a.CV * a.CV)
		scale := 1 / (rate * shape)
		for len(out) < cfg.Jobs {
			t += gammaSample(r, shape) * scale
			out = append(out, t)
		}
	case "mmpp":
		// Normalise relative rates so the time-average rate equals rate.
		tot, avg := 0.0, 0.0
		for i := range a.MMPPRates {
			tot += a.MMPPSojourn[i]
			avg += a.MMPPSojourn[i] * a.MMPPRates[i]
		}
		avg /= tot
		state := 0
		end := r.ExpFloat64() * a.MMPPSojourn[0]
		for len(out) < cfg.Jobs {
			lam := rate * a.MMPPRates[state] / avg
			next := t + r.ExpFloat64()/lam
			if next > end {
				t = end
				state = (state + 1) % len(a.MMPPRates)
				end = t + r.ExpFloat64()*a.MMPPSojourn[state]
				continue
			}
			t = next
			out = append(out, t)
		}
	case "diurnal":
		peak := rate * (1 + a.Amplitude)
		for len(out) < cfg.Jobs {
			t += r.ExpFloat64() / peak
			if r.Float64()*peak <= rate*(1+a.Amplitude*math.Sin(2*math.Pi*t/a.PeriodS)) {
				out = append(out, t)
			}
		}
	}
	return out
}

// gammaSample draws Gamma(shape, 1) by Marsaglia and Tsang (with the
// U^(1/shape) boost for shape < 1).
func gammaSample(r *rand.Rand, shape float64) float64 {
	if shape < 1 {
		u := r.Float64()
		for u == 0 {
			u = r.Float64()
		}
		return gammaSample(r, shape+1) * math.Pow(u, 1/shape)
	}
	d := shape - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := r.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := r.Float64()
		if u < 1-0.0331*x*x*x*x || (u > 0 && math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v))) {
			return d * v
		}
	}
}

func estimate(e EstimateConfig, r *rand.Rand, runtime time.Duration) time.Duration {
	rt := clock.Sec(runtime)
	var est float64
	switch e.Model {
	case "exact":
		return runtime
	case "lognormal":
		est = rt * math.Exp(e.Mu+e.Sigma*r.NormFloat64())
	case "user":
		u := r.Float64()
		switch {
		case u < e.PExact:
			return runtime
		case u < e.PExact+e.PUnder:
			est = rt * (e.UnderMin + (e.UnderMax-e.UnderMin)*r.Float64())
		default:
			acc := e.AccMin + (1-e.AccMin)*r.Float64()
			est = rt / acc
			for _, p := range e.Popular {
				if p >= est {
					est = p
					break
				}
			}
			return max(clock.FromSecondsCeil(est), runtime)
		}
	}
	return max(clock.RoundMs(est), time.Millisecond)
}

// pick draws an index with probability proportional to weight(k).
func pick(r *rand.Rand, n int, weight func(int) float64) int {
	tot := 0.0
	for k := 0; k < n; k++ {
		tot += weight(k)
	}
	u := r.Float64() * tot
	for k := 0; k < n; k++ {
		u -= weight(k)
		if u < 0 {
			return k
		}
	}
	return n - 1
}

// userTypicals draws every user's typical run time up front, in a fixed
// order (tenants in configuration order, users by index), from the stream
// "users": typical = user_median_s * exp(user_sigma * N(0,1)).
func (cfg GenConfig) userTypicals(seed uint64) map[string]float64 {
	r := rng.New(seed, "users")
	out := map[string]float64{}
	for _, t := range cfg.Tenants {
		for u := 0; u < t.Users; u++ {
			out[fmt.Sprintf("%s-u%d", t.Name, u)] = cfg.Runtime.UserMedianS * math.Exp(cfg.Runtime.UserSigma*r.NormFloat64())
		}
	}
	return out
}

// meanWorkGiven is E[GPUs x workers x runtime] of the main mix given the
// drawn user population (reference-GPU-seconds): E[GPUs] times the
// submission-weighted mean over users of E[clip(typical x exp(job_sigma Z))],
// the inner expectation by Monte Carlo on a fixed stream (seed 0, 4000
// draws, the same draws for every user). The arrival rate is calibrated per
// seed with it, so every seed has the configured offered load in
// expectation (the run-time noise of individual jobs remains).
func (cfg GenConfig) meanWorkGiven(typical map[string]float64) float64 {
	meanG, tot := 0.0, 0.0
	for _, s := range cfg.Sizes {
		meanG += s.Weight * float64(s.GPUs*s.Workers)
		tot += s.Weight
	}
	meanG /= tot
	if g := cfg.SlotGrid; g != nil {
		return meanG * g.SlotS * float64(g.MinSlots+g.MaxSlots) / 2
	}
	rc := cfg.Runtime
	r := rng.New(0, "calibration")
	z := make([]float64, 4000)
	for i := range z {
		z[i] = r.NormFloat64()
	}
	wsum, rt := 0.0, 0.0
	for _, t := range cfg.Tenants {
		wsum += t.Weight
	}
	for _, t := range cfg.Tenants {
		for u := 0; u < t.Users; u++ {
			typ := typical[fmt.Sprintf("%s-u%d", t.Name, u)]
			e := 0.0
			for _, x := range z {
				e += math.Min(math.Max(typ*math.Exp(rc.JobSigma*x), rc.MinS), rc.MaxS)
			}
			rt += t.Weight / wsum / float64(t.Users) * e / float64(len(z))
		}
	}
	return meanG * rt
}

// ArrivalRate is the main-stream arrival rate (jobs per second) of seed:
// load x work_capacity / E[GPUs x runtime | the seed's user population].
func (cfg GenConfig) ArrivalRate(seed uint64) float64 {
	return cfg.Arrival.Load * cfg.WorkCapacity / cfg.meanWorkGiven(cfg.userTypicals(seed))
}

func (cfg GenConfig) validate() error {
	fail := func(format string, a ...any) error { return fmt.Errorf("generator config: "+format, a...) }
	if cfg.Jobs < 0 {
		return fail("jobs must be >= 0")
	}
	switch cfg.Arrival.Process {
	case "batch":
	case "poisson", "gamma", "mmpp", "diurnal":
		if !(cfg.Arrival.Load > 0) || !(cfg.WorkCapacity > 0) {
			return fail("load and work_capacity must be > 0")
		}
	default:
		return fail("unknown arrival process %q", cfg.Arrival.Process)
	}
	if cfg.Arrival.Process == "gamma" && !(cfg.Arrival.CV > 0) {
		return fail("gamma needs cv > 0")
	}
	if cfg.Arrival.Process == "mmpp" && (len(cfg.Arrival.MMPPRates) < 2 || len(cfg.Arrival.MMPPRates) != len(cfg.Arrival.MMPPSojourn)) {
		return fail("mmpp needs >= 2 rates and as many sojourn times")
	}
	if cfg.Arrival.Process == "diurnal" && (cfg.Arrival.Amplitude < 0 || cfg.Arrival.Amplitude > 1 || !(cfg.Arrival.PeriodS > 0)) {
		return fail("diurnal needs 0 <= amplitude <= 1 and period_s > 0")
	}
	if len(cfg.Sizes) == 0 || len(cfg.Tenants) == 0 || len(cfg.Priorities) == 0 {
		return fail("sizes, tenants, and priorities must be non-empty")
	}
	for _, s := range cfg.Sizes {
		if s.GPUs < 1 || s.Workers < 1 || s.Weight < 0 {
			return fail("bad size class %+v", s)
		}
	}
	for _, t := range cfg.Tenants {
		if t.Name == "" || t.Users < 1 {
			return fail("bad tenant %+v", t)
		}
	}
	if g := cfg.SlotGrid; g != nil {
		if !(g.SlotS > 0) || g.MinSlots < 1 || g.MaxSlots < g.MinSlots {
			return fail("bad slot grid")
		}
	} else if !(cfg.Runtime.MinS > 0) || cfg.Runtime.MaxS < cfg.Runtime.MinS || !(cfg.Runtime.UserMedianS > 0) {
		return fail("bad runtime config")
	}
	switch cfg.Estimate.Model {
	case "exact", "lognormal":
	case "user":
		if !slices.IsSorted(cfg.Estimate.Popular) {
			return fail("popular_s must be ascending")
		}
	default:
		return fail("unknown estimate model %q", cfg.Estimate.Model)
	}
	if cfg.Burst != nil && (!(cfg.Burst.MeanGapS > 0) || cfg.Burst.Jobs < 1 || cfg.Burst.SpreadS < 0) {
		return fail("bad burst config")
	}
	return nil
}

// Info returns the manifest generator entry for cfg.
func (cfg GenConfig) Info() GeneratorInfo {
	b, _ := json.Marshal(cfg) // plain struct: cannot fail
	return GeneratorInfo{Name: GeneratorName, Version: GeneratorVersion, Params: b}
}
