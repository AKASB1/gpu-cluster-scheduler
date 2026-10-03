package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
)

// SearchPoint is one evaluated configuration of the random search.
type SearchPoint struct {
	Params json.RawMessage `json:"params"`
	JMean  float64         `json:"J_mean"`
}

// ScenarioTuning is the frozen tuning result of one scenario.
type ScenarioTuning struct {
	Reference Reference                  `json:"reference"`
	Params    map[string]json.RawMessage `json:"params"`
	Search    map[string][]SearchPoint   `json:"search"`
}

// DefaultsPoint is one candidate of the defaults selection.
type DefaultsPoint struct {
	Step   string  `json:"step"`
	Policy string  `json:"policy"`
	JMean  float64 `json:"J_mean"`
}

// Tuned is configs/tuned/tuned.json.
type Tuned struct {
	Experiment     string                     `json:"experiment"`
	ConfigSHA256   string                     `json:"config_sha256"`
	TuningSeeds    []uint64                   `json:"tuning_seeds"`
	Configurations int                        `json:"configurations"`
	SearchSeed     uint64                     `json:"search_seed"`
	Defaults       Defaults                   `json:"defaults"`
	DefaultsSearch []DefaultsPoint            `json:"defaults_search"`
	Scenarios      map[string]*ScenarioTuning `json:"scenarios"`
	// Added lists scenarios tuned later with -tune-add (frozen entries unchanged).
	Added []string `json:"added_scenarios,omitempty"`
}

// LoadTuned reads a frozen tuning result.
func LoadTuned(path string) (*Tuned, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tuned configuration %s: %w (run the tuning command first)", path, err)
	}
	var t Tuned
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &t, nil
}

// For returns the tuning entry that applies to scenario sc.
func (t *Tuned) For(sc *Scenario) (*ScenarioTuning, error) {
	id := sc.ID
	if sc.TunedFrom != "" {
		id = sc.TunedFrom
	}
	st := t.Scenarios[id]
	if st == nil {
		return nil, fmt.Errorf("scenario %s: no frozen tuning entry %q", sc.ID, id)
	}
	return st, nil
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

// sampler returns the search space of a policy, or nil if it has no
// parameters. Every parameterized policy gets the same number of
// configurations.
func sampler(name string) func(r *rand.Rand) map[string]any {
	if name == "py:milp_rolling" {
		return func(r *rand.Rand) map[string]any {
			return map[string]any{"k": 4 + r.IntN(21), "h": 8 + r.IntN(41), "slot_s": []float64{120, 300, 600, 900}[r.IntN(4)],
				"mip_rel_gap": []float64{0.01, 0.05}[r.IntN(2)]}
		}
	}
	if name == "py:scenario_milp" {
		return func(r *rand.Rand) map[string]any {
			return map[string]any{"k": 4 + r.IntN(9), "h": 6 + r.IntN(19), "slot_s": []float64{300, 600, 900}[r.IntN(3)],
				"scenarios": []int{4, 8, 12}[r.IntN(3)], "lambda": []float64{0, 0.25, 0.5, 1}[r.IntN(4)],
				"alpha": []float64{0.8, 0.9, 0.95}[r.IntN(3)], "mip_rel_gap": []float64{0.01, 0.05}[r.IntN(2)]}
		}
	}
	parts := strings.Split(name, "+")
	if len(parts) < 3 || strings.HasPrefix(name, "py:") {
		return nil
	}
	prio, pred, pre, risk := parts[0] == "priority", parts[2] == "easy_predicted", slices.Contains(parts[3:], "preempt"), parts[2] == "risk"
	if !prio && !pred && !pre && !risk {
		return nil
	}
	return func(r *rand.Rand) map[string]any {
		m := map[string]any{}
		if prio {
			m["aging_per_hour"] = round3(4 * r.Float64())
		}
		if pred {
			m["history"] = 1 + r.IntN(5)
		}
		if risk {
			m["quantile"] = round3(0.5 + 0.49*r.Float64())
		}
		if pre {
			m["min_gap"] = 1 + r.IntN(7)
			m["max_victims"] = []int{1, 2, 4, 8, 16}[r.IntN(5)]
		}
		return m
	}
}

// meanJ groups rows by policy (with tag) and averages J.
func meanJ(rows []Row) map[string]float64 {
	sum, n := map[string]float64{}, map[string]int{}
	for i := range rows {
		sum[rows[i].Policy] += rows[i].Get("J")
		n[rows[i].Policy]++
	}
	out := map[string]float64{}
	for k := range sum {
		out[k] = sum[k] / float64(n[k])
	}
	return out
}

// Tune runs the tuning procedure on the tuning seeds only and returns the
// frozen configuration (benchmarks/README.md, "Tuning"). With base != nil it
// tunes only the scenarios that base does not contain yet (scenarios added
// later), keeps every frozen entry and the default parts of base unchanged,
// and lists the added scenarios.
func (env *Env) Tune(ctx context.Context, exp *Experiment, configHash string, base *Tuned) (*Tuned, error) {
	t := &Tuned{Experiment: exp.Name, ConfigSHA256: configHash, TuningSeeds: exp.TuningSeeds,
		Configurations: exp.Tuning.Configurations, SearchSeed: exp.Tuning.SearchSeed, Scenarios: map[string]*ScenarioTuning{}}
	if base != nil {
		cp := *base
		cp.ConfigSHA256 = configHash
		cp.Scenarios = map[string]*ScenarioTuning{}
		for k, v := range base.Scenarios {
			cp.Scenarios[k] = v
		}
		t = &cp
	}
	var scs []*Scenario
	traces := map[string]map[uint64]*TraceSet{}
	for i := range exp.Scenarios {
		sc := &exp.Scenarios[i]
		if sc.Kind != KindSim || sc.TunedFrom != "" {
			continue
		}
		if base != nil && base.Scenarios[sc.ID] != nil {
			continue // frozen earlier: never re-tuned
		}
		if base != nil {
			t.Added = append(t.Added, sc.ID)
		}
		ts, err := env.Traces(sc, exp.TuningSeeds)
		if errors.Is(err, ErrReplayData) {
			env.logf("%s not tuned: %v", sc.ID, err)
			if base != nil {
				t.Added = t.Added[:len(t.Added)-1]
			}
			continue
		}
		if err != nil {
			return nil, err
		}
		scs, traces[sc.ID] = append(scs, sc), ts
	}
	tasks := func(sc *Scenario, pol string, params json.RawMessage, ref Reference, tag string) []Task {
		var out []Task
		for _, seed := range exp.TuningSeeds {
			out = append(out, Task{Scenario: sc.ID, Policy: pol, Params: sc.paramsFor(pol, params), Seed: seed, Trace: traces[sc.ID][seed], Shares: sc.QuotaShares,
				Ref: ref, Weights: sc.weights(exp.Weights), Tag: tag})
		}
		return out
	}
	// 1. references R_s, R_w from the baseline on the tuning seeds
	env.logf("tuning: references (%s on %d scenarios)", exp.Baseline, len(scs))
	var all []Task
	for _, sc := range scs {
		all = append(all, tasks(sc, exp.Baseline, nil, Reference{1, 1}, "")...)
	}
	rows, err := env.RunTasks(ctx, all)
	if err != nil {
		return nil, err
	}
	for _, sc := range scs {
		var s, w []float64
		for _, r := range rows {
			if r.Scenario == sc.ID {
				s, w = append(s, r.Get("bsld_p95")), append(w, r.Get("wait_p95"))
			}
		}
		t.Scenarios[sc.ID] = &ScenarioTuning{Reference: Reference{Rs: metrics.Mean(s), Rw: math.Max(1, metrics.Mean(w))},
			Params: map[string]json.RawMessage{}, Search: map[string][]SearchPoint{}}
	}
	// 2. default parts on the defaults scenario, one dimension at a time (kept
	// from base when adding scenarios)
	if base == nil {
		if err := env.tuneDefaults(ctx, exp, t, tasks); err != nil {
			return nil, err
		}
	}
	return env.tuneSearch(ctx, exp, t, scs, tasks)
}

func (env *Env) tuneDefaults(ctx context.Context, exp *Experiment, t *Tuned,
	tasks func(*Scenario, string, json.RawMessage, Reference, string) []Task) error {
	dsc := exp.Scenario(exp.DefaultsScenario)
	ref := t.Scenarios[dsc.ID].Reference
	pick := func(step string, cands []string) (string, error) {
		var ts []Task
		for _, c := range cands {
			ts = append(ts, tasks(dsc, c, nil, ref, "")...)
		}
		rows, err := env.RunTasks(ctx, ts)
		if err != nil {
			return "", err
		}
		mj := meanJ(rows)
		best := ""
		for _, c := range cands {
			t.DefaultsSearch = append(t.DefaultsSearch, DefaultsPoint{Step: step, Policy: c, JMean: mj[c]})
			if best == "" || mj[c] < mj[best] {
				best = c
			}
		}
		env.logf("tuning: %s -> %s (J %.4f)", step, best, mj[best])
		return best, nil
	}
	var cands []string
	for _, p := range []string{"first_fit", "best_fit", "least_fragmentation", "topology_aware"} {
		cands = append(cands, "fifo+"+p+"+none")
	}
	best, err := pick("placement", cands)
	if err != nil {
		return err
	}
	t.Defaults.Placement = strings.Split(best, "+")[1]
	cands = nil
	for _, b := range []string{"none", "easy", "conservative", "easy_predicted"} {
		cands = append(cands, "fifo+"+t.Defaults.Placement+"+"+b)
	}
	if best, err = pick("backfill", cands); err != nil {
		return err
	}
	t.Defaults.Backfill = strings.Split(best, "+")[2]
	cands = nil
	for _, o := range []string{"fifo", "priority", "shortest_estimate", "drf"} {
		cands = append(cands, o+"+"+t.Defaults.Placement+"+"+t.Defaults.Backfill)
	}
	if best, err = pick("order", cands); err != nil {
		return err
	}
	t.Defaults.Order = strings.Split(best, "+")[0]
	return nil
}

// tuneSearch is step 3: equal-budget random search per scenario and
// parameterized policy.
func (env *Env) tuneSearch(ctx context.Context, exp *Experiment, t *Tuned, scs []*Scenario,
	tasks func(*Scenario, string, json.RawMessage, Reference, string) []Task) (*Tuned, error) {
	var all []Task
	type key struct{ sc, pol string }
	configs := map[key][]json.RawMessage{}
	for _, sc := range scs {
		for _, pol := range Expand(sc.Policies, t.Defaults) {
			s := sampler(pol)
			if s == nil {
				continue
			}
			r := rng.New(exp.Tuning.SearchSeed, "tune/"+sc.ID+"/"+pol)
			k := key{sc.ID, pol}
			for c := 0; c < exp.Tuning.Configurations; c++ {
				params := json.RawMessage(`{}`)
				if c > 0 {
					params, _ = json.Marshal(s(r))
				}
				configs[k] = append(configs[k], params)
				all = append(all, tasks(sc, pol, params, t.Scenarios[sc.ID].Reference, fmt.Sprintf("#c%d", c))...)
			}
		}
	}
	env.logf("tuning: random search, %d runs", len(all))
	rows, err := env.RunTasks(ctx, all)
	if err != nil {
		return nil, err
	}
	for _, sc := range scs {
		var srows []Row
		for _, r := range rows {
			if r.Scenario == sc.ID {
				srows = append(srows, r)
			}
		}
		mj := meanJ(srows)
		for _, pol := range Expand(sc.Policies, t.Defaults) {
			cfgs := configs[key{sc.ID, pol}]
			if cfgs == nil {
				continue
			}
			bestC := 0
			var pts []SearchPoint
			for c, params := range cfgs {
				j := mj[fmt.Sprintf("%s#c%d", pol, c)]
				pts = append(pts, SearchPoint{Params: params, JMean: j})
				if j < pts[bestC].JMean {
					bestC = c
				}
			}
			st := t.Scenarios[sc.ID]
			st.Search[pol], st.Params[pol] = pts, cfgs[bestC]
		}
	}
	return t, nil
}

// Save writes the tuned configuration.
func (t *Tuned) Save(path string) error {
	b, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
