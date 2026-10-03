package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

// Scenario kinds.
const (
	KindSim        = "sim"
	KindMILPSmall  = "milp-small"
	KindMILPMedium = "milp-medium"
)

// Weights are the objective weights of J (docs/contracts.md §5).
type Weights struct {
	Alpha   float64 `json:"alpha"`
	Beta    float64 `json:"beta"`
	Gamma   float64 `json:"gamma"`
	Delta   float64 `json:"delta"`
	Epsilon float64 `json:"epsilon"`
}

// Scenario is one simulated setting (cluster + workload + policy list).
type Scenario struct {
	ID       string          `json:"id"`
	Group    string          `json:"group"`
	Kind     string          `json:"kind,omitempty"`
	Note     string          `json:"note,omitempty"`
	Cluster  string          `json:"cluster"`
	Workload string          `json:"workload"`
	Override json.RawMessage `json:"overrides,omitempty"`
	// PenaltyScale and OverheadScale scale the topology penalty (f - 1) and
	// the restart overhead of the cluster (sensitivity sweep; 0 means 1).
	PenaltyScale  float64 `json:"penalty_scale,omitempty"`
	OverheadScale float64 `json:"overhead_scale,omitempty"`
	// Policies may use the placeholders {order}, {placement}, {backfill}:
	// the defaults chosen on the tuning seeds.
	Policies []string `json:"policies"`
	// Seeds overrides the experiment's evaluation seeds (MILP: instance seeds).
	Seeds []uint64 `json:"seeds,omitempty"`
	// TunedFrom: a sensitivity variant reuses the frozen parameters and the
	// reference values of this scenario (no re-tuning).
	TunedFrom string `json:"tuned_from,omitempty"`
	// Assumption and Scale label a sensitivity variant.
	Assumption        string  `json:"assumption,omitempty"`
	Scale             float64 `json:"scale,omitempty"`
	ScheduleIntervalS float64 `json:"schedule_interval_s,omitempty"`
	// MILP settings (S8).
	SlotS      float64 `json:"slot_s,omitempty"`
	NodeLimit  *int    `json:"node_limit,omitempty"`
	TimeLimitS float64 `json:"time_limit_s,omitempty"`
	// Replay replaces the generated workload by an excerpt of a public cluster
	// log (Tier 2); Cluster and Workload are then unused.
	Replay *ReplayConfig `json:"replay,omitempty"`
	// QuotaShares are the nominal GPU quotas per tenant (share of the cluster)
	// of a quota scenario: injected into the parameters of +quota policies and
	// used by the quota metrics.
	QuotaShares map[string]float64 `json:"quota_shares,omitempty"`
	// Failures adds node failures (Tier 2): per node, exponential times
	// between failures with mean MTBFS and exponential repair times with mean
	// RepairS, from the stream "failures/<node>" of the seed (common to all
	// policies).
	Failures *FailureConfig `json:"failures,omitempty"`
	// ZeroWeights lists J terms (J_slowdown, J_wait, J_waste, J_slo,
	// J_fairness) whose weight is 0 in this scenario because the J-term check
	// on the tuning seeds found them uninformative there (benchmarks/README.md).
	ZeroWeights []string `json:"zero_weights,omitempty"`
}

// weights returns the objective weights of the scenario.
func (s *Scenario) weights(w Weights) Weights {
	for _, t := range s.ZeroWeights {
		switch t {
		case "J_slowdown":
			w.Alpha = 0
		case "J_wait":
			w.Beta = 0
		case "J_waste":
			w.Gamma = 0
		case "J_slo":
			w.Delta = 0
		case "J_fairness":
			w.Epsilon = 0
		}
	}
	return w
}

// TuningConfig is the equal-budget random search.
type TuningConfig struct {
	Configurations int    `json:"configurations"`
	SearchSeed     uint64 `json:"search_seed"`
}

// QuickConfig is the -quick subset.
type QuickConfig struct {
	Scenarios []string `json:"scenarios"`
	Seeds     []uint64 `json:"seeds"`
	Jobs      int      `json:"jobs"`
	Instances int      `json:"milp_instances"`
}

// SweepConfig is the MILP solve-time sweep.
type SweepConfig struct {
	Cluster    string    `json:"cluster"`
	Workload   string    `json:"workload"`
	Jobs       []int     `json:"jobs"`
	SlotsS     []float64 `json:"slots_s"`
	Instances  int       `json:"instances"`
	NodeLimit  int       `json:"node_limit"`
	TimeLimitS float64   `json:"time_limit_s"`
}

// Experiment is the committed experiment configuration.
type Experiment struct {
	Name             string       `json:"name"`
	Baseline         string       `json:"baseline"`
	TuningSeeds      []uint64     `json:"tuning_seeds"`
	EvalSeeds        []uint64     `json:"eval_seeds"`
	Weights          Weights      `json:"weights"`
	Tuning           TuningConfig `json:"tuning"`
	DefaultsScenario string       `json:"defaults_scenario"`
	Scenarios        []Scenario   `json:"scenarios"`
	Quick            QuickConfig  `json:"quick"`
	Sweep            SweepConfig  `json:"sweep"`
	// Capacity is the capacity-planning sweep (Tier 2, optional).
	Capacity *CapacityConfig `json:"capacity,omitempty"`
	// KeyMetrics are aggregated in aggregate.csv/paired.csv (runs.csv has all).
	KeyMetrics []string `json:"key_metrics"`
	path       string
}

// LoadExperiment reads an experiment configuration (unknown fields rejected).
func LoadExperiment(path string) (*Experiment, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var e Experiment
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	e.path = path
	seen := map[string]bool{}
	for i := range e.Scenarios {
		s := &e.Scenarios[i]
		if s.Kind == "" {
			s.Kind = KindSim
		}
		if s.ID == "" || seen[s.ID] {
			return nil, fmt.Errorf("scenario ids must be unique and non-empty (%q)", s.ID)
		}
		seen[s.ID] = true
	}
	for _, s := range e.Scenarios {
		if s.TunedFrom != "" && !seen[s.TunedFrom] {
			return nil, fmt.Errorf("scenario %s: tuned_from %q is unknown", s.ID, s.TunedFrom)
		}
		for _, t := range s.ZeroWeights {
			if !slices.Contains(JTerms, t) {
				return nil, fmt.Errorf("scenario %s: zero_weights: unknown J term %q", s.ID, t)
			}
		}
	}
	if !seen[e.DefaultsScenario] {
		return nil, fmt.Errorf("defaults_scenario %q is unknown", e.DefaultsScenario)
	}
	for _, s := range e.TuningSeeds {
		if slices.Contains(e.EvalSeeds, s) {
			return nil, fmt.Errorf("tuning and evaluation seeds must be disjoint (%d in both)", s)
		}
	}
	return &e, nil
}

// Scenario returns the scenario with the given id.
func (e *Experiment) Scenario(id string) *Scenario {
	for i := range e.Scenarios {
		if e.Scenarios[i].ID == id {
			return &e.Scenarios[i]
		}
	}
	return nil
}

// Files lists every configuration file the experiment depends on (for the
// configuration hash), relative to the repository root, sorted.
func (e *Experiment) Files() []string {
	set := map[string]bool{e.path: true}
	for _, s := range e.Scenarios {
		if s.Replay == nil {
			set[s.Cluster], set[s.Workload] = true, true
		}
	}
	if e.Sweep.Cluster != "" {
		set[e.Sweep.Cluster], set[e.Sweep.Workload] = true, true
	}
	var out []string
	for f := range set {
		out = append(out, filepath.ToSlash(f))
	}
	slices.Sort(out)
	return out
}

// mergePatch applies an RFC 7386 JSON merge patch.
func mergePatch(target, patch any) any {
	pm, ok := patch.(map[string]any)
	if !ok {
		return patch
	}
	tm, ok := target.(map[string]any)
	if !ok {
		tm = map[string]any{}
	}
	for k, v := range pm {
		if v == nil {
			delete(tm, k)
		} else {
			tm[k] = mergePatch(tm[k], v)
		}
	}
	return tm
}

// LoadCluster loads the scenario's cluster with its sensitivity scaling.
func (s *Scenario) LoadCluster(root string) (*cluster.Cluster, error) {
	c, err := cluster.Load(filepath.Join(root, s.Cluster))
	if err != nil {
		return nil, err
	}
	ps, os := s.PenaltyScale, s.OverheadScale
	if ps == 0 {
		ps = 1
	}
	if os == 0 {
		os = 1
	}
	if ps != 1 || os != 1 {
		c = c.WithFactors(ps, os)
	}
	return c, nil
}

// GenConfig builds the scenario's generator configuration: the workload file
// with the overrides applied (and extra overrides, e.g. the quick job count),
// with the work capacity of the cluster.
func (s *Scenario) GenConfig(root string, c *cluster.Cluster, extra map[string]any) (trace.GenConfig, error) {
	var g trace.GenConfig
	data, err := os.ReadFile(filepath.Join(root, s.Workload))
	if err != nil {
		return g, err
	}
	var doc any
	if err := json.Unmarshal(data, &doc); err != nil {
		return g, fmt.Errorf("%s: %w", s.Workload, err)
	}
	if len(s.Override) > 0 {
		var patch any
		if err := json.Unmarshal(s.Override, &patch); err != nil {
			return g, fmt.Errorf("scenario %s overrides: %w", s.ID, err)
		}
		doc = mergePatch(doc, patch)
	}
	if extra != nil {
		doc = mergePatch(doc, extra)
	}
	b, _ := json.Marshal(doc)
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&g); err != nil {
		return g, fmt.Errorf("scenario %s workload: %w", s.ID, err)
	}
	g.WorkCapacity = c.WorkCapacity()
	return g, nil
}

// Defaults are the parts chosen on the tuning seeds.
type Defaults struct {
	Order     string `json:"order"`
	Placement string `json:"placement"`
	Backfill  string `json:"backfill"`
}

// Expand resolves placeholders and removes duplicates (first occurrence wins).
func Expand(names []string, d Defaults) []string {
	var out []string
	for _, n := range names {
		n = strings.NewReplacer("{order}", d.Order, "{placement}", d.Placement, "{backfill}", d.Backfill).Replace(n)
		if !slices.Contains(out, n) {
			out = append(out, n)
		}
	}
	return out
}

// Applies reports which J terms apply to a workload: the SLO term when some
// job class has a wait target, the fairness term with more than one tenant.
func Applies(g trace.GenConfig) (slo, fair bool) {
	for _, p := range g.Priorities {
		if p.MaxWaitS != nil {
			slo = true
		}
	}
	if g.Burst != nil && g.Burst.MaxWaitS != nil {
		slo = true
	}
	return slo, len(g.Tenants) > 1
}

// paramsFor adds the scenario's quota shares to the parameters of a +quota
// or +quota_strict policy (other policies are unchanged).
func (s *Scenario) paramsFor(policyName string, params json.RawMessage) json.RawMessage {
	if len(s.QuotaShares) == 0 || !strings.Contains(policyName, "+quota") {
		return params
	}
	m := map[string]any{}
	if len(params) > 0 {
		_ = json.Unmarshal(params, &m) // tuned parameters are a JSON object written by Tune
	}
	m["quota_shares"] = s.QuotaShares
	b, _ := json.Marshal(m)
	return b
}

// FailureConfig is the node-failure model of a scenario.
type FailureConfig struct {
	MTBFS   float64 `json:"mtbf_s"`
	RepairS float64 `json:"repair_s"`
}

// FailureSchedule draws the failures of every node up to horizon.
func (f *FailureConfig) FailureSchedule(c *cluster.Cluster, seed uint64, horizon time.Duration) []simulator.Failure {
	var out []simulator.Failure
	for _, n := range c.Nodes {
		r := rng.New(seed, "failures/"+n.Name)
		t := 0.0
		for {
			t += r.ExpFloat64() * f.MTBFS
			at := clock.RoundMs(t)
			if at >= horizon {
				break
			}
			rep := max(clock.RoundMs(r.ExpFloat64()*f.RepairS), time.Millisecond)
			out = append(out, simulator.Failure{Node: n.Name, At: at, Repair: rep})
			t += clock.Sec(rep)
		}
	}
	return out
}
