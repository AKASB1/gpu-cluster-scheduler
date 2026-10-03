package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

// EvalOptions select what an evaluation runs.
type EvalOptions struct {
	Scenarios []string // nil: all
	Seeds     []uint64 // override evaluation seeds (quick)
	Instances int      // override MILP instance count (quick)
	Sweep     bool     // run the MILP solve-time sweep
	Capacity  bool     // run the capacity-planning sweep
}

func (o EvalOptions) selected(id string) bool {
	return o.Scenarios == nil || slices.Contains(o.Scenarios, id)
}

// Evaluate runs the evaluation with the frozen tuning and writes every
// result file into out.
func (env *Env) Evaluate(ctx context.Context, exp *Experiment, tuned *Tuned, opts EvalOptions, out string) error {
	tasks, groupOf, milpScs, err := env.simTasks(exp, tuned, opts)
	if err != nil {
		return err
	}
	env.logf("evaluation: %d simulation runs on %d workers", len(tasks), env.Workers)
	rows, err := env.RunTasks(ctx, tasks)
	if err != nil {
		return err
	}
	byGroup := map[string][]Row{}
	for _, r := range rows {
		byGroup[groupOf[r.Scenario]] = append(byGroup[groupOf[r.Scenario]], r)
	}
	for _, sc := range milpScs {
		r, err := env.runMILPScenario(ctx, exp, tuned, sc, opts, filepath.Join(out, sc.Group))
		if err != nil {
			return err
		}
		byGroup[sc.Group] = append(byGroup[sc.Group], r...)
	}
	keys := append([]string{"J", "J_slowdown", "J_wait", "J_waste", "J_slo", "J_fairness", "gap"}, exp.KeyMetrics...)
	var groups []string
	for g := range byGroup {
		groups = append(groups, g)
	}
	sort.Strings(groups)
	for _, g := range groups {
		dir := filepath.Join(out, g)
		if err := WriteRows(dir, byGroup[g]); err != nil {
			return err
		}
		aggs, pairs := Summarize(byGroup[g], exp.Baseline, keys)
		if err := WriteSummaries(dir, aggs, pairs); err != nil {
			return err
		}
	}
	if err := writeSensitivity(exp, rows, filepath.Join(out, "sensitivity.csv")); err != nil {
		return err
	}
	if opts.Sweep {
		if err := env.runSweep(ctx, exp, opts, filepath.Join(out, "S8")); err != nil {
			return err
		}
	}
	if opts.Capacity {
		if err := env.runCapacity(ctx, exp, tuned, opts, filepath.Join(out, "CAP")); err != nil {
			return err
		}
	}
	return nil
}

// simTasks builds the simulation runs of the selected scenarios (and lists
// the MILP scenarios) with the frozen tuning.
func (env *Env) simTasks(exp *Experiment, tuned *Tuned, opts EvalOptions) ([]Task, map[string]string, []*Scenario, error) {
	var tasks []Task
	groupOf := map[string]string{}
	var milpScs []*Scenario
	for i := range exp.Scenarios {
		sc := &exp.Scenarios[i]
		if !opts.selected(sc.ID) {
			continue
		}
		groupOf[sc.ID] = sc.Group
		if sc.Kind != KindSim {
			milpScs = append(milpScs, sc)
			continue
		}
		st, err := tuned.For(sc)
		if err != nil {
			return nil, nil, nil, err
		}
		seeds := exp.EvalSeeds
		if len(sc.Seeds) > 0 {
			seeds = sc.Seeds
		}
		if opts.Seeds != nil {
			seeds = opts.Seeds
		}
		ts, err := env.Traces(sc, seeds)
		if errors.Is(err, ErrReplayData) {
			env.logf("%s skipped: %v", sc.ID, err)
			env.Skipped = append(env.Skipped, sc.ID)
			continue
		}
		if err != nil {
			return nil, nil, nil, err
		}
		pols := Expand(append([]string{exp.Baseline}, sc.Policies...), tuned.Defaults)
		for _, pol := range pols {
			for _, seed := range seeds {
				tasks = append(tasks, Task{Scenario: sc.ID, Policy: pol, Params: sc.paramsFor(pol, st.Params[pol]), Seed: seed, Trace: ts[seed], Shares: sc.QuotaShares,
					Ref: st.Reference, Weights: sc.weights(exp.Weights)})
			}
		}
	}
	return tasks, groupOf, milpScs, nil
}

func ptr(f *float64) float64 {
	if f == nil {
		return math.NaN()
	}
	return *f
}

// runMILPScenario runs S8: every Go baseline on whole-slot instances, the
// offline MILP (node model: exact optimum; pool model: lower bound), the
// check-7 assertions on every instance, and the gaps.
func (env *Env) runMILPScenario(ctx context.Context, exp *Experiment, tuned *Tuned, sc *Scenario, opts EvalOptions, dir string) ([]Row, error) {
	seeds := sc.Seeds
	if opts.Instances > 0 && opts.Instances < len(seeds) {
		seeds = seeds[:opts.Instances]
	}
	ts, err := env.Traces(sc, seeds)
	if err != nil {
		return nil, err
	}
	pols := Expand(append([]string{exp.Baseline}, sc.Policies...), tuned.Defaults)
	var tasks []Task
	for _, pol := range pols {
		for _, seed := range seeds {
			tasks = append(tasks, Task{Scenario: sc.ID, Policy: pol, Seed: seed, Trace: ts[seed], Ref: Reference{1, 1}})
		}
	}
	env.logf("%s: %d policy runs on %d instances", sc.ID, len(tasks), len(seeds))
	rows, err := env.RunTasks(ctx, tasks)
	if err != nil {
		return nil, err
	}
	for i := range rows {
		rows[i].Extra = nil // J is not defined for the MILP instances
	}
	model := "node"
	if sc.Kind == KindMILPMedium {
		model = "pool"
	}
	clAbs, _ := filepath.Abs(filepath.Join(env.Root, sc.Cluster))
	var mt []MILPTask
	for _, seed := range seeds {
		p, _ := filepath.Abs(filepath.Join(env.Outputs, "traces", sc.ID, fmt.Sprintf("seed-%d.csv", seed)))
		mt = append(mt, MILPTask{ID: strconv.FormatUint(seed, 10), Trace: p, Cluster: clAbs, Model: model, SlotS: sc.SlotS,
			NodeLimit: sc.NodeLimit, TimeLimitS: sc.TimeLimitS})
	}
	env.logf("%s: solving %d %s-model MILPs", sc.ID, len(mt), model)
	res, err := SolveOffline(ctx, mt, filepath.Join(env.Outputs, "milp", sc.ID), filepath.Join(env.Outputs, "logs", sc.ID, "milp.stderr.log"))
	if err != nil {
		return nil, err
	}
	refOf := map[uint64]float64{}
	var inst []Row
	for k, r := range res {
		seed := seeds[k]
		if !r.WholeSlot {
			return nil, fmt.Errorf("%s instance %d is not whole-slot: no bound may be claimed", sc.ID, seed)
		}
		ref := ptr(r.Bound)
		if r.Status == "optimal" {
			ref = ptr(r.Objective)
		}
		if math.IsNaN(ref) {
			return nil, fmt.Errorf("%s instance %d: solver status %s without a bound", sc.ID, seed, r.Status)
		}
		refOf[seed] = ref
		if model == "node" && r.Status == "optimal" {
			// check 7 on every instance: the plan replayed in the simulator reproduces the optimum
			plan, err := policy.NewPlan(r.Starts)
			if err != nil {
				return nil, err
			}
			rt, err := simulator.Run(simulator.Config{Cluster: ts[seed].Cluster, Jobs: ts[seed].Jobs, Policy: plan})
			if err != nil {
				return nil, fmt.Errorf("%s instance %d: plan replay: %w", sc.ID, seed, err)
			}
			got := metrics.Compute(rt).Get("wflow_sum_all")
			if math.Abs(got-ref) > 1e-6*math.Max(1, ref) {
				return nil, fmt.Errorf("%s instance %d: replayed objective %.9f != MILP optimum %.9f (check 7)", sc.ID, seed, got, ref)
			}
		}
		inst = append(inst, Row{Scenario: sc.ID, Policy: "milp_" + model, Seed: seed, Fields: []metrics.Field{
			{Name: "jobs", Value: float64(r.Jobs)}, {Name: "optimal", Value: b2f(r.Status == "optimal")},
			{Name: "objective", Value: ptr(r.Objective)}, {Name: "bound", Value: ptr(r.Bound)}, {Name: "solver_gap", Value: ptr(r.Gap)},
			{Name: "reference", Value: ref}, {Name: "variables", Value: float64(r.Variables)}, {Name: "horizon_slots", Value: float64(r.Horizon)},
			{Name: "solver_capped", Value: b2f(r.Capped)}},
			Wall: []metrics.Field{{Name: "wall_solve_s", Value: r.WallSolveS}, {Name: "wall_cpu_solve_s", Value: ptr(r.CPUSolveS)}, {Name: "wall_nodes", Value: float64(r.Nodes)}}})
	}
	for i := range rows {
		r := &rows[i]
		w, ref := r.Get("wflow_sum_all"), refOf[r.Seed]
		if w < ref-1e-6*math.Max(1, ref) {
			return nil, fmt.Errorf("%s instance %d: %s reaches %.6f below the MILP reference %.6f: a bug in one of them (check 7)",
				sc.ID, r.Seed, r.Policy, w, ref)
		}
		r.Extra = []metrics.Field{{Name: "milp_reference", Value: ref}, {Name: "gap", Value: (w - ref) / ref}}
	}
	if err := WriteRows(filepath.Join(dir, "milp_"+sc.ID), inst); err != nil {
		return nil, err
	}
	return rows, nil
}

// runSweep measures MILP solve time against the number of jobs and the grid
// size (node model, whole-slot instances solved on several slot lengths).
func (env *Env) runSweep(ctx context.Context, exp *Experiment, opts EvalOptions, dir string) error {
	sw := exp.Sweep
	if len(sw.Jobs) == 0 {
		return nil
	}
	inst := sw.Instances
	if opts.Instances > 0 {
		inst = min(inst, 1)
	}
	var mt []MILPTask
	type meta struct {
		n    int
		slot float64
		seed uint64
	}
	var metas []meta
	nl := sw.NodeLimit
	for _, n := range sw.Jobs {
		sc := &Scenario{ID: fmt.Sprintf("sweep-n%d", n), Kind: KindMILPSmall, Cluster: sw.Cluster, Workload: sw.Workload,
			Override: json.RawMessage(fmt.Sprintf(`{"jobs": %d}`, n))}
		var seeds []uint64
		for s := 1; s <= inst; s++ {
			seeds = append(seeds, uint64(s))
		}
		if _, err := env.Traces(sc, seeds); err != nil {
			return err
		}
		clAbs, _ := filepath.Abs(filepath.Join(env.Root, sw.Cluster))
		for _, slot := range sw.SlotsS {
			for _, seed := range seeds {
				p, _ := filepath.Abs(filepath.Join(env.Outputs, "traces", sc.ID, fmt.Sprintf("seed-%d.csv", seed)))
				mt = append(mt, MILPTask{ID: fmt.Sprintf("n%d-s%g-%d", n, slot, seed), Trace: p, Cluster: clAbs, Model: "node",
					SlotS: slot, NodeLimit: &nl, TimeLimitS: sw.TimeLimitS})
				metas = append(metas, meta{n, slot, seed})
			}
		}
	}
	env.logf("sweep: %d MILP solves", len(mt))
	res, err := SolveOffline(ctx, mt, filepath.Join(env.Outputs, "milp", "sweep"), filepath.Join(env.Outputs, "logs", "sweep.stderr.log"))
	if err != nil {
		return err
	}
	var rows []Row
	optimum := map[[2]uint64]float64{}
	for k, r := range res {
		m := metas[k]
		row := Row{Scenario: "sweep", Policy: fmt.Sprintf("n%03d-slot%04g", m.n, m.slot), Seed: m.seed,
			Fields: []metrics.Field{{Name: "jobs", Value: float64(m.n)}, {Name: "slot_s", Value: m.slot},
				{Name: "variables", Value: float64(r.Variables)}, {Name: "horizon_slots", Value: float64(r.Horizon)},
				{Name: "optimal", Value: b2f(r.Status == "optimal")}, {Name: "objective", Value: ptr(r.Objective)},
				{Name: "solver_capped", Value: b2f(r.Capped)}},
			Wall: []metrics.Field{{Name: "wall_solve_s", Value: r.WallSolveS}, {Name: "wall_cpu_solve_s", Value: ptr(r.CPUSolveS)}, {Name: "wall_nodes", Value: float64(r.Nodes)}}}
		rows = append(rows, row)
		// the same whole-slot instance has the same optimum on every grid
		if r.Status == "optimal" {
			key := [2]uint64{uint64(m.n), m.seed}
			if o, ok := optimum[key]; ok && math.Abs(o-ptr(r.Objective)) > 1e-6*math.Max(1, o) {
				return fmt.Errorf("sweep: instance n=%d seed %d has optimum %.9f on one grid and %.9f on another", m.n, m.seed, o, ptr(r.Objective))
			}
			optimum[key] = ptr(r.Objective)
		}
	}
	return WriteRows(filepath.Join(dir, "sweep"), rows)
}

// writeSensitivity compares each sensitivity variant with its base scenario:
// the best policy by mean J, Kendall's tau between the rankings, and the
// share of significant J differences against the baseline whose sign holds.
func writeSensitivity(exp *Experiment, rows []Row, path string) error {
	byScen := map[string][]Row{}
	for _, r := range rows {
		byScen[r.Scenario] = append(byScen[r.Scenario], r)
	}
	recs := [][]string{{"base", "variant", "assumption", "scale", "policies", "best_base", "best_variant", "same_best",
		"kendall_tau", "significant_in_base", "sign_kept"}}
	// best_* are the best deployable policies by mean J; kendall_tau ranks all policies (oracles included).
	for _, sc := range exp.Scenarios {
		if sc.TunedFrom == "" || byScen[sc.ID] == nil || byScen[sc.TunedFrom] == nil {
			continue
		}
		base, vari := meanJ(byScen[sc.TunedFrom]), meanJ(byScen[sc.ID])
		var pols []string
		for p := range vari {
			if _, ok := base[p]; ok {
				pols = append(pols, p)
			}
		}
		sort.Strings(pols)
		// the best deployable policy: oracles are context, not candidates
		var deployable []string
		for _, p := range pols {
			if !strings.Contains(p, "oracle") {
				deployable = append(deployable, p)
			}
		}
		bestB, bestV := argmin(base, deployable), argmin(vari, deployable)
		_, pb := Summarize(byScen[sc.TunedFrom], exp.Baseline, []string{"J"})
		_, pv := Summarize(byScen[sc.ID], exp.Baseline, []string{"J"})
		sig, kept := 0, 0
		for _, a := range pb {
			if a.Diff.N < 2 || math.Abs(a.Diff.Mean) <= a.Diff.Half {
				continue
			}
			sig++
			for _, b := range pv {
				if b.Policy == a.Policy && math.Signbit(a.Diff.Mean) == math.Signbit(b.Diff.Mean) && math.Abs(b.Diff.Mean) > b.Diff.Half {
					kept++
				}
			}
		}
		recs = append(recs, []string{sc.TunedFrom, sc.ID, sc.Assumption, metrics.Format(sc.Scale), strconv.Itoa(len(pols)), bestB, bestV,
			strconv.FormatBool(bestB == bestV), metrics.Format(kendall(base, vari, pols)), strconv.Itoa(sig), strconv.Itoa(kept)})
	}
	if len(recs) == 1 {
		return nil
	}
	return writeCSV(path, recs)
}

func argmin(m map[string]float64, keys []string) string {
	best := ""
	for _, k := range keys {
		if best == "" || m[k] < m[best] {
			best = k
		}
	}
	return best
}

// kendall is Kendall's tau-a between two scorings of the same keys.
func kendall(a, b map[string]float64, keys []string) float64 {
	c, d := 0, 0
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			x := a[keys[i]] - a[keys[j]]
			y := b[keys[i]] - b[keys[j]]
			switch {
			case x*y > 0:
				c++
			case x*y < 0:
				d++
			}
		}
	}
	n := len(keys) * (len(keys) - 1) / 2
	if n == 0 {
		return math.NaN()
	}
	return float64(c-d) / float64(n)
}
