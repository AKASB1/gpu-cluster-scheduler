package bench

import (
	"context"
	"math"
	"slices"
	"strconv"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

// JTerms are the terms of J in the order of docs/contracts.md §5.
var JTerms = []string{"J_slowdown", "J_wait", "J_waste", "J_slo", "J_fairness"}

// JTermCheck is the informativeness check of the J terms (benchmarks/README.md),
// on the tuning seeds only: every policy of every simulation scenario with
// its own tuning entry runs with the frozen tuning, and per scenario and term
// it reports the spread of the per-policy means across policies (max - min)
// and the seed-to-seed noise (the mean over policies of the standard
// deviation across the tuning seeds). A term is informative when the spread
// exceeds the noise; a term that is 0 for every run does not apply there.
func (env *Env) JTermCheck(ctx context.Context, exp *Experiment, tuned *Tuned) ([][]string, error) {
	var ids []string
	for i := range exp.Scenarios {
		if sc := &exp.Scenarios[i]; sc.Kind == KindSim && sc.TunedFrom == "" {
			ids = append(ids, sc.ID)
		}
	}
	tasks, _, _, err := env.simTasks(exp, tuned, EvalOptions{Scenarios: ids, Seeds: exp.TuningSeeds})
	if err != nil {
		return nil, err
	}
	env.logf("J-term check: %d runs on the tuning seeds", len(tasks))
	rows, err := env.RunTasks(ctx, tasks)
	if err != nil {
		return nil, err
	}
	type key struct{ sc, pol, term string }
	vals := map[key][]float64{}
	pols := map[string][]string{}
	for i := range rows {
		r := &rows[i]
		if !slices.Contains(pols[r.Scenario], r.Policy) {
			pols[r.Scenario] = append(pols[r.Scenario], r.Policy)
		}
		for _, t := range JTerms {
			vals[key{r.Scenario, r.Policy, t}] = append(vals[key{r.Scenario, r.Policy, t}], r.Get(t))
		}
	}
	out := [][]string{{"scenario", "term", "policies", "spread", "noise", "verdict"}}
	for _, sc := range ids {
		if len(pols[sc]) == 0 {
			continue // skipped (replay data missing)
		}
		for _, t := range JTerms {
			lo, hi, noise, n := math.Inf(1), math.Inf(-1), 0.0, 0
			allZero := true
			for _, p := range pols[sc] {
				v := vals[key{sc, p, t}]
				ci := metrics.MeanCI(v)
				lo, hi = math.Min(lo, ci.Mean), math.Max(hi, ci.Mean)
				noise += stdev(v)
				n++
				for _, x := range v {
					allZero = allZero && x == 0
				}
			}
			noise /= float64(n)
			verdict := "informative"
			switch {
			case allZero:
				verdict = "does not apply (0)"
			case hi-lo <= noise:
				verdict = "NOT informative"
			}
			out = append(out, []string{sc, t, strconv.Itoa(n), metrics.Format(hi - lo), metrics.Format(noise), verdict})
		}
	}
	return out, nil
}

func stdev(x []float64) float64 {
	if len(x) < 2 {
		return 0
	}
	m := metrics.Mean(x)
	s := 0.0
	for _, v := range x {
		s += (v - m) * (v - m)
	}
	return math.Sqrt(s / float64(len(x)-1))
}
