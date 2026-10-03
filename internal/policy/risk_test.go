package policy

import (
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// The ratio model: nearest-rank quantile of runtime/estimate, the user's own
// ratios once the user has min_history completions, else all users', else 1.
func TestRatioModelQuantiles(t *testing.T) {
	m := &ratioModel{q: 0.9, minHistory: 3, user: map[string][]float64{}}
	if m.ratio("u") != 1 {
		t.Fatal("no history: ratio 1")
	}
	var h []api.CompletedJob
	for _, r := range []float64{50, 100, 150, 200, 250, 300, 350, 400, 450, 1000} { // estimate 100 -> ratios 0.5 .. 10
		h = append(h, api.CompletedJob{User: "v", Runtime: s(r), Estimate: s(100)})
	}
	h = append(h, api.CompletedJob{User: "u", Runtime: s(80), Estimate: s(100)}, api.CompletedJob{User: "u", Runtime: s(90), Estimate: s(100)})
	m.update(h)
	// u has 2 < 3 completions: all 12 ratios; P90 = ceil(0.9*12) = 11th smallest = 4.5
	if got := m.ratio("u"); got != 4.5 {
		t.Fatalf("fallback quantile %v", got)
	}
	m.update(append(h, api.CompletedJob{User: "u", Runtime: s(70), Estimate: s(100)}))
	// u now has 0.7, 0.8, 0.9: P90 = 3rd smallest = 0.9
	if got := m.ratio("u"); got != 0.9 {
		t.Fatalf("user quantile %v", got)
	}
	if got := m.work("u", 1000); got != 900 {
		t.Fatalf("work %v", got)
	}
}

// risk backfilling by hand: same setting as TestEASYByHand, but the history
// says jobs run 3x their estimate at P90. The shadow time becomes 300 (R:
// estimate 100 -> 300); D (estimate 50 -> 150) still ends before it and
// starts on n0; C (estimate 500 -> 1500) uses the extra capacity on n2.
func TestRiskBackfillByHand(t *testing.T) {
	nodes := []api.NodeState{node(0, "r", 8, 8), node(1, "r", 8, 0), node(2, "r", 8, 4)}
	run := []api.RunningJob{running("R", 1, 8, 1, api.Placement{{Node: 1, Workers: 1}}, 100),
		running("S", 1, 4, 1, api.Placement{{Node: 2, Workers: 1}}, 200)}
	run[0].Estimate, run[1].Estimate = s(100), s(200)
	v := view(nodes, run, pend("H", 1, 1, 8, 2, 10), pend("C", 2, 1, 4, 1, 500), pend("D", 3, 1, 8, 1, 50), pend("E", 4, 1, 8, 1, 500))
	for k := 0; k < 10; k++ {
		v.History = append(v.History, api.CompletedJob{User: "u", Runtime: s(300), Estimate: s(100)})
	}
	p := mustNew(t, "fifo+first_fit+risk", `{"quantile": 0.9}`)
	d, err := p.Schedule(v)
	if err != nil {
		t.Fatal(err)
	}
	if got := acts(d); got != "start:C[{2 1}] start:D[{0 1}] " {
		t.Fatalf("got %s", got)
	}
	if len(d.Reservations) != 1 || d.Reservations[0].Shadow != s(300) {
		t.Fatalf("reservation %+v (want shadow 300 = 3 x R's estimate)", d.Reservations)
	}
	// With a P90 ratio of 6, D needs 300 s > shadow 600? No: R ends at 600, D
	// (300) still fits before it. Make D long enough to cross the shadow:
	// estimate 150 -> 900 > 600, and n0 is reserved for H: D must wait.
	for k := range v.History {
		v.History[k].Runtime = s(600)
	}
	v.Pending[2].Estimate = s(150)
	p = mustNew(t, "fifo+first_fit+risk", `{"quantile": 0.9}`)
	d, _ = p.Schedule(v)
	if got := acts(d); got != "start:C[{2 1}] " {
		t.Fatalf("got %s", got)
	}
}
