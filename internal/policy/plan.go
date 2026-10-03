package policy

import (
	"cmp"
	"fmt"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// PlannedStart is one start of a fixed plan (from the offline MILP).
type PlannedStart struct {
	JobID     string        `json:"job_id"`
	StartS    float64       `json:"start_s"`
	Placement []NodeWorkers `json:"placement"`
}

// NodeWorkers names a node and a worker count.
type NodeWorkers struct {
	Node    string `json:"node"`
	Workers int    `json:"workers"`
}

// PlanParams are the parameters of the plan policy.
type PlanParams struct {
	Starts []PlannedStart `json:"starts"`
}

// Plan replays a fixed plan: every job starts exactly at its planned time on
// its planned nodes. It asks for a wake-up at the next planned start. A
// planned start that is missed (the policy was not invoked at that instant,
// or the job is not pending) is an error.
type Plan struct {
	starts []PlannedStart
	at     []time.Duration
	next   int
}

// NewPlan builds the plan policy.
func NewPlan(starts []PlannedStart) (*Plan, error) {
	p := &Plan{starts: slices.Clone(starts)}
	for _, s := range p.starts {
		if s.StartS < 0 {
			return nil, fmt.Errorf("plan: negative start for %s", s.JobID)
		}
	}
	slices.SortStableFunc(p.starts, func(a, b PlannedStart) int {
		if c := cmp.Compare(a.StartS, b.StartS); c != 0 {
			return c
		}
		return cmp.Compare(a.JobID, b.JobID)
	})
	for _, s := range p.starts {
		p.at = append(p.at, clock.RoundMs(s.StartS))
	}
	return p, nil
}

// Name implements Policy.
func (*Plan) Name() string { return "plan" }

// Schedule implements Policy.
func (p *Plan) Schedule(v *api.View) (api.Decision, error) {
	var d api.Decision
	idx := map[string]int{}
	for i := range v.Nodes {
		idx[v.Nodes[i].Name] = i
	}
	for p.next < len(p.starts) && p.at[p.next] <= v.Now {
		s := p.starts[p.next]
		if p.at[p.next] < v.Now {
			return d, fmt.Errorf("plan: start of %s at %ss was missed (now %ss)", s.JobID,
				clock.FormatSeconds(p.at[p.next]), clock.FormatSeconds(v.Now))
		}
		var pl api.Placement
		for _, nw := range s.Placement {
			n, ok := idx[nw.Node]
			if !ok {
				return d, fmt.Errorf("plan: unknown node %q", nw.Node)
			}
			pl = append(pl, api.NodeWorkers{Node: n, Workers: nw.Workers})
		}
		d.Actions = append(d.Actions, api.Action{Op: api.OpStart, JobID: s.JobID, Placement: pl})
		p.next++
	}
	if p.next < len(p.starts) {
		d.WakeAt = p.at[p.next]
	}
	return d, nil
}
