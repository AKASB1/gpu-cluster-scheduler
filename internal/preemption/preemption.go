// Package preemption holds the preemption part of the Go baselines:
// priority preemption with victims chosen to minimize lost work.
package preemption

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/placement"
)

// Params configure priority preemption.
type Params struct {
	// MinGap: a victim's priority must be at most the preemptor's minus MinGap.
	MinGap int `json:"min_gap,omitempty"`
	// MaxVictims bounds the victims per preemptor.
	MaxVictims int `json:"max_victims,omitempty"`
}

// Defaults fills unset parameters (MinGap 1, MaxVictims 8).
func (p Params) Defaults() (Params, error) {
	if p.MinGap == 0 {
		p.MinGap = 1
	}
	if p.MaxVictims == 0 {
		p.MaxVictims = 8
	}
	if p.MinGap < 1 || p.MaxVictims < 1 {
		return p, fmt.Errorf("min_gap and max_victims must be >= 1")
	}
	return p, nil
}

// Retained is the work a job keeps when preempted after doing work
// reference-seconds with checkpoint interval ck: work rounded down to a
// multiple of ck (1e-9 relative tolerance), capped at work; 0 without
// checkpoints. The simulator uses the same rule.
func Retained(work float64, ck float64) float64 {
	if ck <= 0 {
		return 0
	}
	return math.Min(math.Floor(work/ck+1e-9)*ck, work)
}

// Lost is the work (GPU x reference-seconds) a running job would lose if it
// were preempted now.
func Lost(r *api.RunningJob) float64 {
	ret := Retained(r.WorkDone, clock.Sec(r.CheckpointInterval))
	return float64(r.GPUs*r.Workers) * (r.WorkDone - ret)
}

func release(free []placement.Cap, r *api.RunningJob, sign int) {
	for _, nw := range r.Placement {
		free[nw.Node].GPUs += sign * r.GPUs * nw.Workers
		free[nw.Node].CPUs += sign * r.CPUs * nw.Workers
		free[nw.Node].MemMB += int64(sign) * r.MemMB * int64(nw.Workers)
	}
}

// Select chooses victims among running jobs (excluding those in exclude) for
// a request of priority prio that does not fit free. Candidates are
// preemptible jobs with priority <= prio - MinGap, ordered by lost work
// (ascending), then priority (ascending), then start of the current run
// (latest first), then job ID. Victims are added in that order until the
// request fits (at most MaxVictims); then each victim, last added first, is
// dropped again if the request still fits without it. It returns the victim
// indices into running, in the order chosen, and the placement on the freed
// capacity. free is not modified.
func Select(r placement.Request, prio int, running []api.RunningJob, exclude map[string]bool,
	nodes []api.NodeState, free []placement.Cap, pl placement.Placer, p Params) ([]int, api.Placement, bool) {
	return SelectWith(r, func(rj *api.RunningJob) bool { return rj.Priority <= prio-p.MinGap }, running, exclude, nodes, free, pl, p.MaxVictims)
}

// SelectWith is Select with an arbitrary eligibility rule for victims (quota
// reclaim uses it); victims must also be preemptible and not excluded.
func SelectWith(r placement.Request, eligible func(*api.RunningJob) bool, running []api.RunningJob, exclude map[string]bool,
	nodes []api.NodeState, free []placement.Cap, pl placement.Placer, maxVictims int) ([]int, api.Placement, bool) {
	p := Params{MaxVictims: maxVictims}
	var cand []int
	lost := make([]float64, len(running))
	for i := range running {
		rj := &running[i]
		if rj.Preemptible && !exclude[rj.JobID] && eligible(rj) {
			cand = append(cand, i)
			lost[i] = Lost(rj)
		}
	}
	if len(cand) == 0 {
		return nil, nil, false
	}
	slices.SortFunc(cand, func(a, b int) int {
		if c := cmp.Compare(lost[a], lost[b]); c != 0 {
			return c
		}
		ra, rb := &running[a], &running[b]
		if c := cmp.Compare(ra.Priority, rb.Priority); c != 0 {
			return c
		}
		if c := cmp.Compare(rb.RunStart, ra.RunStart); c != 0 {
			return c
		}
		return cmp.Compare(ra.JobID, rb.JobID)
	})
	hyp := slices.Clone(free)
	var chosen []int
	fits := false
	for _, i := range cand {
		if len(chosen) == p.MaxVictims {
			break
		}
		release(hyp, &running[i], +1)
		chosen = append(chosen, i)
		if _, ok := pl.Place(r, nodes, hyp); ok {
			fits = true
			break
		}
	}
	if !fits {
		return nil, nil, false
	}
	for k := len(chosen) - 1; k >= 0; k-- {
		release(hyp, &running[chosen[k]], -1)
		if _, ok := pl.Place(r, nodes, hyp); ok {
			chosen = slices.Delete(chosen, k, k+1)
		} else {
			release(hyp, &running[chosen[k]], +1)
		}
	}
	pla, _ := pl.Place(r, nodes, hyp)
	return chosen, pla, true
}

// Release adds a running job's resources to free (after it is preempted).
func Release(free []placement.Cap, r *api.RunningJob) { release(free, r, +1) }
