package placement

import (
	"cmp"
	"math"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// ViewAware placements see the whole view before each invocation.
type ViewAware interface {
	Prepare(v *api.View)
}

// HeteroECT is heterogeneity-aware placement (Tier 2): for a job without a
// class constraint it estimates, per GPU class, the completion time
//
//	ECT(c) = wait(c) + estimated remaining work / speed(c)
//
// where wait(c) is 0 if the job fits on class c now, else the time until the
// running jobs on class-c nodes (released in order of estimated end) leave
// room for it. It places the job on the class with the smallest ECT (ties:
// faster class, then class name), first-fit within that class, and does not
// place it now when that class has no room yet: the job waits for the
// faster GPUs when that is expected to finish sooner. A placement never mixes
// classes. Class-constrained jobs are placed first-fit on their class.
type HeteroECT struct {
	now     time.Duration
	speed   map[string]float64
	classes []string
	ends    map[string][]release // per class, sorted by estimated end
}

type release struct {
	end   time.Duration
	node  int
	gpus  int
	cpus  int
	memMB int64
}

// Name implements Placer.
func (*HeteroECT) Name() string { return "hetero_ect" }

// Prepare records the time, the class speeds, and the estimated ends of the
// running jobs per class.
func (h *HeteroECT) Prepare(v *api.View) {
	h.now = v.Now
	h.speed = map[string]float64{}
	h.ends = map[string][]release{}
	h.classes = h.classes[:0]
	for i := range v.Nodes {
		n := &v.Nodes[i]
		if _, ok := h.speed[n.Class]; !ok {
			h.speed[n.Class] = n.Speed
			h.classes = append(h.classes, n.Class)
		}
	}
	slices.Sort(h.classes)
	for i := range v.Running {
		r := &v.Running[i]
		for _, nw := range r.Placement {
			c := v.Nodes[nw.Node].Class
			h.ends[c] = append(h.ends[c], release{r.EstEnd, nw.Node, r.GPUs * nw.Workers, r.CPUs * nw.Workers, r.MemMB * int64(nw.Workers)})
		}
	}
	for c := range h.ends {
		slices.SortFunc(h.ends[c], func(a, b release) int {
			if x := cmp.Compare(a.end, b.end); x != 0 {
				return x
			}
			return cmp.Compare(a.node, b.node)
		})
	}
}

// Place implements Placer.
func (h *HeteroECT) Place(r Request, nodes []api.NodeState, free []Cap) (api.Placement, bool) {
	if r.Class != "" || len(h.classes) <= 1 {
		return FirstFit{}.Place(r, nodes, free)
	}
	best, bestECT, bestPl := "", math.Inf(1), api.Placement(nil)
	for _, c := range h.classes {
		rc := r
		rc.Class = c
		pl, ok := FirstFit{}.Place(rc, nodes, free)
		wait := 0.0
		if !ok {
			wait = h.waitFor(rc, nodes, free)
			if math.IsInf(wait, 1) {
				continue
			}
		}
		ect := wait + r.Work/h.speed[c]
		if best == "" || ect < bestECT || (ect == bestECT && h.speed[c] > h.speed[best]) {
			best, bestECT, bestPl = c, ect, pl
		}
	}
	if best == "" || bestPl == nil {
		return nil, false
	}
	return bestPl, true
}

// waitFor is the estimated time until the request fits on its class, given
// free and the estimated ends of the running jobs on that class.
func (h *HeteroECT) waitFor(r Request, nodes []api.NodeState, free []Cap) float64 {
	capS := slices.Clone(free)
	for _, e := range h.ends[r.Class] {
		capS[e.node].GPUs += e.gpus
		capS[e.node].CPUs += e.cpus
		capS[e.node].MemMB += e.memMB
		if _, ok := (FirstFit{}).Place(r, nodes, capS); ok {
			return max(0, float64((e.end-h.now).Milliseconds())/1000)
		}
	}
	return math.Inf(1)
}
