// Package placement holds the placement part of the Go baselines: given a
// gang request and the free capacity per node, choose a worker count per
// node. All routines are complete for identical workers: they find a
// placement whenever the free capacity admits one.
package placement

import (
	"cmp"
	"fmt"
	"slices"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

// Request is a gang request: per-worker resources, workers, class, topology.
type Request struct {
	GPUs     int
	CPUs     int
	MemMB    int64
	Workers  int
	Class    string
	Topology api.Topology
	// Work is the job's estimated remaining work in reference-seconds
	// (estimate - retained); only hetero_ect reads it.
	Work float64
}

// RequestOf builds the request of a pending job.
func RequestOf(p *api.PendingJob) Request {
	return Request{GPUs: p.GPUs, CPUs: p.CPUs, MemMB: p.MemMB, Workers: p.Workers, Class: p.GPUClass, Topology: p.Topology,
		Work: max(0, float64(p.Estimate.Milliseconds())/1000-p.Retained)}
}

// Cap is free capacity on one node.
type Cap struct {
	GPUs  int
	CPUs  int
	MemMB int64
}

// FreeOf copies the free capacity of the view's nodes.
func FreeOf(nodes []api.NodeState) []Cap {
	out := make([]Cap, len(nodes))
	for i := range nodes {
		out[i] = Cap{nodes[i].FreeGPUs, nodes[i].FreeCPUs, nodes[i].FreeMemMB}
	}
	return out
}

// Fits is the number of workers of r that fit into c on a node of class.
func Fits(r Request, class string, c Cap) int {
	if r.Class != "" && r.Class != class {
		return 0
	}
	w := c.GPUs / r.GPUs
	if r.CPUs > 0 {
		w = min(w, c.CPUs/r.CPUs)
	}
	if r.MemMB > 0 {
		w = min(w, int(c.MemMB/r.MemMB))
	}
	return max(w, 0)
}

// Take subtracts placement p of r from free.
func Take(free []Cap, r Request, p api.Placement) {
	for _, nw := range p {
		free[nw.Node].GPUs -= r.GPUs * nw.Workers
		free[nw.Node].CPUs -= r.CPUs * nw.Workers
		free[nw.Node].MemMB -= r.MemMB * int64(nw.Workers)
	}
}

// Give adds placement p of r to free.
func Give(free []Cap, r Request, p api.Placement) {
	for _, nw := range p {
		free[nw.Node].GPUs += r.GPUs * nw.Workers
		free[nw.Node].CPUs += r.CPUs * nw.Workers
		free[nw.Node].MemMB += r.MemMB * int64(nw.Workers)
	}
}

// Contains reports whether p of r fits into free on every node.
func Contains(free []Cap, r Request, p api.Placement) bool {
	for _, nw := range p {
		c := free[nw.Node]
		if r.GPUs*nw.Workers > c.GPUs || r.CPUs*nw.Workers > c.CPUs || r.MemMB*int64(nw.Workers) > c.MemMB {
			return false
		}
	}
	return true
}

// Placer chooses a placement. It does not modify free.
type Placer interface {
	Name() string
	Place(r Request, nodes []api.NodeState, free []Cap) (api.Placement, bool)
}

// New returns the placer of the given name. least_fragmentation reads the
// size distribution through dist at each call (the policy keeps it current).
func New(name string, dist func() *metrics.SizeDist) (Placer, error) {
	switch name {
	case "first_fit":
		return FirstFit{}, nil
	case "best_fit":
		return BestFit{}, nil
	case "least_fragmentation":
		return LeastFrag{Dist: dist}, nil
	case "hetero_ect":
		return &HeteroECT{}, nil
	case "topology_aware":
		return TopologyAware{}, nil
	}
	return nil, fmt.Errorf("unknown placement %q", name)
}

func normalize(p api.Placement) api.Placement {
	slices.SortFunc(p, func(a, b api.NodeWorkers) int { return cmp.Compare(a.Node, b.Node) })
	return p
}

// FirstFit fills nodes in index order (rack, name), as many workers per node
// as fit.
type FirstFit struct{}

// Name implements Placer.
func (FirstFit) Name() string { return "first_fit" }

// Place implements Placer.
func (FirstFit) Place(r Request, nodes []api.NodeState, free []Cap) (api.Placement, bool) {
	need := r.Workers
	var p api.Placement
	for i := range nodes {
		if w := min(Fits(r, nodes[i].Class, free[i]), need); w > 0 {
			p = append(p, api.NodeWorkers{Node: i, Workers: w})
			need -= w
			if need == 0 {
				return p, true
			}
		}
	}
	return nil, false
}

// perWorker places workers one at a time on the node with the lowest score
// (compared lexicographically; ties: lower node index).
func perWorker(r Request, nodes []api.NodeState, free []Cap, score func(c Cap) [2]float64) (api.Placement, bool) {
	c := slices.Clone(free)
	counts := map[int]int{}
	for k := 0; k < r.Workers; k++ {
		best, bestScore := -1, [2]float64{}
		for i := range nodes {
			if Fits(r, nodes[i].Class, c[i]) < 1 {
				continue
			}
			s := score(c[i])
			if best < 0 || s[0] < bestScore[0] || (s[0] == bestScore[0] && s[1] < bestScore[1]) {
				best, bestScore = i, s
			}
		}
		if best < 0 {
			return nil, false
		}
		counts[best]++
		c[best].GPUs -= r.GPUs
		c[best].CPUs -= r.CPUs
		c[best].MemMB -= r.MemMB
	}
	p := make(api.Placement, 0, len(counts))
	for n, w := range counts {
		p = append(p, api.NodeWorkers{Node: n, Workers: w})
	}
	return normalize(p), true
}

// BestFit places each worker on the node that leaves the fewest free GPUs
// after it (ties: lower index).
type BestFit struct{}

// Name implements Placer.
func (BestFit) Name() string { return "best_fit" }

// Place implements Placer.
func (BestFit) Place(r Request, nodes []api.NodeState, free []Cap) (api.Placement, bool) {
	return perWorker(r, nodes, free, func(c Cap) [2]float64 { return [2]float64{float64(c.GPUs - r.GPUs), 0} })
}

// LeastFrag places each worker on the node where it increases the expected
// stranded GPUs the least (sum over g of p(g) * (free mod g)); ties: best fit,
// then lower index. p(g) is the per-worker size distribution of the jobs the
// policy has seen.
type LeastFrag struct {
	Dist func() *metrics.SizeDist
}

// Name implements Placer.
func (LeastFrag) Name() string { return "least_fragmentation" }

// Place implements Placer.
func (l LeastFrag) Place(r Request, nodes []api.NodeState, free []Cap) (api.Placement, bool) {
	d := l.Dist()
	return perWorker(r, nodes, free, func(c Cap) [2]float64 {
		return [2]float64{d.Stranded(c.GPUs-r.GPUs) - d.Stranded(c.GPUs), float64(c.GPUs - r.GPUs)}
	})
}

// TopologyAware chooses the smallest span first: one node (best fit), else
// one rack (the fitting rack with the fewest free GPUs; inside it, nodes with
// the most room first), else the whole cluster (racks with the most room
// first, then the rack with the most room on one node). Ties: lower index.
type TopologyAware struct{}

// Name implements Placer.
func (TopologyAware) Name() string { return "topology_aware" }

// Place implements Placer.
func (TopologyAware) Place(r Request, nodes []api.NodeState, free []Cap) (api.Placement, bool) {
	// 1. one node, best fit
	best := -1
	for i := range nodes {
		if Fits(r, nodes[i].Class, free[i]) >= r.Workers {
			if best < 0 || free[i].GPUs < free[best].GPUs {
				best = i
			}
		}
	}
	if best >= 0 {
		return api.Placement{{Node: best, Workers: r.Workers}}, true
	}
	// group nodes by rack (nodes are sorted by rack, name)
	type rack struct {
		nodes     []int
		fits, gpu int
		maxNode   int // most workers that fit on one node of the rack
	}
	var racks []rack
	for i := range nodes {
		if len(racks) == 0 || nodes[racks[len(racks)-1].nodes[0]].Rack != nodes[i].Rack {
			racks = append(racks, rack{})
		}
		rk := &racks[len(racks)-1]
		rk.nodes = append(rk.nodes, i)
		f := Fits(r, nodes[i].Class, free[i])
		rk.fits += f
		rk.maxNode = max(rk.maxNode, f)
		rk.gpu += free[i].GPUs
	}
	fill := func(order []int, need int) api.Placement {
		slices.SortStableFunc(order, func(a, b int) int {
			return cmp.Compare(Fits(r, nodes[b].Class, free[b]), Fits(r, nodes[a].Class, free[a]))
		})
		var p api.Placement
		for _, i := range order {
			if w := min(Fits(r, nodes[i].Class, free[i]), need); w > 0 {
				p = append(p, api.NodeWorkers{Node: i, Workers: w})
				need -= w
				if need == 0 {
					break
				}
			}
		}
		return normalize(p)
	}
	// 2. one rack
	bestRack := -1
	for k := range racks {
		if racks[k].fits >= r.Workers && (bestRack < 0 || racks[k].gpu < racks[bestRack].gpu) {
			bestRack = k
		}
	}
	if bestRack >= 0 {
		return fill(slices.Clone(racks[bestRack].nodes), r.Workers), true
	}
	// 3. cluster: racks with the most room first (ties: the rack whose room is
	// most concentrated on one node), nodes with the most room first
	total := 0
	for _, rk := range racks {
		total += rk.fits
	}
	if total < r.Workers {
		return nil, false
	}
	idx := make([]int, len(racks))
	for k := range idx {
		idx[k] = k
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		if c := cmp.Compare(racks[b].fits, racks[a].fits); c != 0 {
			return c
		}
		return cmp.Compare(racks[b].maxNode, racks[a].maxNode)
	})
	need := r.Workers
	var p api.Placement
	for _, k := range idx {
		if need == 0 {
			break
		}
		take := min(racks[k].fits, need)
		if take == 0 {
			continue
		}
		p = append(p, fill(slices.Clone(racks[k].nodes), take)...)
		need -= take
	}
	return normalize(p), true
}
