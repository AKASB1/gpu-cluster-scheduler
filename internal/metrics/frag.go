package metrics

import (
	"cmp"
	"slices"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// SizeDist is a per-worker GPU-size distribution p(g).
type SizeDist struct {
	Sizes []int     // ascending
	Probs []float64 // same length, sums to 1
}

// SizeDistOf builds p(g) from jobs, weighting each job by its worker count.
func SizeDistOf(jobs []api.Job) SizeDist {
	count := map[int]float64{}
	tot := 0.0
	for i := range jobs {
		count[jobs[i].GPUs] += float64(jobs[i].Workers)
		tot += float64(jobs[i].Workers)
	}
	var d SizeDist
	for g := range count {
		d.Sizes = append(d.Sizes, g)
	}
	slices.Sort(d.Sizes)
	for _, g := range d.Sizes {
		d.Probs = append(d.Probs, count[g]/tot)
	}
	return d
}

// Stranded is the expected number of stranded GPUs on a node with free free
// GPUs: sum over g of p(g) * (free mod g) (free itself when free < g).
func (d SizeDist) Stranded(free int) float64 {
	s := 0.0
	for i, g := range d.Sizes {
		s += d.Probs[i] * float64(free%g)
	}
	return s
}

// Shape is a gang request: per-worker resources, worker count, class.
type Shape struct {
	GPUs    int
	CPUs    int
	MemMB   int64
	Workers int
	Class   string
}

// MaxWorkers is the number of workers of shape s that fit on node n's free
// resources (0 if the class does not match). Workers are identical, so a
// gang is placeable exactly when the sum over nodes reaches its worker count.
func MaxWorkers(s Shape, n *api.NodeState) int {
	if s.Class != "" && s.Class != n.Class {
		return 0
	}
	w := n.FreeGPUs / s.GPUs
	if s.CPUs > 0 {
		w = min(w, n.FreeCPUs/s.CPUs)
	}
	if s.MemMB > 0 {
		w = min(w, int(n.FreeMemMB/s.MemMB))
	}
	return w
}

// Classify reports whether shape s fits the free totals of the eligible
// nodes (GPU, CPU, memory, class) and whether it can actually be placed.
func Classify(s Shape, nodes []api.NodeState) (totalsFit, placeable bool) {
	var g, c, w int
	var m int64
	for i := range nodes {
		n := &nodes[i]
		if s.Class != "" && s.Class != n.Class {
			continue
		}
		g += n.FreeGPUs
		c += n.FreeCPUs
		m += n.FreeMemMB
		w += MaxWorkers(s, n)
	}
	totalsFit = g >= s.GPUs*s.Workers && c >= s.CPUs*s.Workers && m >= s.MemMB*int64(s.Workers)
	return totalsFit, w >= s.Workers
}

// Blocked reports whether some shape fits the totals but cannot be placed.
// shapes is deduplicated and sorted by the caller or here.
func Blocked(shapes []Shape, nodes []api.NodeState) bool {
	for _, s := range shapes {
		if fit, ok := Classify(s, nodes); fit && !ok {
			return true
		}
	}
	return false
}

// UniqueShapes deduplicates shapes in a fixed order.
func UniqueShapes(shapes []Shape) []Shape {
	slices.SortFunc(shapes, func(a, b Shape) int {
		if c := cmp.Compare(a.GPUs, b.GPUs); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Workers, b.Workers); c != 0 {
			return c
		}
		if c := cmp.Compare(a.CPUs, b.CPUs); c != 0 {
			return c
		}
		if c := cmp.Compare(a.MemMB, b.MemMB); c != 0 {
			return c
		}
		return cmp.Compare(a.Class, b.Class)
	})
	return slices.Compact(shapes)
}
