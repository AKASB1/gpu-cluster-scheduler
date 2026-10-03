package api

import (
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// SpanOf returns the span of a placement over the view's nodes.
func SpanOf(p Placement, nodes []NodeState) Span {
	if len(p) <= 1 {
		return SpanNode
	}
	rack := nodes[p[0].Node].Rack
	for _, nw := range p[1:] {
		if nodes[nw.Node].Rack != rack {
			return SpanCluster
		}
	}
	return SpanRack
}

// TopologyFactor is f of the simulator model: 1 when the span is allowed by
// the job's sensitivity, crossNode for a node-sensitive job over several nodes
// of one rack, crossRack for a node- or rack-sensitive job over several racks.
func TopologyFactor(t Topology, s Span, crossNode, crossRack float64) float64 {
	switch {
	case s == SpanNode || t == TopoAny:
		return 1
	case s == SpanRack && t == TopoNode:
		return crossNode
	case s == SpanRack:
		return 1
	default: // cluster span, node- or rack-sensitive
		return crossRack
	}
}

// RateOf returns the progress rate (work-seconds per second) of a job with
// topology t placed by p: min speed of the occupied classes divided by f. It
// also returns that min speed and f.
func RateOf(t Topology, p Placement, nodes []NodeState, c *ClusterInfo) (rate, sMin, f float64) {
	sMin = 0
	for i, nw := range p {
		s := nodes[nw.Node].Speed
		if i == 0 || s < sMin {
			sMin = s
		}
	}
	f = TopologyFactor(t, SpanOf(p, nodes), c.CrossNode, c.CrossRack)
	return sMin / f, sMin, f
}

// EstimatedDuration is the time a start with placement p is expected to take
// when the job needs work more reference-seconds and pays overhead first.
func EstimatedDuration(work float64, overhead time.Duration, rate float64) time.Duration {
	return overhead + clock.WorkToDuration(work, rate)
}
