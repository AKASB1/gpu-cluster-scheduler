package simulator

import (
	"cmp"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/preemption"
)

func (s *sim) anyDown() bool {
	for _, d := range s.downTill {
		if d > 0 {
			return true
		}
	}
	return false
}

// repairsAndFailures applies the repairs and then the failures due at t.
func (s *sim) repairsAndFailures(t time.Duration) {
	for i, d := range s.downTill {
		if d > 0 && d == t {
			n := &s.nodes[i]
			n.FreeGPUs, n.FreeCPUs, n.FreeMemMB = n.GPUs, n.CPUs, n.MemMB
			s.downTill[i] = 0
		}
	}
	for s.nextFail < len(s.fails) && s.fails[s.nextFail].at == t {
		f := s.fails[s.nextFail]
		s.nextFail++
		if s.downTill[f.node] > 0 {
			continue // already down: the failure is absorbed by the running repair
		}
		s.failNode(f.node, t+f.repair)
	}
	// failures in the past (nodes repaired only later) are skipped by the loop
	for s.nextFail < len(s.fails) && s.fails[s.nextFail].at < t {
		s.nextFail++
	}
}

// failNode kills every job with a worker on node n, then takes the node down
// until the repair time.
func (s *sim) failNode(n int, until time.Duration) {
	var victims []*jobState
	for id := range s.shares[n] {
		victims = append(victims, s.byID[id])
	}
	slices.SortFunc(victims, func(a, b *jobState) int { return cmp.Compare(a.job.ID, b.job.ID) })
	for _, js := range victims {
		s.kill(js)
	}
	// grace-held GPUs of a preempted job on this node are released by their own event
	node := &s.nodes[n]
	s.out.Downtime = append(s.out.Downtime, metrics.Downtime{Node: n, GPUs: node.GPUs, From: s.now, To: until})
	node.FreeGPUs, node.FreeCPUs, node.FreeMemMB = 0, 0, 0
	s.downTill[n] = until
}

// kill ends a run because its node failed: like a preemption (checkpoint
// rule, lost work, restart overhead at the next start) but involuntary, so
// it applies to non-preemptible jobs too; the GPUs are freed at once.
func (s *sim) kill(js *jobState) {
	j := js.job
	work := js.retainedRun + s.progress(js, s.now)
	retained := preemption.Retained(work, clock.Sec(j.CheckpointInterval))
	seg := s.closeSegment(js, s.now)
	seg.Preempted, seg.Killed, seg.Retained = true, true, retained
	js.rec.LostW += float64(j.TotalGPUs()) * (work - retained)
	js.retained = retained
	js.preemptions++
	js.rec.FailureKills++
	s.free(j, js.memMB, js.placement)
	s.alloc -= j.TotalGPUs()
	js.rec.Segments = append(js.rec.Segments, seg)
	delete(s.running, j.ID)
	s.tenantStop(js)
	js.pendingSince = s.now
	js.st = stPending
	s.pending = append(s.pending, js)
	s.logEntry(api.OpKill, j.ID, nil)
}
