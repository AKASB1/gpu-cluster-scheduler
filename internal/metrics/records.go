// Package metrics defines the per-job records a simulation run produces, the
// fragmentation measures, and the run metrics of docs/contracts.md §5. It
// does not import the simulator.
package metrics

import (
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// Segment is one run of a job (from a start to its completion or preemption).
// The W fields are speed-weighted GPU-seconds (reference-GPU-seconds).
type Segment struct {
	Start     time.Duration
	End       time.Duration
	Overhead  time.Duration // restart overhead planned for this run
	Placement api.Placement
	SMin      float64
	F         float64
	Rate      float64
	Work      float64 // work done in this run (reference-seconds, per GPU)
	Preempted bool
	Killed    bool          // ended by a node failure (Preempted is also set)
	Retained  float64       // work retained after a preemption
	Grace     time.Duration // GPUs held after a preemption

	ConsumedW float64 // all speed-weighted GPU-seconds of the run (incl. grace)
	OverheadW float64 // restart overhead
	SlackW    float64 // faster GPUs waiting for the slowest worker
	TopoW     float64 // topology slowdown
	GraceW    float64 // GPUs held during the preemption grace period
}

// JobRecord is the outcome of one job.
type JobRecord struct {
	Job          api.Job
	Infeasible   bool
	FirstStart   time.Duration
	Completion   time.Duration
	QueueTotal   time.Duration // all pending time, including after preemptions
	Preemptions  int           // policy preemptions (failure kills are counted apart)
	FailureKills int
	Segments     []Segment

	// Waste decomposition (speed-weighted GPU-seconds), see docs/simulator.md.
	ConsumedW float64
	UsefulW   float64 // runtime * gpus * workers
	LostW     float64 // progress discarded at preemptions
	OverheadW float64
	TopoW     float64
	SlackW    float64
	RoundingW float64 // progress beyond the run time due to ms rounding (tiny, >= 0)
	GraceW    float64

	AllocGPUms int64 // GPU-milliseconds allocated (raw GPUs), incl. overhead and grace

	HasShadow bool          // a backfilling policy reserved for this job as head
	Shadow    time.Duration // the first shadow time reported for it
	ShadowAt  time.Duration // when it was reported (the job was pending then)
}

// Sample is the cluster state after the policy acted at one scheduling
// instant; it holds until the next sample.
type Sample struct {
	T        time.Duration
	Alloc    int     // allocated GPUs (incl. restart overhead and grace)
	Free     int     // free GPUs
	Stranded float64 // expected stranded free GPUs
	Blocked  bool    // some pending job fits the totals but cannot be placed
	Pending  int
	Running  int
}

// LogEntry is one line of the assignment log.
type LogEntry struct {
	T         time.Duration
	Op        api.Op
	JobID     string
	Placement []NamedWorkers // start only
}

// NamedWorkers is a worker count on a named node.
type NamedWorkers struct {
	Node    string
	Workers int
}

// SolverStats aggregates the solver info of a solver-based policy.
type SolverStats struct {
	Calls      int
	Capped     int
	NotOptimal int
	WallSolve  []float64
	CPUSolve   []float64
	MaxGap     float64
}

// RunTrace is everything a run produces; metrics are computed from it.
type RunTrace struct {
	PolicyName  string
	Info        api.ClusterInfo
	FastestAny  float64 // fastest class speed in the cluster
	ClassSpeed  map[string]float64
	SizeDist    SizeDist      // per-worker GPU-size distribution of the trace
	Jobs        []JobRecord   // in trace order
	Samples     []Sample      // increasing T
	End         time.Duration // last completion
	NIntegral   int64         // integral of jobs in the system, job-milliseconds
	AllocIntMs  int64         // integral of allocated GPUs, GPU-milliseconds
	Invocations int
	WallDecide  []float64 // seconds per invocation; nil when not measured
	Solver      *SolverStats
	Log         []LogEntry
	Downtime    []Downtime
}

// Downtime is one node failure: the node had no capacity in [From, To).
type Downtime struct {
	Node     int
	GPUs     int
	From, To time.Duration
}
