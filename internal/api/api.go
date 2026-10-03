// Package api holds the types shared by the trace loader, the simulator, the
// policies, and the external-policy protocol: the job record of trace schema
// v1, the read-only scheduling view, and the actions a policy returns.
//
// All times are time.Duration values that are whole milliseconds. Work is
// float64 reference-seconds (seconds of progress on a GPU of speed 1.0).
package api

import (
	"cmp"
	"math"
	"time"
)

// Topology is the tightest placement domain a job is sensitive to.
type Topology string

// Topology sensitivities of trace schema v1.
const (
	TopoAny  Topology = "any"
	TopoRack Topology = "rack"
	TopoNode Topology = "node"
)

// Span is the domain a placement covers.
type Span int

// Placement spans, from narrowest to widest.
const (
	SpanNode Span = iota
	SpanRack
	SpanCluster
)

func (s Span) String() string {
	switch s {
	case SpanNode:
		return "node"
	case SpanRack:
		return "rack"
	default:
		return "cluster"
	}
}

// Job is one row of job trace schema v1.
type Job struct {
	ID                 string
	Submit             time.Duration
	Tenant             string
	User               string // never empty after loading: defaults to the tenant
	Priority           int
	GPUs               int // per worker
	Workers            int
	GPUClass           string // empty = any class
	Topology           Topology
	CPUs               int     // per worker
	MemGB              float64 // per worker
	Runtime            time.Duration
	Estimate           time.Duration
	Preemptible        bool
	CheckpointInterval time.Duration
	MaxWait            time.Duration
	HasMaxWait         bool
}

// TotalGPUs is the gang's GPU count.
func (j *Job) TotalGPUs() int { return j.GPUs * j.Workers }

// MemMB is the per-worker memory in whole MB. Capacity checks everywhere
// (simulator, Go and Python policies) compare whole MB, so they agree exactly.
func (j *Job) MemMB() int64 { return MB(j.MemGB) }

// MB converts GB to whole MB (0.001 GB), rounding to nearest.
func MB(gb float64) int64 { return int64(math.Round(gb * 1000)) }

// GPUClass is a class of GPUs and its speed relative to the reference class.
type GPUClass struct {
	Name  string  `json:"name"`
	Speed float64 `json:"speed"`
}

// ClusterInfo is the static part of the view.
type ClusterInfo struct {
	Classes         []GPUClass
	Racks           []string
	CrossNode       float64
	CrossRack       float64
	RestartOverhead time.Duration
	PreemptGrace    time.Duration
	TotalGPUs       int
	TotalCPUs       int
	TotalMemMB      int64
}

// NodeShare is one running job's worker count on a node.
type NodeShare struct {
	JobID   string
	Workers int
}

// NodeState is one node in the view. Nodes are sorted by (rack, name) and
// Index is the position in that order.
type NodeState struct {
	Index     int
	Name      string
	Rack      string
	Class     string
	Speed     float64
	GPUs      int
	CPUs      int
	MemMB     int64 // memory in whole MB (0.001 GB)
	FreeGPUs  int
	FreeCPUs  int
	FreeMemMB int64
	Running   []NodeShare // sorted by job ID
	Down      bool        // failed and under repair (no free capacity)
}

// NodeWorkers is the number of workers of one job on one node (by index).
type NodeWorkers struct {
	Node    int
	Workers int
}

// Placement is a worker count per node, sorted by node index, no zero entries.
type Placement []NodeWorkers

// Workers sums the worker counts.
func (p Placement) Workers() int {
	n := 0
	for _, nw := range p {
		n += nw.Workers
	}
	return n
}

// RunningJob is a running job in the view. It never contains the true run
// time.
type RunningJob struct {
	JobID              string
	Tenant             string
	User               string
	Priority           int
	GPUs               int
	Workers            int
	CPUs               int
	MemMB              int64
	GPUClass           string
	Topology           Topology
	Submit             time.Duration
	FirstStart         time.Duration
	RunStart           time.Duration // start of the current run
	Overhead           time.Duration // restart overhead of the current run
	Placement          Placement
	Rate               float64 // work-seconds per second in this run
	Estimate           time.Duration
	RetainedAtStart    float64 // work retained when this run started
	WorkDone           float64 // retained + progress of this run so far
	EstRemainingWork   float64 // max(0, estimate - WorkDone)
	EstEnd             time.Duration
	Preemptible        bool
	CheckpointInterval time.Duration
	Preemptions        int
}

// PendingJob is a queued job in the view.
type PendingJob struct {
	JobID              string
	Tenant             string
	User               string
	Submit             time.Duration
	Priority           int
	GPUs               int
	Workers            int
	CPUs               int
	MemMB              int64
	GPUClass           string
	Topology           Topology
	Estimate           time.Duration
	Wait               time.Duration // pending time so far, all pending periods
	Retained           float64       // work retained from earlier runs
	MaxWait            time.Duration
	HasMaxWait         bool
	Preemptible        bool
	CheckpointInterval time.Duration
	Preemptions        int
	Started            bool // ran before (a start now pays the restart overhead)
}

// TotalGPUs is the gang's GPU count.
func (p *PendingJob) TotalGPUs() int { return p.GPUs * p.Workers }

// TenantUsage is the usage of one tenant so far.
type TenantUsage struct {
	Tenant      string
	GPUSeconds  float64 // allocated GPU-seconds so far, including running jobs
	RunningGPUs int
	RunningCPUs int
	RunningMem  int64 // MB
}

// CompletedJob is one entry of the completion history (for predictors).
type CompletedJob struct {
	JobID    string
	Tenant   string
	User     string
	Submit   time.Duration
	Runtime  time.Duration // true run time, revealed at completion
	Estimate time.Duration
	Finish   time.Duration
}

// View is the read-only input of one policy invocation.
type View struct {
	Now     time.Duration
	Cluster *ClusterInfo
	Nodes   []NodeState
	Running []RunningJob // sorted by job ID
	Pending []PendingJob // sorted by priority desc, submit asc, job ID asc
	Tenants []TenantUsage
	// History is append-only and shared between invocations: policies must
	// not modify it. Entries are in completion order.
	History []CompletedJob
}

// Op is an action kind.
type Op string

// Action kinds.
const (
	OpStart   Op = "start"
	OpPreempt Op = "preempt"
	// OpKill appears only in the assignment log: a job killed by a node failure.
	OpKill Op = "kill"
)

// Action is one policy action.
type Action struct {
	Op        Op
	JobID     string
	Placement Placement // start only
}

// Reservation reports the shadow time a backfilling policy computed for the
// head of its queue (used for the reservation-guarantee check).
type Reservation struct {
	JobID  string
	Shadow time.Duration
}

// SolverInfo carries solver statistics of one invocation (wall-clock values
// are reported only in wall_ columns).
type SolverInfo struct {
	Status    string
	WallSolve float64
	CPUSolve  float64 // process CPU time of the solve (seconds), next to the wall time
	Gap       float64
	Capped    bool
}

// Decision is the answer of one policy invocation. Actions apply in order.
type Decision struct {
	Actions      []Action
	WakeAt       time.Duration // > Now: ask for an invocation at that time; 0 = none
	Reservations []Reservation
	Solver       *SolverInfo
}

// PendingLess is the canonical pending order: priority desc, submit asc, ID asc.
func PendingLess(a, b *PendingJob) int {
	if c := cmp.Compare(b.Priority, a.Priority); c != 0 {
		return c
	}
	if c := cmp.Compare(a.Submit, b.Submit); c != 0 {
		return c
	}
	return cmp.Compare(a.JobID, b.JobID)
}
