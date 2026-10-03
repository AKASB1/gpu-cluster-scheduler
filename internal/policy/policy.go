// Package policy defines the scheduling-policy interface, the composition of
// the Go baselines from four parts (order, placement, backfill, preemption),
// and the one factory used by the simulator, the benchmark, and the CLI.
package policy

import "github.com/AKASB1/gpu-cluster-scheduler/internal/api"

// Policy is a scheduling policy. Schedule receives a read-only view and
// returns an ordered action list. Implementations are deterministic given
// their configuration and seed; they never read the wall clock and never
// import the simulator.
type Policy interface {
	Name() string
	Schedule(v *api.View) (api.Decision, error)
}

// Closer is implemented by policies that own resources (external processes).
type Closer interface {
	Close() error
}
