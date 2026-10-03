package policy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Spec is a policy configuration: {"name": "...", "params": {...}}.
type Spec struct {
	Name   string          `json:"name"`
	Params json.RawMessage `json:"params,omitempty"`
}

// Env carries what some policies need beyond their configuration.
type Env struct {
	// Oracle returns the true run time of a job (benchmark only; oracles are
	// labelled and never deployable).
	Oracle func(jobID string) (time.Duration, bool)
	// External builds a policy that runs in another process (names "py:...").
	External func(spec Spec) (Policy, error)
}

// New is the one factory for the simulator, the benchmark, and the CLI.
//
// Names: "order+placement+backfill[+preempt]" for the Go baselines (order:
// fifo, priority, shortest_estimate, drf; placement: first_fit, best_fit,
// least_fragmentation, topology_aware; backfill: none, easy, conservative,
// easy_predicted, easy_oracle); "plan" for the fixed-plan replay; "py:<name>"
// for a policy of the Python harness through the external-policy protocol.
func New(spec Spec, env Env) (Policy, error) {
	switch {
	case strings.HasPrefix(spec.Name, "py:"):
		if env.External == nil {
			return nil, fmt.Errorf("policy %q: no external-policy runner configured", spec.Name)
		}
		return env.External(spec)
	case spec.Name == "plan":
		var pp PlanParams
		if err := decode(spec.Params, &pp); err != nil {
			return nil, fmt.Errorf("policy plan: %w", err)
		}
		return NewPlan(pp.Starts)
	}
	var prm Params
	if err := decode(spec.Params, &prm); err != nil {
		return nil, fmt.Errorf("policy %q: %w", spec.Name, err)
	}
	c, err := newComposite(spec.Name, prm, env.Oracle)
	if err != nil {
		return nil, err
	}
	return c, nil
}

// decode parses params strictly (unknown fields are an error).
func decode(raw json.RawMessage, v any) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	return d.Decode(v)
}
