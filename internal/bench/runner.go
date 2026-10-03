package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

// Env is the execution environment of a benchmark command.
type Env struct {
	Root    string        // repository root
	Outputs string        // ignored output folder (traces, logs, per-run JSON)
	Workers int           // parallel runs
	Timeout time.Duration // external-policy call timeout
	Log     func(format string, a ...any)
	// JobsOverride replaces the workload's job count (quick mode); 0 = none.
	JobsOverride int
	// TuningSeeds marks the tuning seeds (replay scenarios use the tuning window for them).
	TuningSeeds map[uint64]bool
	// Skipped collects scenarios skipped because their replay data is missing.
	Skipped []string
}

func (env *Env) logf(format string, a ...any) {
	if env.Log != nil {
		env.Log(format, a...)
	}
}

// Reference holds R_s and R_w of J.
type Reference struct {
	Rs float64 `json:"R_s"`
	Rw float64 `json:"R_w"`
}

// TraceSet is one generated trace with its cluster.
type TraceSet struct {
	Jobs      []api.Job
	Cluster   *cluster.Cluster
	SLO, Fair bool
	Interval  time.Duration
	Failures  []simulator.Failure
}

// Traces generates (and saves, then replays from file) the scenario's traces.
// A trace depends only on the scenario's generator configuration and the
// seed, never on the policies.
func (env *Env) Traces(sc *Scenario, seeds []uint64) (map[uint64]*TraceSet, error) {
	if sc.Replay != nil {
		return env.replayTraces(sc, seeds)
	}
	cl, err := sc.LoadCluster(env.Root)
	if err != nil {
		return nil, err
	}
	var extra map[string]any
	if env.JobsOverride > 0 && sc.Kind == KindSim {
		extra = map[string]any{"jobs": env.JobsOverride}
	}
	g, err := sc.GenConfig(env.Root, cl, extra)
	if err != nil {
		return nil, err
	}
	slo, fair := Applies(g)
	out := map[uint64]*TraceSet{}
	empty := cl.EmptyNodes()
	for _, seed := range seeds {
		jobs, err := trace.Generate(g, seed)
		if err != nil {
			return nil, err
		}
		path := filepath.Join(env.Outputs, "traces", sc.ID, fmt.Sprintf("seed-%d.csv", seed))
		if _, err := trace.SaveWithManifest(path, jobs, g.Info(), seed); err != nil {
			return nil, err
		}
		loaded, _, err := trace.LoadVerified(path)
		if err != nil {
			return nil, err
		}
		for i := range loaded {
			j := &loaded[i]
			if _, ok := metrics.Classify(metrics.Shape{GPUs: j.GPUs, CPUs: j.CPUs, MemMB: j.MemMB(), Workers: j.Workers, Class: j.GPUClass}, empty); !ok {
				return nil, fmt.Errorf("scenario %s seed %d: generated job %s cannot fit the cluster (generated traces must contain no infeasible job)", sc.ID, seed, j.ID)
			}
		}
		ts := &TraceSet{Jobs: loaded, Cluster: cl, SLO: slo, Fair: fair, Interval: time.Duration(sc.ScheduleIntervalS * float64(time.Second))}
		if sc.Failures != nil && len(loaded) > 0 {
			// failures up to 30 days after the last submission (the run drains before)
			ts.Failures = sc.Failures.FailureSchedule(cl, seed, loaded[len(loaded)-1].Submit+30*24*time.Hour)
		}
		out[seed] = ts
	}
	return out, nil
}

// Task is one simulation run.
type Task struct {
	Scenario string
	Policy   string
	Params   json.RawMessage
	Seed     uint64
	Trace    *TraceSet
	Ref      Reference
	Weights  Weights
	Tag      string             // extra label (tuning configuration)
	Shares   map[string]float64 // quota shares of the scenario (quota metrics)
}

// RunTasks runs tasks on Workers goroutines. Results are in task order. The
// first error cancels the remaining runs and is returned.
func (env *Env) RunTasks(ctx context.Context, tasks []Task) ([]Row, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	rows := make([]Row, len(tasks))
	var (
		mu       sync.Mutex
		firstErr error
		done     int
		wg       sync.WaitGroup
	)
	next := make(chan int)
	workers := max(1, env.Workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				row, err := env.runOne(ctx, &tasks[i])
				mu.Lock()
				if err != nil && firstErr == nil {
					firstErr = err
					cancel()
				}
				rows[i] = row
				done++
				if done%200 == 0 {
					env.logf("  %d/%d runs", done, len(tasks))
				}
				mu.Unlock()
			}
		}()
	}
loop:
	for i := range tasks {
		select {
		case next <- i:
		case <-ctx.Done():
			break loop
		}
	}
	close(next)
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	if err := ctx.Err(); err != nil && !errors.Is(err, context.Canceled) {
		return nil, err
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return rows, nil
}

func sanitize(s string) string { return strings.NewReplacer(":", "_", "/", "_", "\\", "_").Replace(s) }

func (env *Env) runOne(ctx context.Context, t *Task) (row Row, err error) {
	if ctx.Err() != nil {
		return row, ctx.Err()
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("scenario %s policy %s seed %d: panic: %v\n%s", t.Scenario, t.Policy, t.Seed, r, debug.Stack())
		}
	}()
	label := fmt.Sprintf("scenario %s policy %s seed %d%s", t.Scenario, t.Policy, t.Seed, t.Tag)
	truth := make(map[string]time.Duration, len(t.Trace.Jobs))
	for _, j := range t.Trace.Jobs {
		truth[j.ID] = j.Runtime
	}
	var logFile *os.File
	penv := policy.Env{
		Oracle: func(id string) (time.Duration, bool) { r, ok := truth[id]; return r, ok },
		External: func(spec policy.Spec) (policy.Policy, error) {
			p := filepath.Join(env.Outputs, "logs", t.Scenario, sanitize(t.Policy)+t.Tag, fmt.Sprintf("seed-%d.stderr.log", t.Seed))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return nil, err
			}
			f, err := os.Create(p)
			if err != nil {
				return nil, err
			}
			logFile = f
			return extpolicy.New(strings.TrimPrefix(spec.Name, "py:"), spec.Params,
				extpolicy.Config{Dir: env.Root, Timeout: env.Timeout, Stderr: f, Seed: t.Seed}), nil
		},
	}
	p, err := policy.New(policy.Spec{Name: t.Policy, Params: t.Params}, penv)
	if err != nil {
		return row, fmt.Errorf("%s: %w", label, err)
	}
	defer func() {
		if c, ok := p.(policy.Closer); ok {
			if cerr := c.Close(); cerr != nil && err == nil {
				err = fmt.Errorf("%s: %w", label, cerr)
			}
		}
		if logFile != nil {
			logFile.Close()
		}
	}()
	// The external client is closed on every path above; a cancelled context
	// (interrupt, another run's error) stops the run between invocations.
	rt, err := simulator.Run(simulator.Config{Cluster: t.Trace.Cluster, Jobs: t.Trace.Jobs, Policy: &ctxPolicy{ctx, p},
		ScheduleInterval: t.Trace.Interval, Wall: time.Now, Failures: t.Trace.Failures})
	if err != nil {
		return row, fmt.Errorf("%s: %w", label, err)
	}
	if err := metrics.Verify(rt, t.Trace.Cluster.EmptyNodes()); err != nil {
		return row, fmt.Errorf("%s: invariant violated: %w", label, err)
	}
	if err := metrics.CheckIdentities(rt); err != nil {
		return row, fmt.Errorf("%s: accounting identity violated: %w", label, err)
	}
	m := metrics.Compute(rt)
	row = Row{Scenario: t.Scenario, Policy: t.Policy + t.Tag, Seed: t.Seed, Fields: m.Fields, Wall: metrics.Wall(rt).Fields}
	row.Extra = Objective(m, t.Ref, t.Weights, t.Trace.SLO, t.Trace.Fair)
	if len(t.Shares) > 0 {
		row.Fields = append(row.Fields, metrics.QuotaFields(rt, t.Shares)...)
	}
	if rt.Solver != nil {
		row.Extra = append(row.Extra, metrics.Field{Name: "solver_capped", Value: b2f(rt.Solver.Capped > 0)})
	}
	if err := env.writeRunJSON(t, &row); err != nil {
		return row, err
	}
	return row, nil
}

// ctxPolicy stops a run when the context is cancelled.
type ctxPolicy struct {
	ctx context.Context
	p   policy.Policy
}

func (c *ctxPolicy) Name() string { return c.p.Name() }
func (c *ctxPolicy) Schedule(v *api.View) (api.Decision, error) {
	if err := c.ctx.Err(); err != nil {
		return api.Decision{}, err
	}
	return c.p.Schedule(v)
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// Objective computes J and its terms (docs/contracts.md §5).
func Objective(m *metrics.Run, ref Reference, w Weights, slo, fair bool) []metrics.Field {
	t1 := w.Alpha * m.Get("bsld_p95") / ref.Rs
	t2 := w.Beta * m.Get("wait_p95") / ref.Rw
	t3 := w.Gamma * (1 - m.Get("goodput_ratio"))
	t4, t5 := 0.0, 0.0
	if v := m.Get("slo_violation_rate"); slo && !math.IsNaN(v) {
		t4 = w.Delta * v
	}
	if j := m.Get("jain_bsld"); fair && !math.IsNaN(j) {
		t5 = w.Epsilon * (1 - j)
	}
	return []metrics.Field{{Name: "J", Value: t1 + t2 + t3 + t4 + t5}, {Name: "J_slowdown", Value: t1}, {Name: "J_wait", Value: t2},
		{Name: "J_waste", Value: t3}, {Name: "J_slo", Value: t4}, {Name: "J_fairness", Value: t5}}
}

func (env *Env) writeRunJSON(t *Task, r *Row) error {
	p := filepath.Join(env.Outputs, "runs", t.Scenario, sanitize(t.Policy)+t.Tag, fmt.Sprintf("seed-%d.json", t.Seed))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	obj := map[string]any{"scenario": r.Scenario, "policy": r.Policy, "seed": r.Seed}
	for _, fs := range [][]metrics.Field{r.Extra, r.Fields, r.Wall} {
		for _, f := range fs {
			if math.IsNaN(f.Value) || math.IsInf(f.Value, 0) {
				obj[f.Name] = nil
			} else {
				obj[f.Name] = f.Value
			}
		}
	}
	b, err := json.MarshalIndent(obj, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, b, 0o644)
}
