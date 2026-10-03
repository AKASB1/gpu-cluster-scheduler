// Package simulator is the discrete-event simulator of a GPU cluster
// (docs/simulator.md). One run is single-threaded and deterministic: the same
// cluster, trace, policy configuration, and seed give the same RunTrace.
package simulator

import (
	"cmp"
	"container/heap"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/preemption"
)

// Config is the input of one run.
type Config struct {
	Cluster *cluster.Cluster
	Jobs    []api.Job // trace order (sorted by submit time)
	Policy  policy.Policy
	// ScheduleInterval adds periodic policy invocations (0 = off).
	ScheduleInterval time.Duration
	// Wall measures decision times (wall_ columns); nil = not measured.
	Wall func() time.Time
	// Log records the assignment log.
	Log bool
	// MaxInvocations aborts a run whose policy keeps asking for wake-ups
	// without progress (0 = 1000 * (jobs + 1)).
	MaxInvocations int
	// Failures is an explicit node-failure schedule (Tier 2; none by default):
	// at At the node goes down for Repair; the jobs running on it are killed
	// and requeued under the checkpoint rule (whether preemptible or not).
	Failures []Failure
	// OnSample, when set, receives every sample (the state after each
	// scheduling instant): the replay exporter uses it to pace and publish.
	OnSample func(metrics.Sample)
}

// Failure is one node failure.
type Failure struct {
	Node   string
	At     time.Duration
	Repair time.Duration
}

// DeadlockError: nothing runs, jobs are pending, no submission is left, and
// the policy starts nothing when called again. A policy bug, not a result.
type DeadlockError struct {
	Policy  string
	At      time.Duration
	Pending int
}

func (e *DeadlockError) Error() string {
	return fmt.Sprintf("deadlock: policy %q starts nothing at t=%ss with %d pending jobs, nothing running, and no future submission",
		e.Policy, clock.FormatSeconds(e.At), e.Pending)
}

// ActionError is an invalid action; the run is aborted.
type ActionError struct {
	Policy string
	At     time.Duration
	Index  int
	Action api.Action
	Reason string
}

func (e *ActionError) Error() string {
	return fmt.Sprintf("invalid action #%d by policy %q at t=%ss: %s %s: %s", e.Index, e.Policy,
		clock.FormatSeconds(e.At), e.Action.Op, e.Action.JobID, e.Reason)
}

type status int

const (
	stFuture status = iota
	stPending
	stRunning
	stStopping // preempted, holding its GPUs during the grace period
	stDone
	stInfeasible
)

type jobState struct {
	job          *api.Job
	rec          *metrics.JobRecord
	st           status
	memMB        int64
	retained     float64
	started      bool
	preemptions  int
	pendingSince time.Duration
	// current run
	gen         int // run generation, for lazy deletion of completion events
	runStart    time.Duration
	overhead    time.Duration
	placement   api.Placement
	rate        float64
	sMin, f     float64
	end         time.Duration
	retainedRun float64 // retained when this run started
}

type event struct {
	t   time.Duration
	id  string
	j   *jobState
	gen int
	// release events (grace) carry the resources to free
	release api.Placement
}

type eventHeap []event

func (h eventHeap) Len() int { return len(h) }
func (h eventHeap) Less(a, b int) bool {
	if h[a].t != h[b].t {
		return h[a].t < h[b].t
	}
	return h[a].id < h[b].id
}
func (h eventHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h *eventHeap) Push(x any)   { *h = append(*h, x.(event)) }
func (h *eventHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

type failureEv struct {
	node   int
	at     time.Duration
	repair time.Duration
}

type tenantAcc struct {
	gpuMs   int64 // finished allocation, GPU-ms
	running int
	cpus    int
	memMB   int64
}

// sim is the state of one run.
type sim struct {
	cfg      Config
	info     api.ClusterInfo
	nodes    []api.NodeState // current free resources; Running filled per view
	shares   []map[string]int
	jobs     []jobState
	byID     map[string]*jobState
	pending  []*jobState
	running  map[string]*jobState
	finishes eventHeap
	releases eventHeap
	held     int // GPUs held by grace periods
	tenants  map[string]*tenantAcc
	history  []api.CompletedJob
	now      time.Duration
	clk      clock.Virtual
	out      *metrics.RunTrace
	inSystem int
	alloc    int
	wake     time.Duration
	fails    []failureEv
	nextFail int
	downTill []time.Duration // per node: end of the current repair (0 = up)
	nodeIdx  map[string]int
	logOn    bool
}

// Run simulates the trace under the policy until every feasible job has
// completed.
func Run(cfg Config) (*metrics.RunTrace, error) {
	s, err := newSim(cfg)
	if err != nil {
		return nil, err
	}
	if err := s.loop(); err != nil {
		return nil, err
	}
	return s.out, nil
}

func newSim(cfg Config) (*sim, error) {
	if cfg.Cluster == nil || cfg.Policy == nil {
		return nil, errors.New("simulator: cluster and policy are required")
	}
	c := cfg.Cluster
	s := &sim{cfg: cfg, info: c.Info, nodes: c.EmptyNodes(), byID: map[string]*jobState{},
		running: map[string]*jobState{}, tenants: map[string]*tenantAcc{}, nodeIdx: map[string]int{}, logOn: cfg.Log}
	s.shares = make([]map[string]int, len(s.nodes))
	for i := range s.nodes {
		s.shares[i] = map[string]int{}
		s.nodeIdx[s.nodes[i].Name] = i
	}
	s.downTill = make([]time.Duration, len(s.nodes))
	if len(cfg.Failures) > 0 && c.Info.PreemptGrace > 0 {
		return nil, fmt.Errorf("simulator: node failures with preempt_grace_s > 0 are not supported")
	}
	for _, f := range cfg.Failures {
		n, ok := s.nodeIdx[f.Node]
		if !ok || f.At < 0 || f.Repair <= 0 {
			return nil, fmt.Errorf("simulator: invalid failure %+v", f)
		}
		s.fails = append(s.fails, failureEv{n, f.At, f.Repair})
	}
	slices.SortStableFunc(s.fails, func(a, b failureEv) int {
		if c := cmp.Compare(a.at, b.at); c != 0 {
			return c
		}
		return cmp.Compare(a.node, b.node)
	})
	classSpeed := map[string]float64{}
	for _, gc := range c.Info.Classes {
		classSpeed[gc.Name] = gc.Speed
	}
	s.out = &metrics.RunTrace{PolicyName: cfg.Policy.Name(), Info: c.Info, FastestAny: c.FastestSpeed(""),
		ClassSpeed: classSpeed, Jobs: make([]metrics.JobRecord, len(cfg.Jobs))}
	s.jobs = make([]jobState, len(cfg.Jobs))
	var feasible []api.Job
	empty := c.EmptyNodes()
	for i := range cfg.Jobs {
		j := &cfg.Jobs[i]
		if i > 0 && j.Submit < cfg.Jobs[i-1].Submit {
			return nil, fmt.Errorf("simulator: trace not sorted by submit time at job %s", j.ID)
		}
		if s.byID[j.ID] != nil {
			return nil, fmt.Errorf("simulator: duplicate job_id %s", j.ID)
		}
		if j.GPUClass != "" && !c.HasClass(j.GPUClass) {
			return nil, fmt.Errorf("simulator: job %s requests unknown class %q", j.ID, j.GPUClass)
		}
		js := &s.jobs[i]
		js.job, js.rec, js.memMB = j, &s.out.Jobs[i], j.MemMB()
		js.rec.Job = *j
		s.byID[j.ID] = js
		if _, ok := metrics.Classify(shapeOf(j, js.memMB), empty); !ok {
			js.st, js.rec.Infeasible = stInfeasible, true
			continue
		}
		feasible = append(feasible, *j)
		if s.tenants[j.Tenant] == nil {
			s.tenants[j.Tenant] = &tenantAcc{}
		}
	}
	s.out.SizeDist = metrics.SizeDistOf(feasible)
	return s, nil
}

func shapeOf(j *api.Job, memMB int64) metrics.Shape {
	return metrics.Shape{GPUs: j.GPUs, CPUs: j.CPUs, MemMB: memMB, Workers: j.Workers, Class: j.GPUClass}
}

func (s *sim) loop() error {
	next := 0 // next trace index to submit
	skip := func() {
		for next < len(s.jobs) && s.jobs[next].st == stInfeasible {
			next++
		}
	}
	skip()
	var tick time.Duration
	if s.cfg.ScheduleInterval > 0 {
		tick = s.cfg.ScheduleInterval
	}
	maxInv := s.cfg.MaxInvocations
	if maxInv == 0 {
		maxInv = 1000 * (len(s.jobs) + 1)
	}
	for {
		t, ok := time.Duration(math.MaxInt64), false
		consider := func(c time.Duration) {
			if c < t {
				t, ok = c, true
			}
		}
		if next < len(s.jobs) {
			consider(s.jobs[next].job.Submit)
		}
		for s.finishes.Len() > 0 && s.stale(s.finishes[0]) {
			heap.Pop(&s.finishes)
		}
		if s.finishes.Len() > 0 {
			consider(s.finishes[0].t)
		}
		if s.releases.Len() > 0 {
			consider(s.releases[0].t)
		}
		busy := next < len(s.jobs) || len(s.pending) > 0 || len(s.running) > 0 || s.releases.Len() > 0
		if tick > 0 && busy {
			consider(tick)
		}
		if s.wake > 0 {
			consider(s.wake)
		}
		if busy {
			if s.nextFail < len(s.fails) {
				consider(s.fails[s.nextFail].at)
			}
			for _, d := range s.downTill {
				if d > 0 {
					consider(d)
				}
			}
		}
		if !ok {
			break
		}
		s.advance(t)
		// 1. finishes (grace releases first, then completions, by job ID)
		for s.releases.Len() > 0 && s.releases[0].t == t {
			ev := heap.Pop(&s.releases).(event)
			s.free(ev.j.job, ev.j.memMB, ev.release)
			s.held -= ev.j.job.TotalGPUs()
			s.alloc -= ev.j.job.TotalGPUs()
			// the job re-enters the queue when its GPUs are released
			ev.j.st = stPending
			s.pending = append(s.pending, ev.j)
		}
		for s.finishes.Len() > 0 && s.finishes[0].t == t {
			ev := heap.Pop(&s.finishes).(event)
			if !s.stale(ev) {
				s.complete(ev.j)
			}
		}
		// node repairs, then failures (jobs completing at this instant completed)
		s.repairsAndFailures(t)
		// 2. submissions in trace order
		for next < len(s.jobs) && s.jobs[next].job.Submit == t {
			js := &s.jobs[next]
			js.st, js.pendingSince = stPending, t
			s.pending = append(s.pending, js)
			s.inSystem++
			next++
			skip()
		}
		if tick > 0 && tick == t {
			tick += s.cfg.ScheduleInterval
		}
		if s.wake == t {
			s.wake = 0
		}
		// 3. the policy, once for this instant
		if len(s.pending) > 0 {
			started, err := s.invoke()
			if err != nil {
				return err
			}
			if !started && len(s.running) == 0 && next >= len(s.jobs) && s.releases.Len() == 0 && s.wake == 0 && !s.anyDown() {
				if started, err = s.invoke(); err != nil {
					return err
				}
				if !started {
					return &DeadlockError{Policy: s.cfg.Policy.Name(), At: t, Pending: len(s.pending)}
				}
			}
			if s.out.Invocations > maxInv {
				return fmt.Errorf("simulator: policy %q exceeded %d invocations without draining", s.cfg.Policy.Name(), maxInv)
			}
		}
		s.sample()
	}
	if len(s.pending) > 0 || len(s.running) > 0 {
		return fmt.Errorf("simulator: internal error: run ended with %d pending and %d running", len(s.pending), len(s.running))
	}
	return nil
}

func (s *sim) stale(ev event) bool { return ev.j.st != stRunning || ev.gen != ev.j.gen }

// advance integrates the number of jobs in the system and the allocated GPUs
// up to t.
func (s *sim) advance(t time.Duration) {
	dt := (t - s.now).Milliseconds()
	s.out.NIntegral += int64(s.inSystem) * dt
	s.out.AllocIntMs += int64(s.alloc) * dt
	s.now = t
	s.clk.Set(t)
}

func (s *sim) sample() {
	free := 0
	stranded := 0.0
	for i := range s.nodes {
		free += s.nodes[i].FreeGPUs
		stranded += s.out.SizeDist.Stranded(s.nodes[i].FreeGPUs)
	}
	shapes := make([]metrics.Shape, 0, len(s.pending))
	for _, js := range s.pending {
		shapes = append(shapes, shapeOf(js.job, js.memMB))
	}
	s.out.Samples = append(s.out.Samples, metrics.Sample{T: s.now, Alloc: s.alloc, Free: free, Stranded: stranded,
		Blocked: metrics.Blocked(metrics.UniqueShapes(shapes), s.nodes), Pending: len(s.pending), Running: len(s.running)})
	if s.cfg.OnSample != nil {
		s.cfg.OnSample(s.out.Samples[len(s.out.Samples)-1])
	}
}

// invoke calls the policy once and applies its decision. It reports whether
// a job was started.
func (s *sim) invoke() (bool, error) {
	v := s.view()
	var t0 time.Time
	if s.cfg.Wall != nil {
		t0 = s.cfg.Wall()
	}
	dec, err := s.cfg.Policy.Schedule(v)
	if s.cfg.Wall != nil {
		s.out.WallDecide = append(s.out.WallDecide, s.cfg.Wall().Sub(t0).Seconds())
	}
	s.out.Invocations++
	if err != nil {
		return false, fmt.Errorf("policy %q failed at t=%ss: %w", s.cfg.Policy.Name(), clock.FormatSeconds(s.now), err)
	}
	if dec.Solver != nil {
		st := s.out.Solver
		if st == nil {
			st = &metrics.SolverStats{}
			s.out.Solver = st
		}
		st.Calls++
		if dec.Solver.Capped {
			st.Capped++
		}
		if dec.Solver.Status != "optimal" {
			st.NotOptimal++
		}
		st.WallSolve = append(st.WallSolve, dec.Solver.WallSolve)
		st.CPUSolve = append(st.CPUSolve, dec.Solver.CPUSolve)
		st.MaxGap = max(st.MaxGap, dec.Solver.Gap)
	}
	started := false
	for i, a := range dec.Actions {
		var reason string
		switch a.Op {
		case api.OpStart:
			reason = s.start(a)
			started = started || reason == ""
		case api.OpPreempt:
			reason = s.preempt(a)
		default:
			reason = "unknown op"
		}
		if reason != "" {
			return false, &ActionError{Policy: s.cfg.Policy.Name(), At: s.now, Index: i, Action: a, Reason: reason}
		}
	}
	for _, r := range dec.Reservations {
		if js := s.byID[r.JobID]; js != nil && !js.rec.HasShadow && js.st == stPending {
			js.rec.HasShadow, js.rec.Shadow, js.rec.ShadowAt = true, r.Shadow, s.now
		}
	}
	if dec.WakeAt > s.now && (s.wake == 0 || dec.WakeAt < s.wake) {
		s.wake = dec.WakeAt
	}
	return started, nil
}

func (s *sim) start(a api.Action) string {
	js := s.byID[a.JobID]
	if js == nil {
		return "unknown job"
	}
	if js.st != stPending {
		return "job is not pending"
	}
	j := js.job
	p := slices.Clone(a.Placement)
	slices.SortFunc(p, func(x, y api.NodeWorkers) int { return cmp.Compare(x.Node, y.Node) })
	total := 0
	for i, nw := range p {
		if nw.Node < 0 || nw.Node >= len(s.nodes) {
			return fmt.Sprintf("unknown node index %d", nw.Node)
		}
		if i > 0 && p[i-1].Node == nw.Node {
			return "node listed twice: " + s.nodes[nw.Node].Name
		}
		if nw.Workers < 1 {
			return "worker count < 1 on " + s.nodes[nw.Node].Name
		}
		n := &s.nodes[nw.Node]
		if j.GPUClass != "" && n.Class != j.GPUClass {
			return fmt.Sprintf("node %s has class %s, job needs %s", n.Name, n.Class, j.GPUClass)
		}
		if j.GPUs*nw.Workers > n.FreeGPUs || j.CPUs*nw.Workers > n.FreeCPUs || js.memMB*int64(nw.Workers) > n.FreeMemMB {
			return fmt.Sprintf("capacity exceeded on node %s (%d workers)", n.Name, nw.Workers)
		}
		total += nw.Workers
	}
	if total != j.Workers {
		return fmt.Sprintf("placement has %d workers, job needs %d (gang)", total, j.Workers)
	}
	// apply
	for _, nw := range p {
		n := &s.nodes[nw.Node]
		n.FreeGPUs -= j.GPUs * nw.Workers
		n.FreeCPUs -= j.CPUs * nw.Workers
		n.FreeMemMB -= js.memMB * int64(nw.Workers)
		s.shares[nw.Node][j.ID] = nw.Workers
	}
	s.removePending(js)
	js.st = stRunning
	js.gen++
	js.runStart, js.placement, js.retainedRun = s.now, p, js.retained
	js.overhead = 0
	if js.started {
		js.overhead = s.info.RestartOverhead
	}
	if !js.started {
		js.rec.FirstStart = s.now
		js.started = true
	}
	js.rec.QueueTotal += s.now - js.pendingSince
	js.rate, js.sMin, js.f = api.RateOf(j.Topology, p, s.nodes, &s.info)
	js.end = s.now + api.EstimatedDuration(clock.Sec(j.Runtime)-js.retained, js.overhead, js.rate)
	heap.Push(&s.finishes, event{t: js.end, id: j.ID, j: js, gen: js.gen})
	s.running[j.ID] = js
	s.alloc += j.TotalGPUs()
	ta := s.tenants[j.Tenant]
	ta.running += j.TotalGPUs()
	ta.cpus += j.CPUs * j.Workers
	ta.memMB += js.memMB * int64(j.Workers)
	s.logEntry(api.OpStart, j.ID, p)
	return ""
}

func (s *sim) preempt(a api.Action) string {
	js := s.byID[a.JobID]
	if js == nil {
		return "unknown job"
	}
	if js.st != stRunning {
		return "job is not running"
	}
	if !js.job.Preemptible {
		return "job is not preemptible"
	}
	if js.runStart == s.now {
		// a start and a preemption in the same decision would mark the job as
		// started (wait, SLO) without giving it any GPU time
		return "job was started at this instant (a run must last at least 1 ms)"
	}
	j := js.job
	work := js.retainedRun + s.progress(js, s.now)
	retained := preemption.Retained(work, clock.Sec(j.CheckpointInterval))
	seg := s.closeSegment(js, s.now)
	seg.Preempted, seg.Retained = true, retained
	G := float64(j.TotalGPUs())
	js.rec.LostW += G * (work - retained)
	js.retained = retained
	js.preemptions++
	js.rec.Preemptions++
	if g := s.info.PreemptGrace; g > 0 {
		seg.Grace = g
		w := 0.0
		for _, nw := range js.placement {
			w += float64(j.GPUs*nw.Workers) * s.nodes[nw.Node].Speed * clock.Sec(g)
		}
		seg.GraceW = w
		seg.ConsumedW += w
		js.rec.GraceW += w
		js.rec.ConsumedW += w
		js.rec.AllocGPUms += int64(j.TotalGPUs()) * g.Milliseconds()
		s.tenants[j.Tenant].gpuMs += int64(j.TotalGPUs()) * g.Milliseconds()
		s.held += j.TotalGPUs()
		heap.Push(&s.releases, event{t: s.now + g, id: j.ID, j: js, release: js.placement})
		for _, nw := range js.placement {
			delete(s.shares[nw.Node], j.ID)
		}
	} else {
		s.free(j, js.memMB, js.placement)
		s.alloc -= j.TotalGPUs()
	}
	js.rec.Segments = append(js.rec.Segments, seg)
	delete(s.running, j.ID)
	s.tenantStop(js)
	// Pending time (queue_total) counts from the preemption; with a grace
	// period the job re-enters the queue only when its GPUs are released.
	js.pendingSince = s.now
	if s.info.PreemptGrace > 0 {
		js.st = stStopping
	} else {
		js.st = stPending
		s.pending = append(s.pending, js)
	}
	s.logEntry(api.OpPreempt, j.ID, nil)
	return ""
}

func (s *sim) complete(js *jobState) {
	j := js.job
	seg := s.closeSegment(js, s.now)
	G := float64(j.TotalGPUs())
	js.rec.UsefulW = G * clock.Sec(j.Runtime)
	js.rec.RoundingW = G * (js.retainedRun + seg.Work - clock.Sec(j.Runtime))
	js.rec.Segments = append(js.rec.Segments, seg)
	js.rec.Completion = s.now
	s.free(j, js.memMB, js.placement)
	s.alloc -= j.TotalGPUs()
	delete(s.running, j.ID)
	s.tenantStop(js)
	js.st = stDone
	s.inSystem--
	s.out.End = s.now
	s.history = append(s.history, api.CompletedJob{JobID: j.ID, Tenant: j.Tenant, User: j.User, Submit: j.Submit,
		Runtime: j.Runtime, Estimate: j.Estimate, Finish: s.now})
}

// progress is the work done in the current run up to t.
func (s *sim) progress(js *jobState, t time.Duration) float64 {
	d := t - js.runStart - js.overhead
	if d <= 0 {
		return 0
	}
	return clock.Sec(d) * js.rate
}

// closeSegment ends the current run at t and books its accounting terms.
func (s *sim) closeSegment(js *jobState, t time.Duration) metrics.Segment {
	j := js.job
	D := t - js.runStart
	O := min(D, js.overhead)
	Dp := clock.Sec(D - O)
	seg := metrics.Segment{Start: js.runStart, End: t, Overhead: js.overhead, Placement: js.placement,
		SMin: js.sMin, F: js.f, Rate: js.rate, Work: s.progress(js, t)}
	for _, nw := range js.placement {
		g := float64(j.GPUs * nw.Workers)
		sp := s.nodes[nw.Node].Speed
		seg.ConsumedW += g * sp * clock.Sec(D)
		seg.OverheadW += g * sp * clock.Sec(O)
		seg.SlackW += g * (sp - js.sMin) * Dp
	}
	G := float64(j.TotalGPUs())
	seg.TopoW = G * js.sMin * Dp * (1 - 1/js.f)
	r := js.rec
	r.ConsumedW += seg.ConsumedW
	r.OverheadW += seg.OverheadW
	r.SlackW += seg.SlackW
	r.TopoW += seg.TopoW
	r.AllocGPUms += int64(j.TotalGPUs()) * D.Milliseconds()
	s.tenants[j.Tenant].gpuMs += int64(j.TotalGPUs()) * D.Milliseconds()
	return seg
}

func (s *sim) free(j *api.Job, memMB int64, p api.Placement) {
	for _, nw := range p {
		n := &s.nodes[nw.Node]
		n.FreeGPUs += j.GPUs * nw.Workers
		n.FreeCPUs += j.CPUs * nw.Workers
		n.FreeMemMB += memMB * int64(nw.Workers)
		delete(s.shares[nw.Node], j.ID)
	}
}

func (s *sim) tenantStop(js *jobState) {
	ta := s.tenants[js.job.Tenant]
	ta.running -= js.job.TotalGPUs()
	ta.cpus -= js.job.CPUs * js.job.Workers
	ta.memMB -= js.memMB * int64(js.job.Workers)
}

func (s *sim) removePending(js *jobState) {
	for i, p := range s.pending {
		if p == js {
			s.pending = slices.Delete(s.pending, i, i+1)
			return
		}
	}
}

func (s *sim) logEntry(op api.Op, id string, p api.Placement) {
	if !s.logOn {
		return
	}
	e := metrics.LogEntry{T: s.now, Op: op, JobID: id}
	for _, nw := range p {
		e.Placement = append(e.Placement, metrics.NamedWorkers{Node: s.nodes[nw.Node].Name, Workers: nw.Workers})
	}
	s.out.Log = append(s.out.Log, e)
}

// view builds the read-only input of one invocation.
func (s *sim) view() *api.View {
	v := &api.View{Now: s.now, Cluster: &s.info, History: s.history[:len(s.history):len(s.history)]}
	v.Nodes = make([]api.NodeState, len(s.nodes))
	for i := range s.nodes {
		v.Nodes[i] = s.nodes[i]
		v.Nodes[i].Down = s.downTill[i] > 0
		if len(s.shares[i]) > 0 {
			sh := make([]api.NodeShare, 0, len(s.shares[i]))
			for id, w := range s.shares[i] {
				sh = append(sh, api.NodeShare{JobID: id, Workers: w})
			}
			slices.SortFunc(sh, func(a, b api.NodeShare) int { return cmp.Compare(a.JobID, b.JobID) })
			v.Nodes[i].Running = sh
		}
	}
	v.Running = make([]api.RunningJob, 0, len(s.running))
	for _, js := range s.running {
		j := js.job
		done := js.retainedRun + s.progress(js, s.now)
		estEnd := js.runStart + api.EstimatedDuration(clock.Sec(j.Estimate)-js.retainedRun, js.overhead, js.rate)
		v.Running = append(v.Running, api.RunningJob{JobID: j.ID, Tenant: j.Tenant, User: j.User, Priority: j.Priority,
			GPUs: j.GPUs, Workers: j.Workers, CPUs: j.CPUs, MemMB: js.memMB, GPUClass: j.GPUClass, Topology: j.Topology,
			Submit: j.Submit, FirstStart: js.rec.FirstStart, RunStart: js.runStart, Overhead: js.overhead,
			Placement: js.placement, Rate: js.rate, Estimate: j.Estimate, RetainedAtStart: js.retainedRun,
			WorkDone: done, EstRemainingWork: math.Max(0, clock.Sec(j.Estimate)-done), EstEnd: max(estEnd, s.now),
			Preemptible: j.Preemptible, CheckpointInterval: j.CheckpointInterval, Preemptions: js.preemptions})
	}
	slices.SortFunc(v.Running, func(a, b api.RunningJob) int { return cmp.Compare(a.JobID, b.JobID) })
	v.Pending = make([]api.PendingJob, len(s.pending))
	for i, js := range s.pending {
		j := js.job
		v.Pending[i] = api.PendingJob{JobID: j.ID, Tenant: j.Tenant, User: j.User, Submit: j.Submit, Priority: j.Priority,
			GPUs: j.GPUs, Workers: j.Workers, CPUs: j.CPUs, MemMB: js.memMB, GPUClass: j.GPUClass, Topology: j.Topology,
			Estimate: j.Estimate, Wait: js.rec.QueueTotal + s.now - js.pendingSince, Retained: js.retained,
			MaxWait: j.MaxWait, HasMaxWait: j.HasMaxWait, Preemptible: j.Preemptible,
			CheckpointInterval: j.CheckpointInterval, Preemptions: js.preemptions, Started: js.started}
	}
	slices.SortFunc(v.Pending, func(a, b api.PendingJob) int { return api.PendingLess(&a, &b) })
	names := make([]string, 0, len(s.tenants))
	for name := range s.tenants {
		names = append(names, name)
	}
	slices.Sort(names)
	v.Tenants = make([]api.TenantUsage, len(names))
	for i, name := range names {
		ta := s.tenants[name]
		ms := ta.gpuMs
		for _, js := range s.running {
			if js.job.Tenant == name {
				ms += int64(js.job.TotalGPUs()) * (s.now - js.runStart).Milliseconds()
			}
		}
		v.Tenants[i] = api.TenantUsage{Tenant: name, GPUSeconds: float64(ms) / 1000, RunningGPUs: ta.running,
			RunningCPUs: ta.cpus, RunningMem: ta.memMB}
	}
	return v
}
