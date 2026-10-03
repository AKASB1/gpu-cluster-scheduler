package extpolicy

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// ProtocolVersion is the external-policy protocol version.
const ProtocolVersion = 1

// Secs is a whole-millisecond duration written as decimal seconds with at
// most three decimals (an exact JSON number).
type Secs time.Duration

// MarshalJSON implements json.Marshaler.
func (s Secs) MarshalJSON() ([]byte, error) {
	return []byte(clock.FormatSeconds(time.Duration(s))), nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *Secs) UnmarshalJSON(b []byte) error {
	d, err := clock.ParseSeconds(string(b))
	if err != nil {
		return fmt.Errorf("time value %s: %w", b, err)
	}
	*s = Secs(d)
	return nil
}

type wireClass struct {
	Name  string  `json:"name"`
	Speed float64 `json:"speed"`
}

type wireNodeStatic struct {
	Name  string  `json:"name"`
	Rack  string  `json:"rack"`
	Class string  `json:"class"`
	Speed float64 `json:"speed"`
	GPUs  int     `json:"gpus"`
	CPUs  int     `json:"cpus"`
	MemMB int64   `json:"mem_mb"`
}

type wireCluster struct {
	Classes          []wireClass      `json:"classes"`
	Racks            []string         `json:"racks"`
	CrossNodeFactor  float64          `json:"cross_node_factor"`
	CrossRackFactor  float64          `json:"cross_rack_factor"`
	RestartOverheadS Secs             `json:"restart_overhead_s"`
	PreemptGraceS    Secs             `json:"preempt_grace_s"`
	Nodes            []wireNodeStatic `json:"nodes"`
}

type wirePolicy struct {
	Name   string          `json:"name"`
	Params json.RawMessage `json:"params,omitempty"`
}

type helloMsg struct {
	Type            string      `json:"type"`
	ProtocolVersion int         `json:"protocol_version"`
	Policy          wirePolicy  `json:"policy"`
	Seed            uint64      `json:"seed"`
	Cluster         wireCluster `json:"cluster"`
}

type helloReply struct {
	Type    string `json:"type"`
	Name    string `json:"name"`
	Version string `json:"version"`
	Message string `json:"message"`
}

type wireShare struct {
	JobID   string `json:"job_id"`
	Workers int    `json:"workers"`
}

type wireNode struct {
	Name      string      `json:"name"`
	FreeGPUs  int         `json:"free_gpus"`
	FreeCPUs  int         `json:"free_cpus"`
	FreeMemMB int64       `json:"free_mem_mb"`
	Running   []wireShare `json:"running"`
}

type wireNW struct {
	Node    string `json:"node"`
	Workers int    `json:"workers"`
}

type wireRunning struct {
	JobID              string   `json:"job_id"`
	Tenant             string   `json:"tenant"`
	User               string   `json:"user"`
	Priority           int      `json:"priority"`
	GPUs               int      `json:"gpus"`
	Workers            int      `json:"workers"`
	CPUs               int      `json:"cpus"`
	MemMB              int64    `json:"mem_mb"`
	GPUClass           string   `json:"gpu_class"`
	Topology           string   `json:"topology"`
	SubmitS            Secs     `json:"submit_s"`
	FirstStartS        Secs     `json:"first_start_s"`
	RunStartS          Secs     `json:"run_start_s"`
	OverheadS          Secs     `json:"overhead_s"`
	Placement          []wireNW `json:"placement"`
	Rate               float64  `json:"rate"`
	EstimateS          Secs     `json:"estimate_s"`
	RetainedAtStartS   float64  `json:"retained_at_start_s"`
	WorkDoneS          float64  `json:"work_done_s"`
	EstRemainingWorkS  float64  `json:"est_remaining_work_s"`
	EstEndS            Secs     `json:"est_end_s"`
	Preemptible        bool     `json:"preemptible"`
	CheckpointInterval Secs     `json:"checkpoint_interval_s"`
	Preemptions        int      `json:"preemptions"`
}

type wirePending struct {
	JobID              string  `json:"job_id"`
	Tenant             string  `json:"tenant"`
	User               string  `json:"user"`
	SubmitS            Secs    `json:"submit_s"`
	Priority           int     `json:"priority"`
	GPUs               int     `json:"gpus"`
	Workers            int     `json:"workers"`
	CPUs               int     `json:"cpus"`
	MemMB              int64   `json:"mem_mb"`
	GPUClass           string  `json:"gpu_class"`
	Topology           string  `json:"topology"`
	EstimateS          Secs    `json:"estimate_s"`
	WaitS              Secs    `json:"wait_s"`
	RetainedS          float64 `json:"retained_s"`
	MaxWaitS           *Secs   `json:"max_wait_s"`
	Preemptible        bool    `json:"preemptible"`
	CheckpointInterval Secs    `json:"checkpoint_interval_s"`
	Preemptions        int     `json:"preemptions"`
	Started            bool    `json:"started"`
}

type wireTenant struct {
	Tenant       string  `json:"tenant"`
	GPUSeconds   float64 `json:"gpu_seconds"`
	RunningGPUs  int     `json:"running_gpus"`
	RunningCPUs  int     `json:"running_cpus"`
	RunningMemMB int64   `json:"running_mem_mb"`
}

type wireCompleted struct {
	JobID     string `json:"job_id"`
	Tenant    string `json:"tenant"`
	User      string `json:"user"`
	SubmitS   Secs   `json:"submit_s"`
	RuntimeS  Secs   `json:"runtime_s"`
	EstimateS Secs   `json:"estimate_s"`
	FinishS   Secs   `json:"finish_s"`
}

type wireView struct {
	NowS       Secs            `json:"now_s"`
	Nodes      []wireNode      `json:"nodes"`
	Running    []wireRunning   `json:"running"`
	Pending    []wirePending   `json:"pending"`
	Tenants    []wireTenant    `json:"tenants"`
	HistoryNew []wireCompleted `json:"history_new"`
}

type scheduleMsg struct {
	Type string   `json:"type"`
	Seq  int      `json:"seq"`
	View wireView `json:"view"`
}

type wireAction struct {
	Op        string   `json:"op"`
	JobID     string   `json:"job_id"`
	Placement []wireNW `json:"placement"`
}

type wireReservation struct {
	JobID   string `json:"job_id"`
	ShadowS Secs   `json:"shadow_s"`
}

type wireSolver struct {
	Status string  `json:"status"`
	SolveS float64 `json:"solve_s"`
	CPUS   float64 `json:"cpu_s"`
	Gap    float64 `json:"gap"`
	Capped bool    `json:"capped"`
}

type decisionReply struct {
	Type         string            `json:"type"`
	Seq          int               `json:"seq"`
	Actions      []wireAction      `json:"actions"`
	WakeAtS      *Secs             `json:"wake_at_s"`
	Reservations []wireReservation `json:"reservations"`
	Solver       *wireSolver       `json:"solver"`
	Message      string            `json:"message"`
}

func names(p api.Placement, nodes []api.NodeState) []wireNW {
	out := make([]wireNW, len(p))
	for i, nw := range p {
		out[i] = wireNW{Node: nodes[nw.Node].Name, Workers: nw.Workers}
	}
	return out
}

func clusterOf(v *api.View) wireCluster {
	c := v.Cluster
	w := wireCluster{Racks: c.Racks, CrossNodeFactor: c.CrossNode, CrossRackFactor: c.CrossRack,
		RestartOverheadS: Secs(c.RestartOverhead), PreemptGraceS: Secs(c.PreemptGrace)}
	for _, gc := range c.Classes {
		w.Classes = append(w.Classes, wireClass(gc))
	}
	for _, n := range v.Nodes {
		w.Nodes = append(w.Nodes, wireNodeStatic{Name: n.Name, Rack: n.Rack, Class: n.Class, Speed: n.Speed, GPUs: n.GPUs, CPUs: n.CPUs, MemMB: n.MemMB})
	}
	return w
}

func viewOf(v *api.View, histFrom int) wireView {
	w := wireView{NowS: Secs(v.Now), Nodes: make([]wireNode, len(v.Nodes)), Running: make([]wireRunning, len(v.Running)),
		Pending: make([]wirePending, len(v.Pending)), Tenants: make([]wireTenant, len(v.Tenants)), HistoryNew: []wireCompleted{}}
	for i, n := range v.Nodes {
		sh := make([]wireShare, len(n.Running))
		for k, s := range n.Running {
			sh[k] = wireShare(s)
		}
		w.Nodes[i] = wireNode{Name: n.Name, FreeGPUs: n.FreeGPUs, FreeCPUs: n.FreeCPUs, FreeMemMB: n.FreeMemMB, Running: sh}
	}
	for i, r := range v.Running {
		w.Running[i] = wireRunning{JobID: r.JobID, Tenant: r.Tenant, User: r.User, Priority: r.Priority, GPUs: r.GPUs, Workers: r.Workers,
			CPUs: r.CPUs, MemMB: r.MemMB, GPUClass: r.GPUClass, Topology: string(r.Topology), SubmitS: Secs(r.Submit),
			FirstStartS: Secs(r.FirstStart), RunStartS: Secs(r.RunStart), OverheadS: Secs(r.Overhead), Placement: names(r.Placement, v.Nodes),
			Rate: r.Rate, EstimateS: Secs(r.Estimate), RetainedAtStartS: r.RetainedAtStart, WorkDoneS: r.WorkDone,
			EstRemainingWorkS: r.EstRemainingWork, EstEndS: Secs(r.EstEnd), Preemptible: r.Preemptible,
			CheckpointInterval: Secs(r.CheckpointInterval), Preemptions: r.Preemptions}
	}
	for i, p := range v.Pending {
		w.Pending[i] = wirePending{JobID: p.JobID, Tenant: p.Tenant, User: p.User, SubmitS: Secs(p.Submit), Priority: p.Priority,
			GPUs: p.GPUs, Workers: p.Workers, CPUs: p.CPUs, MemMB: p.MemMB, GPUClass: p.GPUClass, Topology: string(p.Topology),
			EstimateS: Secs(p.Estimate), WaitS: Secs(p.Wait), RetainedS: p.Retained, Preemptible: p.Preemptible,
			CheckpointInterval: Secs(p.CheckpointInterval), Preemptions: p.Preemptions, Started: p.Started}
		if p.HasMaxWait {
			mw := Secs(p.MaxWait)
			w.Pending[i].MaxWaitS = &mw
		}
	}
	for i, t := range v.Tenants {
		w.Tenants[i] = wireTenant{Tenant: t.Tenant, GPUSeconds: t.GPUSeconds, RunningGPUs: t.RunningGPUs, RunningCPUs: t.RunningCPUs,
			RunningMemMB: t.RunningMem}
	}
	for _, h := range v.History[min(histFrom, len(v.History)):] {
		w.HistoryNew = append(w.HistoryNew, wireCompleted{JobID: h.JobID, Tenant: h.Tenant, User: h.User, SubmitS: Secs(h.Submit),
			RuntimeS: Secs(h.Runtime), EstimateS: Secs(h.Estimate), FinishS: Secs(h.Finish)})
	}
	return w
}

// decisionOf converts a reply into a Decision; node names map to indices.
func decisionOf(r *decisionReply, v *api.View) (api.Decision, error) {
	idx := make(map[string]int, len(v.Nodes))
	for i := range v.Nodes {
		idx[v.Nodes[i].Name] = i
	}
	var d api.Decision
	for k, a := range r.Actions {
		act := api.Action{Op: api.Op(a.Op), JobID: a.JobID}
		for _, nw := range a.Placement {
			n, ok := idx[nw.Node]
			if !ok {
				// keep the index invalid so that the simulator rejects it with context
				return d, fmt.Errorf("action #%d (%s %s): unknown node %q", k, a.Op, a.JobID, nw.Node)
			}
			act.Placement = append(act.Placement, api.NodeWorkers{Node: n, Workers: nw.Workers})
		}
		d.Actions = append(d.Actions, act)
	}
	if r.WakeAtS != nil {
		d.WakeAt = time.Duration(*r.WakeAtS)
	}
	for _, res := range r.Reservations {
		d.Reservations = append(d.Reservations, api.Reservation{JobID: res.JobID, Shadow: time.Duration(res.ShadowS)})
	}
	if s := r.Solver; s != nil {
		d.Solver = &api.SolverInfo{Status: s.Status, WallSolve: s.SolveS, CPUSolve: s.CPUS, Gap: s.Gap, Capped: s.Capped}
	}
	return d, nil
}
