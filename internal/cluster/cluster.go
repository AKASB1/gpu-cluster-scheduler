// Package cluster loads and validates cluster configuration v1
// (docs/contracts.md §3) and builds the static part of the scheduling view.
package cluster

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// SchemaVersion is the cluster configuration version.
const SchemaVersion = 1

// NodeSpec is one node of the configuration.
type NodeSpec struct {
	Name  string  `json:"name"`
	Rack  string  `json:"rack"`
	Class string  `json:"class"`
	GPUs  int     `json:"gpus"`
	CPUs  int     `json:"cpus"`
	MemGB float64 `json:"mem_gb"`
}

// NodeGroup is Count nodes named <Prefix><index> (two digits, from 0) built
// from one template.
type NodeGroup struct {
	Count  int     `json:"count"`
	Prefix string  `json:"prefix"`
	Rack   string  `json:"rack"`
	Class  string  `json:"class"`
	GPUs   int     `json:"gpus"`
	CPUs   int     `json:"cpus"`
	MemGB  float64 `json:"mem_gb"`
}

// Config is cluster configuration v1 as stored in JSON.
type Config struct {
	SchemaVersion    int               `json:"schema_version"`
	Name             string            `json:"name"`
	Classes          []api.GPUClass    `json:"classes"`
	Nodes            []NodeSpec        `json:"nodes,omitempty"`
	NodeGroups       []NodeGroup       `json:"node_groups,omitempty"`
	CrossNodeFactor  float64           `json:"cross_node_factor"`
	CrossRackFactor  float64           `json:"cross_rack_factor"`
	RestartOverheadS float64           `json:"restart_overhead_s"`
	PreemptGraceS    float64           `json:"preempt_grace_s"`
	Sources          map[string]string `json:"sources,omitempty"`
}

// Cluster is a validated configuration with nodes sorted by (rack, name).
type Cluster struct {
	Name  string
	Nodes []NodeSpec
	Info  api.ClusterInfo
	speed map[string]float64
}

// Load reads and validates a cluster configuration file.
func Load(path string) (*Cluster, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}

// Parse validates configuration bytes (unknown fields are rejected).
func Parse(data []byte) (*Cluster, error) {
	var c Config
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("cluster config: %w", err)
	}
	return Build(c)
}

// Build validates a configuration value.
func Build(c Config) (*Cluster, error) {
	fail := func(format string, a ...any) (*Cluster, error) {
		return nil, fmt.Errorf("cluster config %q: "+format, append([]any{c.Name}, a...)...)
	}
	if c.SchemaVersion != SchemaVersion {
		return fail("unknown schema_version %d", c.SchemaVersion)
	}
	if len(c.Classes) == 0 {
		return fail("no GPU classes")
	}
	speed := map[string]float64{}
	for _, gc := range c.Classes {
		if gc.Name == "" || speed[gc.Name] != 0 {
			return fail("class names must be non-empty and unique (%q)", gc.Name)
		}
		if !(gc.Speed > 0) {
			return fail("class %q: speed must be > 0", gc.Name)
		}
		speed[gc.Name] = gc.Speed
	}
	nodes := slices.Clone(c.Nodes)
	for _, g := range c.NodeGroups {
		if g.Count < 1 {
			return fail("node group %q: count must be >= 1", g.Prefix)
		}
		for i := 0; i < g.Count; i++ {
			nodes = append(nodes, NodeSpec{Name: fmt.Sprintf("%s%02d", g.Prefix, i), Rack: g.Rack, Class: g.Class,
				GPUs: g.GPUs, CPUs: g.CPUs, MemGB: g.MemGB})
		}
	}
	if len(nodes) == 0 {
		return fail("no nodes")
	}
	names := map[string]bool{}
	for _, n := range nodes {
		switch {
		case n.Name == "" || names[n.Name]:
			return fail("node names must be non-empty and unique (%q)", n.Name)
		case n.Rack == "":
			return fail("node %q: rack required", n.Name)
		case speed[n.Class] == 0:
			return fail("node %q: unknown class %q", n.Name, n.Class)
		case n.GPUs < 1 || n.CPUs < 1 || !(n.MemGB > 0):
			return fail("node %q: gpus, cpus, and mem_gb must be positive", n.Name)
		}
		names[n.Name] = true
	}
	if !(c.CrossNodeFactor >= 1 && c.CrossRackFactor >= c.CrossNodeFactor) {
		return fail("need 1 <= cross_node_factor <= cross_rack_factor")
	}
	overhead, err := seconds(c.RestartOverheadS)
	if err != nil {
		return fail("restart_overhead_s: %v", err)
	}
	grace, err := seconds(c.PreemptGraceS)
	if err != nil {
		return fail("preempt_grace_s: %v", err)
	}
	slices.SortFunc(nodes, func(a, b NodeSpec) int {
		if r := cmp.Compare(a.Rack, b.Rack); r != 0 {
			return r
		}
		return cmp.Compare(a.Name, b.Name)
	})
	info := api.ClusterInfo{Classes: slices.Clone(c.Classes), CrossNode: c.CrossNodeFactor, CrossRack: c.CrossRackFactor,
		RestartOverhead: overhead, PreemptGrace: grace}
	slices.SortFunc(info.Classes, func(a, b api.GPUClass) int { return cmp.Compare(a.Name, b.Name) })
	for _, n := range nodes {
		if len(info.Racks) == 0 || info.Racks[len(info.Racks)-1] != n.Rack {
			info.Racks = append(info.Racks, n.Rack)
		}
		info.TotalGPUs += n.GPUs
		info.TotalCPUs += n.CPUs
		info.TotalMemMB += api.MB(n.MemGB)
	}
	return &Cluster{Name: c.Name, Nodes: nodes, Info: info, speed: speed}, nil
}

func seconds(s float64) (time.Duration, error) {
	if !(s >= 0) {
		return 0, fmt.Errorf("must be >= 0")
	}
	d := clock.RoundMs(s)
	if clock.Sec(d) != s {
		return 0, fmt.Errorf("at most three decimals")
	}
	return d, nil
}

// Speed returns the speed of a class (0 if unknown).
func (c *Cluster) Speed(class string) float64 { return c.speed[class] }

// HasClass reports whether the class exists.
func (c *Cluster) HasClass(class string) bool { return c.speed[class] > 0 }

// WorkCapacity is the sum over nodes of GPUs times class speed: the
// reference-GPU-seconds of work the cluster can do per second.
func (c *Cluster) WorkCapacity() float64 {
	w := 0.0
	for _, n := range c.Nodes {
		w += float64(n.GPUs) * c.speed[n.Class]
	}
	return w
}

// EmptyNodes returns the node states of the empty cluster.
func (c *Cluster) EmptyNodes() []api.NodeState {
	out := make([]api.NodeState, len(c.Nodes))
	for i, n := range c.Nodes {
		out[i] = api.NodeState{Index: i, Name: n.Name, Rack: n.Rack, Class: n.Class, Speed: c.speed[n.Class],
			GPUs: n.GPUs, CPUs: n.CPUs, MemMB: api.MB(n.MemGB), FreeGPUs: n.GPUs, FreeCPUs: n.CPUs, FreeMemMB: api.MB(n.MemGB)}
	}
	return out
}

// FastestSpeed returns the speed of the fastest class a job may use (its
// class if constrained, else the fastest class that has nodes).
func (c *Cluster) FastestSpeed(gpuClass string) float64 {
	if gpuClass != "" {
		return c.speed[gpuClass]
	}
	best := 0.0
	for _, n := range c.Nodes {
		best = max(best, c.speed[n.Class])
	}
	return best
}

// WithFactors returns a copy whose topology penalties (f - 1) and restart
// overhead are scaled (used by the sensitivity sweep).
func (c *Cluster) WithFactors(penaltyScale, overheadScale float64) *Cluster {
	out := *c
	out.Info = c.Info
	// Rounded to 1e-9 so that scaled factors stay readable (1.2, not 1.2000000000000002).
	out.Info.CrossNode = math.Round((1+(c.Info.CrossNode-1)*penaltyScale)*1e9) / 1e9
	out.Info.CrossRack = math.Round((1+(c.Info.CrossRack-1)*penaltyScale)*1e9) / 1e9
	out.Info.RestartOverhead = clock.RoundMs(clock.Sec(c.Info.RestartOverhead) * overheadScale)
	return &out
}
