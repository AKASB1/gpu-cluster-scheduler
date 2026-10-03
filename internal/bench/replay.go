package bench

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
)

// ErrReplayData means the public trace of a replay scenario has not been
// downloaded (it is not committed): the scenario is skipped.
var ErrReplayData = errors.New("replay data not found (run scripts/fetch_helios.sh or scripts\\fetch_helios.ps1 first)")

// ReplayConfig selects an excerpt of a public cluster log. Tuning seeds use
// the tuning window, evaluation seeds the evaluation window; the seed only
// drives the synthetic estimates.
type ReplayConfig struct {
	Format      string               `json:"format"` // "helios"
	Log         string               `json:"log"`
	Capacity    string               `json:"capacity"`
	Start       string               `json:"start"`
	TuningStart string               `json:"tuning_start"`
	Days        float64              `json:"days"`
	Estimate    trace.EstimateConfig `json:"estimate"`
}

const heliosTime = "2006-01-02 15:04:05"

// ReplayTrace converts the window that belongs to seed and derives the
// simulated cluster from the log's own capacity on the window's first day.
func ReplayTrace(root string, rc *ReplayConfig, seed uint64, tuning bool) ([]api.Job, cluster.Config, json.RawMessage, error) {
	var cc cluster.Config
	if rc.Format != "helios" {
		return nil, cc, nil, fmt.Errorf("replay: unknown format %q", rc.Format)
	}
	startS := rc.Start
	if tuning {
		startS = rc.TuningStart
	}
	start, err := time.Parse(heliosTime, startS)
	if err != nil {
		return nil, cc, nil, fmt.Errorf("replay: start: %w", err)
	}
	logPath := filepath.Join(root, rc.Log)
	f, err := os.Open(logPath)
	if err != nil {
		return nil, cc, nil, fmt.Errorf("%w: %v", ErrReplayData, err)
	}
	defer f.Close()
	jobs, st, err := trace.ConvertHelios(bufio.NewReaderSize(f, 1<<20), trace.HeliosOptions{Start: start,
		Length: time.Duration(rc.Days * 24 * float64(time.Hour)), GPUsPerNode: 8, Estimate: rc.Estimate, Seed: seed})
	if err != nil {
		return nil, cc, nil, err
	}
	vcs, err := capacitiesOn(filepath.Join(root, rc.Capacity), start)
	if err != nil {
		return nil, cc, nil, fmt.Errorf("%w: %v", ErrReplayData, err)
	}
	// as in the original cluster, a job runs only in its own virtual cluster
	// (a GPU class of the derived cluster); jobs of a VC without GPUs on that
	// day stay infeasible and are excluded by the simulator
	for i := range jobs {
		jobs[i].GPUClass = jobs[i].Tenant
	}
	cc = DerivedCluster("helios-derived", vcs, jobs)
	params, _ := json.Marshal(map[string]any{"format": rc.Format, "log": filepath.Base(rc.Log), "log_sha256": FileSHA256(logPath),
		"start": startS, "days": rc.Days, "gpus_per_node": 8, "estimate": rc.Estimate, "synthetic_estimates": true,
		"license": "CC-BY-4.0, S-Lab-System-Group/HeliosData", "rows": st.Rows, "in_window": st.InWindow,
		"dropped_non_gpu": st.NonGPU, "dropped_zero_duration": st.ZeroDuration, "kept": st.Kept})
	return jobs, cc, params, nil
}

// capacitiesOn returns the GPUs of every virtual cluster (VC) with a
// positive count in a Helios cluster_gpu_number.csv on the last date not
// after t (the columns other than date and total are VCs).
func capacitiesOn(path string, t time.Time) (map[string]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	head, err := r.Read()
	if err != nil {
		return nil, err
	}
	if len(head) < 3 || head[0] != "date" || head[len(head)-1] != "total" {
		return nil, fmt.Errorf("%s: need the columns date, one per VC, and total", path)
	}
	var best []string
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		d, err := time.Parse("2006-01-02", rec[0])
		if err != nil {
			return nil, err
		}
		if d.After(t) {
			break
		}
		best = rec
	}
	if best == nil {
		return nil, fmt.Errorf("%s: no capacity on or before %s", path, t.Format("2006-01-02"))
	}
	out := map[string]int{}
	for i := 1; i < len(head)-1; i++ {
		g, err := strconv.Atoi(best[i])
		if err != nil {
			return nil, fmt.Errorf("%s: %s: %w", path, head[i], err)
		}
		if g > 0 {
			out[head[i]] = g
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no GPUs on %s", path, best[0])
	}
	return out, nil
}

// DerivedCluster is the simulated cluster of a replay (an assumption): one
// partition per virtual cluster with that VC's GPUs on the window's first
// day, modelled as a GPU class of speed 1.0 named after the VC; nodes of 8
// GPUs (the last node of a VC holds the remainder), racks of up to 8 nodes
// within a VC; CPUs per node enough for the largest per-node CPU request of
// the excerpt (at least 64), 1024 GB memory (the log has none), and the
// default penalties and restart overhead.
func DerivedCluster(name string, vcs map[string]int, jobs []api.Job) cluster.Config {
	cpus := 64
	for _, j := range jobs {
		perNode := min(j.Workers, max(1, 8/min(j.GPUs, 8)))
		cpus = max(cpus, j.CPUs*perNode)
	}
	c := cluster.Config{SchemaVersion: 1, Name: name, CrossNodeFactor: 1.1, CrossRackFactor: 1.25, RestartOverheadS: 120,
		Sources: map[string]string{"nodes": "derived: per virtual cluster, the log's GPUs on the first day of the excerpt, 8 GPUs per node (assumed)",
			"classes": "one class per virtual cluster, speed 1.0 (assumed); a job runs only in its own virtual cluster, as in the original cluster",
			"cpus":    "assumed: enough for the largest per-node CPU request of the excerpt (at least 64)",
			"memory":  "assumed: 1024 GB per node (the log has no memory requests)", "factors": "assumed, as configs/clusters/homogeneous.json"}}
	names := make([]string, 0, len(vcs))
	for vc := range vcs {
		names = append(names, vc)
	}
	slices.Sort(names)
	for _, vc := range names {
		c.Classes = append(c.Classes, api.GPUClass{Name: vc, Speed: 1})
		full, rem := vcs[vc]/8, vcs[vc]%8
		nodes := full
		if rem > 0 {
			nodes++
		}
		for r := 0; r*8 < nodes; r++ {
			n := min(8, nodes-r*8)
			rack := fmt.Sprintf("%s-r%d", vc, r)
			g := cluster.NodeGroup{Count: n, Prefix: rack + "-n", Rack: rack, Class: vc, GPUs: 8, CPUs: cpus, MemGB: 1024}
			if rem > 0 && r*8+n == nodes { // the remainder node is the last one: a group of its own
				if g.Count--; g.Count > 0 {
					c.NodeGroups = append(c.NodeGroups, g)
				}
				c.NodeGroups = append(c.NodeGroups, cluster.NodeGroup{Count: 1, Prefix: rack + "-m", Rack: rack, Class: vc, GPUs: rem,
					CPUs: cpus, MemGB: 1024})
				continue
			}
			c.NodeGroups = append(c.NodeGroups, g)
		}
	}
	return c
}

// replayTraces is Traces for a replay scenario: converted (not generated)
// traces, saved with manifests and replayed from the files; infeasible jobs
// are reported and excluded by the simulator.
func (env *Env) replayTraces(sc *Scenario, seeds []uint64) (map[uint64]*TraceSet, error) {
	out := map[uint64]*TraceSet{}
	for _, seed := range seeds {
		jobs, cc, params, err := ReplayTrace(env.Root, sc.Replay, seed, env.TuningSeeds[seed])
		if err != nil {
			return nil, err
		}
		cl, err := cluster.Build(cc)
		if err != nil {
			return nil, err
		}
		path := filepath.Join(env.Outputs, "traces", sc.ID, fmt.Sprintf("seed-%d.csv", seed))
		if _, err := trace.SaveWithManifest(path, jobs, trace.GeneratorInfo{Name: "helios-converter", Version: 1, Params: params}, seed); err != nil {
			return nil, err
		}
		cb, _ := json.MarshalIndent(cc, "", "  ")
		if err := os.WriteFile(filepath.Join(env.Outputs, "traces", sc.ID, fmt.Sprintf("seed-%d.cluster.json", seed)), cb, 0o644); err != nil {
			return nil, err
		}
		loaded, _, err := trace.LoadVerified(path)
		if err != nil {
			return nil, err
		}
		infeasible := 0
		empty := cl.EmptyNodes()
		for i := range loaded {
			j := &loaded[i]
			if _, ok := metrics.Classify(metrics.Shape{GPUs: j.GPUs, CPUs: j.CPUs, MemMB: j.MemMB(), Workers: j.Workers, Class: j.GPUClass}, empty); !ok {
				infeasible++
			}
		}
		env.logf("%s seed %d: %d replayed jobs on %d GPUs (%d infeasible, excluded)", sc.ID, seed, len(loaded), cl.Info.TotalGPUs, infeasible)
		out[seed] = &TraceSet{Jobs: loaded, Cluster: cl, SLO: false, Fair: true}
	}
	return out, nil
}
