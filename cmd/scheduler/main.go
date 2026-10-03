// Command scheduler runs one simulation: a cluster configuration, a trace
// (a schema-v1 CSV file, or one generated from a workload configuration and a
// seed), and one policy. It prints the run's metrics as JSON. All results are
// simulated with assumed parameters.
//
//	go run ./cmd/scheduler -cluster configs/clusters/homogeneous.json \
//	    -workload configs/workloads/base.json -seed 1 -policy fifo+first_fit+easy
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/exporter"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

func main() {
	var (
		clusterPath = flag.String("cluster", "configs/clusters/homogeneous.json", "cluster configuration v1")
		tracePath   = flag.String("trace", "", "trace CSV (schema v1); a manifest beside it is checked when present")
		workload    = flag.String("workload", "configs/workloads/base.json", "workload configuration (used when -trace is empty)")
		seed        = flag.Uint64("seed", 1, "seed for the generated trace and the policy")
		jobs        = flag.Int("jobs", 0, "override the workload's job count (0: keep)")
		name        = flag.String("policy", "fifo+first_fit+easy", "policy name: order+placement+backfill[+preempt], or py:<name>")
		params      = flag.String("params", "", "policy parameters as a JSON object")
		save        = flag.String("save-trace", "", "write the generated trace (and its manifest) to this CSV path")
		logPath     = flag.String("log", "", "write the assignment log (time, op, job, placement) as CSV to this path")
		speedup     = flag.Float64("replay-speedup", 0, "replay mode: pace the simulation at this many simulated seconds per wall second and serve /metrics (0 = off)")
		port        = flag.Int("metrics-port", 18200, "replay mode: port of the Prometheus /metrics endpoint on 127.0.0.1 (0 = any free port)")
	)
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(*clusterPath, *tracePath, *workload, *seed, *jobs, *name, *params, *save, *logPath, *speedup, *port, log); err != nil {
		log.Error("run failed", "err", err)
		os.Exit(1)
	}
}

func run(clusterPath, tracePath, workload string, seed uint64, jobsOverride int, name, params, save, logPath string,
	speedup float64, port int, log *slog.Logger) error {
	cl, err := cluster.Load(clusterPath)
	if err != nil {
		return err
	}
	var jobs []api.Job
	switch {
	case tracePath != "":
		if _, statErr := os.Stat(trace.ManifestPath(tracePath)); statErr == nil {
			jobs, _, err = trace.LoadVerified(tracePath)
		} else {
			jobs, err = trace.LoadFile(tracePath)
		}
	default:
		var g trace.GenConfig
		data, rerr := os.ReadFile(workload)
		if rerr != nil {
			return rerr
		}
		if err = json.Unmarshal(data, &g); err != nil {
			return fmt.Errorf("%s: %w", workload, err)
		}
		g.WorkCapacity = cl.WorkCapacity()
		if jobsOverride > 0 {
			g.Jobs = jobsOverride
		}
		jobs, err = trace.Generate(g, seed)
		if err == nil && save != "" {
			_, err = trace.SaveWithManifest(save, jobs, g.Info(), seed)
		}
	}
	if err != nil {
		return err
	}
	truth := map[string]time.Duration{}
	for _, j := range jobs {
		truth[j.ID] = j.Runtime
	}
	root, _ := extpolicy.RepoRoot()
	spec := policy.Spec{Name: name}
	if params != "" {
		spec.Params = json.RawMessage(params)
	}
	p, err := policy.New(spec, policy.Env{
		Oracle: func(id string) (time.Duration, bool) { r, ok := truth[id]; return r, ok },
		External: func(s policy.Spec) (policy.Policy, error) {
			return extpolicy.New(s.Name[len("py:"):], s.Params, extpolicy.Config{Dir: root, Stderr: os.Stderr, Seed: seed}), nil
		},
	})
	if err != nil {
		return err
	}
	var onSample func(metrics.Sample)
	if speedup > 0 {
		exp := exporter.New(cl.Info.TotalGPUs, speedup, time.Sleep)
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err != nil {
			return err
		}
		srv := &http.Server{Handler: exp, ReadHeaderTimeout: 5 * time.Second}
		go func() { _ = srv.Serve(ln) }()
		defer srv.Close()
		log.Info("replay: serving /metrics", "addr", "http://"+ln.Addr().String()+"/metrics", "speedup", speedup)
		onSample = exp.Observe
	}
	start := time.Now()
	rt, err := simulator.Run(simulator.Config{Cluster: cl, Jobs: jobs, Policy: p, Wall: time.Now, Log: logPath != "", OnSample: onSample})
	if c, ok := p.(policy.Closer); ok {
		if cerr := c.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}
	if err != nil {
		return err
	}
	if err := metrics.Verify(rt, cl.EmptyNodes()); err != nil {
		return fmt.Errorf("invariant violated: %w", err)
	}
	if err := metrics.CheckIdentities(rt); err != nil {
		return fmt.Errorf("accounting identity violated: %w", err)
	}
	if logPath != "" {
		if err := writeLog(logPath, rt.Log); err != nil {
			return err
		}
	}
	out := map[string]any{"policy": name, "cluster": cl.Name, "jobs": len(jobs), "seed": seed, "simulated": true,
		"wall_run_s": time.Since(start).Seconds()}
	for _, f := range append(metrics.Compute(rt).Fields, metrics.Wall(rt).Fields...) {
		if math.IsNaN(f.Value) || math.IsInf(f.Value, 0) {
			out[f.Name] = nil
		} else {
			out[f.Name] = f.Value
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func writeLog(path string, log []metrics.LogEntry) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintln(f, "t_s,op,job_id,placement")
	for _, e := range log {
		pl := ""
		for i, nw := range e.Placement {
			if i > 0 {
				pl += ";"
			}
			pl += fmt.Sprintf("%s:%d", nw.Node, nw.Workers)
		}
		fmt.Fprintf(f, "%.3f,%s,%s,%s\n", float64(e.T.Milliseconds())/1000, e.Op, e.JobID, pl)
	}
	return nil
}
