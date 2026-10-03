// Command benchmark runs the simulated benchmark of docs and
// benchmarks/README.md.
//
//	go run ./cmd/benchmark -tune     # tuning on the tuning seeds -> configs/tuned/tuned.json
//	go run ./cmd/benchmark           # full evaluation with the frozen tuning -> benchmarks/results/<name>/
//	go run ./cmd/benchmark -quick    # smoke subset -> benchmarks/outputs/quick/
//
// All results are simulated with assumed parameters.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/bench"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

func main() {
	var (
		config    = flag.String("config", "configs/experiments/benchmark.json", "experiment configuration")
		tunedPath = flag.String("tuned", "configs/tuned/tuned.json", "frozen tuning result")
		tune      = flag.Bool("tune", false, "run the tuning on the tuning seeds and write -tuned")
		tuneAdd   = flag.Bool("tune-add", false, "tune only scenarios missing from -tuned; frozen entries stay unchanged")
		jterm     = flag.Bool("jterm-check", false, "check on the tuning seeds that every J term varies across policies by more than its seed-to-seed noise; writes jterm_check.csv beside -tuned")
		quick     = flag.Bool("quick", false, "run the quick smoke subset (about 3 minutes)")
		only      = flag.String("scenarios", "", "comma-separated scenario ids (default: all)")
		workers   = flag.Int("workers", max(1, runtime.NumCPU()/2), "parallel runs (default: half of the logical processors)")
		out       = flag.String("out", "", "results folder (default benchmarks/results/<name>, quick: benchmarks/outputs/quick/results)")
		outputs   = flag.String("outputs", "", "ignored outputs folder for traces, logs, per-run JSON (default benchmarks/outputs/<name>)")
		timeout   = flag.Duration("timeout", extpolicy.DefaultTimeout, "external-policy call timeout")
		plots     = flag.Bool("plots", true, "write the figures with scripts/plot_results.py after a full evaluation")
		label     = flag.String("label", "", "free-text label stored in the manifest")
		loadNote  = flag.String("load-note", "shared workstation: other processes (another agent run) may be active; wall_ values are indicative only",
			"machine-load note stored in the manifest")
	)
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := runMain(*config, *tunedPath, *tune, *tuneAdd, *jterm, *quick, *only, *workers, *out, *outputs, *timeout, *plots, *label, *loadNote, log); err != nil {
		log.Error("benchmark failed", "err", err)
		os.Exit(1)
	}
}

func runMain(config, tunedPath string, tune, tuneAdd, jterm, quick bool, only string, workers int, out, outputs string, timeout time.Duration,
	plots bool, label, loadNote string, log *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	root, err := extpolicy.RepoRoot()
	if err != nil {
		return err
	}
	if err := os.Chdir(root); err != nil {
		return err
	}
	exp, err := bench.LoadExperiment(config)
	if err != nil {
		return err
	}
	mode := "full"
	switch {
	case tune || tuneAdd:
		mode = "tune"
	case quick:
		mode = "quick"
	}
	if outputs == "" {
		outputs = filepath.Join("benchmarks", "outputs", exp.Name)
		if quick {
			outputs = filepath.Join("benchmarks", "outputs", "quick")
		}
	}
	if out == "" {
		out = filepath.Join("benchmarks", "results", exp.Name)
		if quick {
			out = filepath.Join("benchmarks", "outputs", "quick", "results")
		}
	}
	env := &bench.Env{Root: root, Outputs: outputs, Workers: workers, Timeout: timeout,
		Log: func(f string, a ...any) { log.Info(fmt.Sprintf(f, a...)) }, TuningSeeds: map[uint64]bool{}}
	for _, s := range exp.TuningSeeds {
		env.TuningSeeds[s] = true
	}
	files := exp.Files()
	hash, err := bench.ConfigHash(root, files)
	if err != nil {
		return err
	}
	start := time.Now()
	commit, dirty := bench.GitState(ctx, root)
	man := &bench.Manifest{Experiment: exp.Name, Mode: mode, Command: "go run ./cmd/benchmark " + strings.Join(os.Args[1:], " "),
		Commit: commit, Dirty: dirty, ConfigSHA256: hash, ConfigFiles: files, TuningSeeds: exp.TuningSeeds, EvalSeeds: exp.EvalSeeds,
		Workers: workers, Versions: bench.ProbeVersions(ctx, root), Hardware: bench.ProbeHardware(ctx), LoadNote: loadNote,
		WallStart: start.UTC().Format(time.RFC3339), Label: label}
	log.Info("benchmark", "mode", mode, "experiment", exp.Name, "workers", workers, "commit", commit, "dirty", dirty,
		"python", man.Versions.Python, "scipy", man.Versions.SciPy, "highs", man.Versions.HiGHS)

	if jterm {
		tuned, err := bench.LoadTuned(tunedPath)
		if err != nil {
			return err
		}
		if tuned.ConfigSHA256 != hash {
			return errors.New("the experiment configuration changed after tuning (config_sha256 differs): re-run the tuning")
		}
		table, err := env.JTermCheck(ctx, exp, tuned)
		if err != nil {
			return err
		}
		path := filepath.Join(filepath.Dir(tunedPath), "jterm_check.csv")
		if err := bench.WriteCSV(path, table); err != nil {
			return err
		}
		for _, r := range table[1:] {
			if r[5] == "NOT informative" {
				log.Warn("J term not informative", "scenario", r[0], "term", r[1], "spread", r[3], "noise", r[4])
			}
		}
		log.Info("J-term check written", "path", path, "seconds", int(time.Since(start).Seconds()))
		return nil
	}
	if tune || tuneAdd {
		var base *bench.Tuned
		if tuneAdd {
			if base, err = bench.LoadTuned(tunedPath); err != nil {
				return err
			}
		}
		t, err := env.Tune(ctx, exp, hash, base)
		if err != nil {
			return err
		}
		if err := t.Save(tunedPath); err != nil {
			return err
		}
		man.WallSeconds = time.Since(start).Seconds()
		log.Info("tuning written", "path", tunedPath, "defaults", fmt.Sprintf("%+v", t.Defaults), "seconds", int(man.WallSeconds))
		name := "tuning_manifest.json"
		if tuneAdd {
			name = "tuning_add_manifest.json"
		}
		return man.Save(filepath.Join(filepath.Dir(tunedPath), name))
	}

	full := !quick && only == ""
	opts := bench.EvalOptions{Sweep: full, Capacity: full}
	if only != "" {
		opts.Scenarios = strings.Split(only, ",")
	}
	tuned, err := bench.LoadTuned(tunedPath)
	switch {
	case err == nil:
		if tuned.ConfigSHA256 != hash {
			msg := "the experiment configuration changed after tuning (config_sha256 differs): re-run the tuning"
			if !quick {
				return errors.New(msg)
			}
			log.Warn(msg + "; quick mode continues")
		}
		man.TunedSHA256, man.TunedCommit = bench.FileSHA256(tunedPath), bench.LastCommitOf(ctx, root, tunedPath)
	case quick:
		log.Warn("no frozen tuning: quick mode uses default parts and references from its own baseline runs", "err", err)
		if tuned, err = quickFallback(ctx, env, exp); err != nil {
			return err
		}
	default:
		return err
	}
	if quick {
		opts.Scenarios, opts.Seeds, opts.Instances = exp.Quick.Scenarios, exp.Quick.Seeds, exp.Quick.Instances
		env.JobsOverride = exp.Quick.Jobs
		man.EvalSeeds = exp.Quick.Seeds
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	if err := env.Evaluate(ctx, exp, tuned, opts, out); err != nil {
		return err
	}
	if err := decisionCost(ctx, out, quick, log); err != nil {
		return err
	}
	man.WallSeconds = time.Since(start).Seconds()
	man.Skipped = env.Skipped
	if err := man.Save(filepath.Join(out, "manifest.json")); err != nil {
		return err
	}
	log.Info("results written", "folder", out, "seconds", int(man.WallSeconds))
	if plots && !quick {
		cmd := exec.CommandContext(ctx, extpolicy.Python(), filepath.Join("scripts", "plot_results.py"), "--results", out)
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		cmd.Env = append(os.Environ(), "PYTHONUTF8=1")
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("plots: %w", err)
		}
	}
	return nil
}

// quickFallback builds a stand-in tuning for the smoke run when no frozen
// tuning exists: default parts and references from the baseline's own runs.
func quickFallback(ctx context.Context, env *bench.Env, exp *bench.Experiment) (*bench.Tuned, error) {
	t := &bench.Tuned{Experiment: exp.Name + " (quick fallback, not frozen)", Defaults: bench.Defaults{Order: "fifo", Placement: "first_fit", Backfill: "easy"},
		Scenarios: map[string]*bench.ScenarioTuning{}}
	env.JobsOverride = exp.Quick.Jobs
	defer func() { env.JobsOverride = 0 }()
	for _, id := range exp.Quick.Scenarios {
		sc := exp.Scenario(id)
		if sc == nil {
			return nil, fmt.Errorf("quick scenario %q is unknown", id)
		}
		st := &bench.ScenarioTuning{Reference: bench.Reference{Rs: 1, Rw: 1}, Params: map[string]json.RawMessage{}}
		t.Scenarios[id] = st
		if sc.Kind != bench.KindSim {
			continue
		}
		ts, err := env.Traces(sc, exp.Quick.Seeds)
		if err != nil {
			return nil, err
		}
		var tasks []bench.Task
		for _, s := range exp.Quick.Seeds {
			tasks = append(tasks, bench.Task{Scenario: id, Policy: exp.Baseline, Seed: s, Trace: ts[s], Ref: bench.Reference{Rs: 1, Rw: 1}})
		}
		rows, err := env.RunTasks(ctx, tasks)
		if err != nil {
			return nil, err
		}
		var rs, rw []float64
		for _, r := range rows {
			rs, rw = append(rs, r.Get("bsld_p95")), append(rw, r.Get("wait_p95"))
		}
		st.Reference = bench.Reference{Rs: metrics.Mean(rs), Rw: max(1, metrics.Mean(rw))}
	}
	return t, nil
}

// decisionCost runs the decision-cost microbenchmark in a separate,
// otherwise idle process:
//
//	go test -run '^$' -bench BenchmarkSchedule -benchmem -count 5 ./internal/policy
//
// and writes the parsed numbers to wall_decision_cost.csv (wall-clock values).
// Quick mode runs only the smallest queue length.
func decisionCost(ctx context.Context, out string, quick bool, log *slog.Logger) error {
	pattern := "BenchmarkSchedule"
	if quick {
		pattern = "BenchmarkSchedule/.*/pending=64$"
	}
	cmd := exec.CommandContext(ctx, "go", "test", "-run", "^$", "-bench", pattern, "-benchmem", "-benchtime", "1s", "-count", "5", "./internal/policy")
	raw, err := cmd.CombinedOutput()
	if err := os.WriteFile(filepath.Join(out, "wall_decision_cost.txt"), raw, 0o644); err != nil {
		return err
	}
	if err != nil {
		return fmt.Errorf("decision-cost benchmark: %w: %s", err, raw)
	}
	re := regexp.MustCompile(`^BenchmarkSchedule/(\S+)/pending=(\d+)-\d+\s+(\d+)\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op`)
	// Five repetitions per case (-count 5) on a shared machine: report the
	// minimum, which is the least disturbed measurement, with the median and
	// the maximum next to it.
	type sample struct{ ns, bytes, allocs []float64 }
	var keys []string
	cases := map[string]*sample{}
	for _, line := range strings.Split(string(raw), "\n") {
		if m := re.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			k := m[1] + "," + m[2]
			if cases[k] == nil {
				cases[k] = &sample{}
				keys = append(keys, k)
			}
			ns, _ := strconv.ParseFloat(m[4], 64)
			b, _ := strconv.ParseFloat(m[5], 64)
			a, _ := strconv.ParseFloat(m[6], 64)
			cases[k].ns, cases[k].bytes, cases[k].allocs = append(cases[k].ns, ns), append(cases[k].bytes, b), append(cases[k].allocs, a)
		}
	}
	if len(keys) == 0 {
		return fmt.Errorf("decision-cost benchmark: no result lines parsed")
	}
	recs := []string{"policy,pending,repetitions,wall_ns_per_op_min,wall_ns_per_op_median,wall_ns_per_op_max,bytes_per_op,allocs_per_op"}
	for _, k := range keys {
		c := cases[k]
		slices.Sort(c.ns)
		recs = append(recs, fmt.Sprintf("%s,%d,%.0f,%.0f,%.0f,%.0f,%.0f", k, len(c.ns), c.ns[0], metrics.Percentile(c.ns, 0.5), c.ns[len(c.ns)-1],
			metrics.Mean(c.bytes), metrics.Mean(c.allocs)))
		log.Info("decision cost", "case", k, "min_ns_per_op", c.ns[0], "repetitions", len(c.ns))
	}
	return os.WriteFile(filepath.Join(out, "wall_decision_cost.csv"), []byte(strings.Join(recs, "\n")+"\n"), 0o644)
}
