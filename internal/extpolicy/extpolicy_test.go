package extpolicy_test

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/trace"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

// needPython skips (with a clear message) when the harness cannot be
// imported, unless GCS_REQUIRE_PYTHON=1 (CI), where it fails.
func needPython(t *testing.T) string {
	t.Helper()
	root, err := extpolicy.RepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(extpolicy.Python(), "-c", "import harness.policy_server")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		msg := fmt.Sprintf("Python harness not available (%s: %v %s); set PYTHON or activate the virtual environment", extpolicy.Python(), err, out)
		if os.Getenv("GCS_REQUIRE_PYTHON") == "1" {
			t.Fatal(msg)
		}
		t.Skip(msg)
	}
	return root
}

const clusterJSON = `{"schema_version":1,"name":"conf","classes":[{"name":"a100","speed":1},{"name":"v100","speed":0.4}],
"node_groups":[{"count":2,"prefix":"r0-n","rack":"r0","class":"a100","gpus":8,"cpus":128,"mem_gb":1024},
{"count":2,"prefix":"r1-n","rack":"r1","class":"a100","gpus":8,"cpus":128,"mem_gb":1024},
{"count":2,"prefix":"r2-n","rack":"r2","class":"v100","gpus":8,"cpus":128,"mem_gb":1024}],
"cross_node_factor":1.1,"cross_rack_factor":1.25,"restart_overhead_s":60,"preempt_grace_s":0}`

func sharedTraces(t *testing.T, cl *cluster.Cluster) [][]api.Job {
	t.Helper()
	data, err := os.ReadFile("../../configs/workloads/base.json")
	if err != nil {
		t.Fatal(err)
	}
	var out [][]api.Job
	for i, est := range []trace.EstimateConfig{{Model: "exact"}, {Model: "lognormal", Sigma: 0.5}, {Model: "lognormal", Sigma: 1}} {
		var c trace.GenConfig
		if err := json.Unmarshal(data, &c); err != nil {
			t.Fatal(err)
		}
		c.Jobs, c.WorkCapacity, c.Estimate = 250, cl.WorkCapacity(), est
		c.Arrival.Load = 0.9
		c.ClassConstraints = []trace.ClassFraction{{Class: "v100", Frac: 0.15}}
		jobs, err := trace.Generate(c, uint64(100+i))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, jobs)
	}
	return out
}

func runWith(t *testing.T, cl *cluster.Cluster, jobs []api.Job, p policy.Policy) (*metrics.RunTrace, error) {
	t.Helper()
	rt, err := simulator.Run(simulator.Config{Cluster: cl, Jobs: jobs, Policy: p, Log: true})
	if c, ok := p.(policy.Closer); ok {
		if cerr := c.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}
	return rt, err
}

func client(root, name string, timeout time.Duration) *extpolicy.Client {
	return extpolicy.New(name, nil, extpolicy.Config{Dir: root, Timeout: timeout, Seed: 1})
}

// Check 8: the Go and the Python implementations of fifo+first_fit+none and
// fifo+first_fit+easy produce identical assignment logs (time, job, workers
// per node, preemptions) on three shared traces, through the protocol.
func TestConformanceGoPython(t *testing.T) {
	root := needPython(t)
	cl, err := cluster.Parse([]byte(clusterJSON))
	if err != nil {
		t.Fatal(err)
	}
	backfilled := 0
	for k, jobs := range sharedTraces(t, cl) {
		var logs [][]metrics.LogEntry
		for _, name := range []string{"fifo+first_fit+none", "fifo+first_fit+easy"} {
			gp, err := policy.New(policy.Spec{Name: name}, policy.Env{})
			if err != nil {
				t.Fatal(err)
			}
			goRT, err := runWith(t, cl, jobs, gp)
			if err != nil {
				t.Fatal(err)
			}
			pc := client(root, name, extpolicy.DefaultTimeout)
			pyRT, err := runWith(t, cl, jobs, pc)
			if err != nil {
				t.Fatal(err)
			}
			if !pc.Exited() {
				t.Fatal("python child left running")
			}
			if len(goRT.Log) == 0 || !reflect.DeepEqual(goRT.Log, pyRT.Log) {
				for i := range goRT.Log {
					if i >= len(pyRT.Log) || !reflect.DeepEqual(goRT.Log[i], pyRT.Log[i]) {
						t.Fatalf("trace %d %s: logs differ at entry %d: go %+v", k, name, i, goRT.Log[i])
					}
				}
				t.Fatalf("trace %d %s: logs differ in length %d vs %d", k, name, len(goRT.Log), len(pyRT.Log))
			}
			// reservations reported through the protocol match too
			for i := range goRT.Jobs {
				g, p := goRT.Jobs[i], pyRT.Jobs[i]
				if g.HasShadow != p.HasShadow || g.Shadow != p.Shadow {
					t.Fatalf("trace %d %s: job %s reservation differs", k, name, g.Job.ID)
				}
			}
			t.Logf("trace %d %s: %d identical log entries", k, name, len(goRT.Log))
			logs = append(logs, goRT.Log)
		}
		if !reflect.DeepEqual(logs[0], logs[1]) {
			backfilled++
		}
	}
	// the check is not vacuous: EASY changes the schedule on the shared traces
	if backfilled == 0 {
		t.Fatal("EASY never differed from no backfilling on the shared traces")
	}
}

// A misbehaving policy aborts the run with a clear error naming the policy,
// and leaves no child process behind.
func TestMisbehavingPolicyAbortsTheRun(t *testing.T) {
	root := needPython(t)
	cl, err := cluster.Parse([]byte(clusterJSON))
	if err != nil {
		t.Fatal(err)
	}
	jobs := sharedTraces(t, cl)[0][:20]
	cases := []struct {
		name, want string
	}{
		{"test:sleep", "no answer within"},
		{"test:badjson", "malformed reply line"},
		{"test:invalid", "invalid action #0 by policy \"py:test:invalid\""},
		{"test:crash", "exited"},
		{"no-such-policy", "unknown policy"},
	}
	for _, c := range cases {
		pc := client(root, c.name, 10*time.Second) // generous: interpreter start-up on a loaded machine
		_, err := runWith(t, cl, jobs, pc)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: want error containing %q, got %v", c.name, c.want, err)
		}
		if !strings.Contains(err.Error(), c.name) {
			t.Fatalf("%s: error does not name the policy: %v", c.name, err)
		}
		// poll with a timeout: the child must be gone
		deadline := time.Now().Add(10 * time.Second)
		for !pc.Exited() && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if !pc.Exited() {
			t.Fatalf("%s: child process %d still running", c.name, pc.PID())
		}
	}
}

func TestCloseIsIdempotentAndNoStartNoProcess(t *testing.T) {
	c := extpolicy.New("fifo+first_fit+none", nil, extpolicy.Config{})
	if err := c.Close(); err != nil || !c.Exited() {
		t.Fatal("close before start")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
}

// milp_rolling end to end through the protocol: the run drains, the
// invariants and identities hold, and the solver statistics arrive.
func TestMilpRollingEndToEnd(t *testing.T) {
	for _, k := range []struct{ name, params string }{
		{"milp_rolling", `{"k": 8, "h": 16, "slot_s": 600}`},
		{"scenario_milp", `{"k": 6, "h": 10, "slot_s": 900, "scenarios": 4, "lambda": 0.5}`},
	} {
		t.Run(k.name, func(t *testing.T) { solverEndToEnd(t, k.name, k.params) })
	}
}

func solverEndToEnd(t *testing.T, name, params string) {
	root := needPython(t)
	cl, err := cluster.Parse([]byte(clusterJSON))
	if err != nil {
		t.Fatal(err)
	}
	jobs := sharedTraces(t, cl)[1][:80]
	pc := extpolicy.New(name, json.RawMessage(params), extpolicy.Config{Dir: root, Seed: 1})
	rt, err := simulator.Run(simulator.Config{Cluster: cl, Jobs: jobs, Policy: pc})
	if cerr := pc.Close(); cerr != nil && err == nil {
		err = cerr
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := metrics.Verify(rt, cl.EmptyNodes()); err != nil {
		t.Fatal(err)
	}
	if err := metrics.CheckIdentities(rt); err != nil {
		t.Fatal(err)
	}
	if rt.Solver == nil || rt.Solver.Calls == 0 || !pc.Exited() {
		t.Fatalf("solver stats %+v exited %v", rt.Solver, pc.Exited())
	}
	t.Logf("%s: %d invocations, %d solver calls, %d capped, %d not optimal, cpu %.3fs total", name, rt.Invocations, rt.Solver.Calls,
		rt.Solver.Capped, rt.Solver.NotOptimal, sum(rt.Solver.CPUSolve))
	if sum(rt.Solver.CPUSolve) <= 0 {
		t.Fatal("no CPU time reported for the solves")
	}
}

func sum(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s
}
