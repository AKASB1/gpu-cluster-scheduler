package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
)

// MILPTask is one offline solve (harness/milp/solve.py).
type MILPTask struct {
	ID         string  `json:"id"`
	Trace      string  `json:"trace"`
	Cluster    string  `json:"cluster"`
	Model      string  `json:"model"` // node or pool
	SlotS      float64 `json:"slot_s"`
	MipRelGap  float64 `json:"mip_rel_gap"`
	NodeLimit  *int    `json:"node_limit,omitempty"`
	TimeLimitS float64 `json:"time_limit_s"`
}

// MILPResult is the solver's answer for one task.
type MILPResult struct {
	ID         string                `json:"id"`
	Status     string                `json:"status"`
	Objective  *float64              `json:"objective"`
	Bound      *float64              `json:"bound"`
	Gap        *float64              `json:"gap"`
	Nodes      int                   `json:"nodes"`
	Capped     bool                  `json:"capped"`
	WallSolveS float64               `json:"wall_solve_s"`
	CPUSolveS  *float64              `json:"cpu_solve_s"`
	Variables  int                   `json:"variables"`
	Horizon    int                   `json:"horizon"`
	WholeSlot  bool                  `json:"whole_slot"`
	Model      string                `json:"model"`
	SlotS      float64               `json:"slot_s"`
	Jobs       int                   `json:"jobs"`
	Starts     []policy.PlannedStart `json:"starts"`
}

// SolveOffline runs the Python offline solver on tasks in one process.
// workDir receives the task and result files; stderr goes to logPath.
func SolveOffline(ctx context.Context, tasks []MILPTask, workDir, logPath string) ([]MILPResult, error) {
	root, err := extpolicy.RepoRoot()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return nil, err
	}
	tf := filepath.Join(workDir, "milp_tasks.json")
	of := filepath.Join(workDir, "milp_results.jsonl")
	data, err := json.MarshalIndent(tasks, "", " ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(tf, data, 0o644); err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, extpolicy.Python(), "-m", "harness.milp.solve", "--tasks", tf, "--out", of)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "PYTHONUTF8=1")
	cmd.WaitDelay = 5 * time.Second
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	err = cmd.Run()
	if logPath != "" {
		_ = os.MkdirAll(filepath.Dir(logPath), 0o755)
		_ = os.WriteFile(logPath, stderr.Bytes(), 0o644)
	}
	if err != nil {
		return nil, fmt.Errorf("offline MILP solver: %w\n%s", err, tailOf(stderr.String()))
	}
	f, err := os.Open(of)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []MILPResult
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		var r MILPResult
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("offline MILP result: %w", err)
		}
		out = append(out, r)
	}
	if len(out) != len(tasks) {
		return nil, fmt.Errorf("offline MILP solver returned %d results for %d tasks", len(out), len(tasks))
	}
	return out, sc.Err()
}

func tailOf(s string) string {
	if len(s) > 2000 {
		return "..." + s[len(s)-2000:]
	}
	return s
}
