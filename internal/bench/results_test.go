package bench

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

func row(p string, seed uint64, wait, util float64) Row {
	return Row{Scenario: "s", Policy: p, Seed: seed,
		Fields: []metrics.Field{{Name: "wait_mean", Value: wait}, {Name: "utilization", Value: util}},
		Wall:   []metrics.Field{{Name: "wall_decide_mean_s", Value: 0.001 * float64(seed)}}}
}

func TestRowsRoundTripAndWallApart(t *testing.T) {
	dir := t.TempDir()
	rows := []Row{row("b", 2, 4, 0.5), row("a", 1, 3, math.NaN()), row("a", 2, 5, 0.75)}
	if err := WriteRows(dir, rows); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "runs.csv"))
	want := "scenario,policy,seed,wait_mean,utilization\ns,a,1,3,\ns,a,2,5,0.75\ns,b,2,4,0.5\n"
	if string(data) != want {
		t.Fatalf("runs.csv:\n%s", data)
	}
	if strings.Contains(string(data), "wall_") {
		t.Fatal("wall columns must be apart")
	}
	wall, _ := os.ReadFile(filepath.Join(dir, "wall.csv"))
	if !strings.HasPrefix(string(wall), "scenario,policy,seed,wall_decide_mean_s\n") {
		t.Fatalf("wall.csv:\n%s", wall)
	}
	back, err := ReadRows(filepath.Join(dir, "runs.csv"))
	if err != nil || len(back) != 3 || back[2].Get("wait_mean") != 4 || !math.IsNaN(back[0].Get("utilization")) {
		t.Fatalf("read back %+v %v", back, err)
	}
}

func TestSummarizePairedAgainstBaseline(t *testing.T) {
	rows := []Row{row("base", 1, 10, 0.5), row("base", 2, 20, 0.5), row("base", 3, 30, 0.5),
		row("x", 1, 8, 0.6), row("x", 2, 20, 0.4), row("x", 3, 33, 0.7)}
	aggs, pairs := Summarize(rows, "base", nil)
	if len(aggs) != 4 || len(pairs) != 2 {
		t.Fatalf("aggs %d pairs %d", len(aggs), len(pairs))
	}
	for _, p := range pairs {
		switch p.Metric {
		case "wait_mean": // diffs -2, 0, +3: lower is better -> 1 win, 1 tie, 1 loss
			if p.Win != 1 || p.Tie != 1 || p.Loss != 1 || math.Abs(p.Diff.Mean-1.0/3) > 1e-12 {
				t.Fatalf("wait %+v", p)
			}
		case "utilization": // higher is better: +0.1, -0.1, +0.2 -> 2 wins, 1 loss
			if p.Win != 2 || p.Loss != 1 {
				t.Fatalf("util %+v", p)
			}
		}
	}
}
