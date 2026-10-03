package bench

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
)

// The derived cluster of a replay has one partition (GPU class) per virtual
// cluster with that VC's GPUs on the last date not after the window start;
// VCs without GPUs are left out.
func TestDerivedClusterPerVirtualCluster(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cluster_gpu_number.csv")
	data := "date,vcA,vcB,vcC,total\n2020-08-01,64,8,0,72\n2020-08-02,72,12,0,84\n2020-08-03,16,16,16,48\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	vcs, err := capacitiesOn(path, time.Date(2020, 8, 2, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(vcs) != 2 || vcs["vcA"] != 72 || vcs["vcB"] != 12 {
		t.Fatalf("capacities %v", vcs)
	}
	jobs := []api.Job{{ID: "j", GPUs: 8, Workers: 1, CPUs: 96}}
	cl, err := cluster.Build(DerivedCluster("d", vcs, jobs))
	if err != nil {
		t.Fatal(err)
	}
	per := map[string]int{}
	racks := map[string]bool{}
	for _, n := range cl.EmptyNodes() {
		per[n.Class] += n.GPUs
		racks[n.Rack] = true
		if n.CPUs != 96 {
			t.Fatalf("node %s has %d CPUs, want 96 (largest per-node request)", n.Name, n.CPUs)
		}
	}
	// vcA: 9 nodes of 8 (racks of up to 8 nodes: two racks); vcB: 8 + a 4-GPU node
	if per["vcA"] != 72 || per["vcB"] != 12 || cl.Info.TotalGPUs != 84 || len(racks) != 3 {
		t.Fatalf("GPUs per VC %v, total %d, racks %v", per, cl.Info.TotalGPUs, racks)
	}
}
