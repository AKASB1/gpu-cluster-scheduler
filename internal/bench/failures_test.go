package bench

import (
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
)

// The failure schedule depends only on (cluster, seed): common to all
// policies. Over a long horizon the mean time between failures of a node is
// MTBF + mean repair (within 10% on 16 nodes x ~250 failures, seed 3).
func TestFailureScheduleDeterministicAndCalibrated(t *testing.T) {
	c, err := cluster.Load("../../configs/clusters/homogeneous.json")
	if err != nil {
		t.Fatal(err)
	}
	f := &FailureConfig{MTBFS: 3600, RepairS: 600}
	h := 1000 * time.Hour
	a, b := f.FailureSchedule(c, 3, h), f.FailureSchedule(c, 3, h)
	if !reflect.DeepEqual(a, b) || reflect.DeepEqual(a, f.FailureSchedule(c, 4, h)) {
		t.Fatal("schedule must depend on the seed only")
	}
	perNode := float64(len(a)) / float64(len(c.Nodes))
	want := h.Seconds() / (3600 + 600)
	if math.Abs(perNode/want-1) > 0.10 {
		t.Fatalf("%.1f failures per node, want about %.1f", perNode, want)
	}
	for _, x := range a {
		if x.At < 0 || x.At >= h || x.Repair <= 0 {
			t.Fatalf("bad failure %+v", x)
		}
	}
}
