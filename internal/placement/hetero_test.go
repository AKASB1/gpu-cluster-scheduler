package placement

import (
	"fmt"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// hetero_ect by hand: node 0 is fast (speed 1, 8 GPUs, fully busy until
// 1000 s), node 1 is slow (speed 0.4, 8 GPUs free).
func heteroView(fastEnd float64) *api.View {
	return &api.View{
		Nodes: []api.NodeState{
			{Index: 0, Name: "f", Rack: "r", Class: "fast", Speed: 1, GPUs: 8, CPUs: 64, MemMB: 1 << 20, FreeGPUs: 0, FreeCPUs: 64, FreeMemMB: 1 << 20},
			{Index: 1, Name: "s", Rack: "r", Class: "slow", Speed: 0.4, GPUs: 8, CPUs: 64, MemMB: 1 << 20, FreeGPUs: 8, FreeCPUs: 64, FreeMemMB: 1 << 20},
		},
		Running: []api.RunningJob{{JobID: "R", GPUs: 8, Workers: 1, Placement: api.Placement{{Node: 0, Workers: 1}},
			EstEnd: time.Duration(fastEnd * float64(time.Second))}},
	}
}

func TestHeteroECTByHand(t *testing.T) {
	cases := []struct {
		fastEnd, work float64
		class         string
		want          string
	}{
		// work 1000 s: slow now = 2500 s; fast after 1000 s wait = 2000 s -> wait (no placement now)
		{1000, 1000, "", "[] false"},
		// work 100 s: slow now = 250 s; fast = 1000 + 100 = 1100 s -> slow node now
		{1000, 100, "", "[{1 1}] true"},
		// fast frees soon (10 s): fast = 10 + 1000 = 1010 < slow 2500 -> wait
		{10, 1000, "", "[] false"},
		// a class-constrained job is placed on its class regardless of ECT
		{1000, 1000, "slow", "[{1 1}] true"},
	}
	for i, k := range cases {
		v := heteroView(k.fastEnd)
		h := &HeteroECT{}
		h.Prepare(v)
		r := Request{GPUs: 8, Workers: 1, Class: k.class, Work: k.work}
		pl, ok := h.Place(r, v.Nodes, FreeOf(v.Nodes))
		if got := fmt.Sprint(pl, ok); got != k.want {
			t.Errorf("case %d: got %s, want %s", i, got, k.want)
		}
	}
	// when the fast node is free, it is chosen (ECT 1000 < 2500)
	v := heteroView(1000)
	v.Nodes[0].FreeGPUs, v.Running = 8, nil
	h := &HeteroECT{}
	h.Prepare(v)
	if pl, ok := h.Place(Request{GPUs: 4, Workers: 1, Work: 1000}, v.Nodes, FreeOf(v.Nodes)); !ok || pl[0].Node != 0 {
		t.Fatalf("free fast node not chosen: %v %v", pl, ok)
	}
}
