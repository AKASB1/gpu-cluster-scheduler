package policy

import (
	"fmt"
	"slices"
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// BenchmarkSchedule measures one scheduling invocation (decision cost) with
// 64, 512, and 4096 pending jobs:
//
//	go test -run '^$' -bench BenchmarkSchedule -benchmem ./internal/policy
func BenchmarkSchedule(b *testing.B) {
	for _, name := range DecisionCostPolicies {
		for _, n := range DecisionCostPending {
			v := DecisionCostView(n)
			b.Run(fmt.Sprintf("%s/pending=%d", name, n), func(b *testing.B) {
				p, err := New(Spec{Name: name}, Env{})
				if err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := p.Schedule(v); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestDecisionCostViewIsConsistent(t *testing.T) {
	v := DecisionCostView(64)
	used := 0
	for i := range v.Nodes {
		if v.Nodes[i].FreeGPUs < 0 {
			t.Fatal("negative free GPUs")
		}
		used += v.Nodes[i].GPUs - v.Nodes[i].FreeGPUs
	}
	if used < v.Cluster.TotalGPUs/3 || len(v.Pending) != 64 {
		t.Fatalf("used %d pending %d", used, len(v.Pending))
	}
	if !slices.IsSortedFunc(v.Pending, func(a, b api.PendingJob) int { return api.PendingLess(&a, &b) }) {
		t.Fatal("pending jobs must be in the view's canonical order")
	}
	for _, name := range DecisionCostPolicies {
		p, _ := New(Spec{Name: name}, Env{})
		d, err := p.Schedule(v)
		if err != nil || len(d.Actions) == 0 {
			t.Fatalf("%s: %v %d actions", name, err, len(d.Actions))
		}
	}
}
