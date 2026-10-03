package metrics

import (
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// Hand-made run on 8 GPUs, quotas 4 + 4: A1 (tenant A, 8 GPUs) runs [0, 10)
// and borrows 4 GPUs; B1 (tenant B, 4 GPUs) waits [0, 10) and runs [10, 20).
// Entitled demand: A 4 x 10 + B 4 x 20 = 120 GPU-s; served: A 4 x 10 + B 4 x
// 10 = 80 -> satisfaction 2/3. Borrowed: 4 GPUs x 10 s.
func TestQuotaFieldsByHand(t *testing.T) {
	s := func(x float64) time.Duration { return time.Duration(x * float64(time.Second)) }
	rt := &RunTrace{Info: api.ClusterInfo{TotalGPUs: 8}, FastestAny: 1, End: s(20)}
	rt.Jobs = []JobRecord{
		{Job: api.Job{ID: "A1", Tenant: "A", GPUs: 8, Workers: 1, Runtime: s(10)}, FirstStart: 0, Completion: s(10),
			Segments: []Segment{{Start: 0, End: s(10)}}},
		{Job: api.Job{ID: "B1", Tenant: "B", GPUs: 4, Workers: 1, Runtime: s(10)}, FirstStart: s(10), Completion: s(20),
			Segments: []Segment{{Start: s(10), End: s(20)}}},
	}
	get := func(fs []Field, name string) float64 {
		for _, f := range fs {
			if f.Name == name {
				return f.Value
			}
		}
		return -1
	}
	fs := QuotaFields(rt, map[string]float64{"A": 0.5, "B": 0.5})
	if !near(get(fs, "quota_satisfaction"), 80.0/120) || !near(get(fs, "borrowed_gpu_hours"), 40.0/3600) {
		t.Fatalf("fields %+v", fs)
	}
	// bounded slowdowns: A 1, B 2; equal weights give the plain Jain index 0.9
	if !near(get(fs, "jain_weighted_bsld"), 0.9) {
		t.Fatalf("jain %v", get(fs, "jain_weighted_bsld"))
	}
	// weights 0.75 / 0.25: (0.75 + 0.5)^2 / (1 * (0.75 + 1)) = 1.5625 / 1.75
	fs = QuotaFields(rt, map[string]float64{"A": 0.75, "B": 0.25})
	if !near(get(fs, "jain_weighted_bsld"), 1.5625/1.75) {
		t.Fatalf("weighted jain %v", get(fs, "jain_weighted_bsld"))
	}
}
