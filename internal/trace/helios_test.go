package trace

import (
	"strings"
	"testing"
	"time"
)

// A synthetic fixture in the format of the Helios cluster_log.csv files (no
// rows of the real trace are committed).
const heliosFixture = HeliosHeader + `
10,uA,vcX,8,16,1,COMPLETED,2020-08-10 00:00:05,2020-08-10 00:00:05,2020-08-10 01:00:05,3600,0
11,uB,vcY,0,4,1,COMPLETED,2020-08-10 00:01:00,2020-08-10 00:01:00,2020-08-10 00:11:00,600,0
12,uB,vcY,16,64,2,FAILED,2020-08-10 00:02:00,2020-08-10 00:02:00,2020-08-10 00:12:00,600,0
13,uC,vcX,4,8,1,CANCELLED,2020-08-10 00:03:00,2020-08-10 00:03:00,2020-08-10 00:03:00,0,0
14,uC,vcX,12,24,1,TIMEOUT,2020-08-10 00:04:00,2020-08-10 00:04:00,2020-08-10 10:04:00,36000,0
15,uD,vcZ,3,6,2,COMPLETED,2020-08-10 00:05:00,2020-08-10 00:05:00,2020-08-10 00:15:00,600,0
16,uD,vcZ,2,4,1,COMPLETED,2020-08-09 23:59:59,2020-08-09 23:59:59,2020-08-10 00:09:59,600,0
17,uE,vcZ,1,2,1,COMPLETED,2020-08-13 00:00:00,2020-08-13 00:00:00,2020-08-13 00:10:00,600,0
18,uA,vcX,1,2,1,COMPLETED,2020-08-10 00:00:05,2020-08-10 00:00:05,2020-08-10 00:10:05,600,0
`

func TestConvertHeliosFixture(t *testing.T) {
	start, _ := time.Parse(heliosTime, "2020-08-10 00:00:00")
	o := HeliosOptions{Start: start, Length: 3 * 24 * time.Hour, GPUsPerNode: 8, Estimate: EstimateConfig{Model: "exact"}, Seed: 1}
	jobs, st, err := ConvertHelios(strings.NewReader(heliosFixture), o)
	if err != nil {
		t.Fatal(err)
	}
	// rows 16 (before) and 17 (at the end, excluded) are outside the window;
	// 11 has no GPU; 13 has zero duration
	if st.Rows != 9 || st.InWindow != 7 || st.NonGPU != 1 || st.ZeroDuration != 1 || st.Kept != 5 {
		t.Fatalf("stats %+v", st)
	}
	type want struct {
		id               string
		submit           time.Duration
		gpus, workers, c int
	}
	wants := []want{
		{"h10", 5 * time.Second, 8, 1, 16},   // 8 GPUs on 1 node
		{"h18", 5 * time.Second, 1, 1, 2},    // same submit time: ordered by id
		{"h12", 120 * time.Second, 8, 2, 32}, // 16 GPUs on 2 nodes -> 2 x 8
		{"h14", 240 * time.Second, 1, 12, 2}, // 12 GPUs "on 1 node" exceeds a node, not divisible by 8 -> 12 x 1, cpus ceil(24/12)
		{"h15", 300 * time.Second, 3, 1, 6},  // 3 GPUs on 2 nodes (not divisible) -> 1 x 3
	}
	if len(jobs) != len(wants) {
		t.Fatalf("got %d jobs", len(jobs))
	}
	for i, w := range wants {
		j := jobs[i]
		if j.ID != w.id || j.Submit != w.submit || j.GPUs != w.gpus || j.Workers != w.workers || j.CPUs != w.c {
			t.Errorf("job %d: got %+v, want %+v", i, j, w)
		}
		if j.Estimate != j.Runtime || j.Priority != 1 || j.Preemptible || j.HasMaxWait || j.Topology != "any" {
			t.Errorf("job %d attributes %+v", i, j)
		}
	}
	if jobs[0].Tenant != "vcX" || jobs[0].User != "uA" {
		t.Fatalf("tenant/user %+v", jobs[0])
	}
	// the result is a valid schema-v1 trace
	if _, err := Parse(Encode(jobs)); err != nil {
		t.Fatal(err)
	}
	// synthetic estimates depend on the seed only
	o.Estimate = EstimateConfig{Model: "lognormal", Sigma: 0.5}
	a, _, _ := ConvertHelios(strings.NewReader(heliosFixture), o)
	b, _, _ := ConvertHelios(strings.NewReader(heliosFixture), o)
	o.Seed = 2
	c, _, _ := ConvertHelios(strings.NewReader(heliosFixture), o)
	if string(Encode(a)) != string(Encode(b)) || string(Encode(a)) == string(Encode(c)) {
		t.Fatal("estimates must depend on the seed only")
	}
}

func TestConvertHeliosRejectsOtherFormats(t *testing.T) {
	if _, _, err := ConvertHelios(strings.NewReader(Header+"\n"), HeliosOptions{GPUsPerNode: 8}); err == nil {
		t.Fatal("a schema-v1 trace is not a Helios log")
	}
	bad := HeliosHeader + "\n10,u,v,x,1,1,COMPLETED,2020-08-10 00:00:05,a,b,1,0\n"
	start, _ := time.Parse(heliosTime, "2020-08-10 00:00:00")
	if _, _, err := ConvertHelios(strings.NewReader(bad), HeliosOptions{Start: start, Length: time.Hour, GPUsPerNode: 8}); err == nil {
		t.Fatal("unparsable gpu_num accepted")
	}
}
