package exporter

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

func TestMetricsEndpointAndPacing(t *testing.T) {
	var slept []time.Duration
	e := New(16, 100, func(d time.Duration) { slept = append(slept, d) })
	e.Observe(metrics.Sample{T: 100 * time.Second, Alloc: 12, Free: 4, Stranded: 1.5, Blocked: true, Pending: 3, Running: 2})
	e.Observe(metrics.Sample{T: 300 * time.Second, Alloc: 8, Free: 8, Pending: 1, Running: 1})
	// virtual 100 s then 200 s at speed-up 100: 1 s and 2 s of wall time
	if len(slept) != 2 || slept[0] != time.Second || slept[1] != 2*time.Second {
		t.Fatalf("pacing %v", slept)
	}
	srv := httptest.NewServer(e)
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	text := string(body)
	for _, want := range []string{"# TYPE gcs_pending_jobs gauge", "gcs_pending_jobs 1\n", "gcs_running_jobs 1\n", "gcs_free_gpus 8\n",
		"gcs_utilization_ratio 0.5\n", "gcs_fragmentation_blocked 0\n", "gcs_sim_time_seconds 300\n", "gcs_samples_total 2\n"} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in:\n%s", want, text)
		}
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/plain; version=0.0.4") {
		t.Fatalf("content type %q", resp.Header.Get("Content-Type"))
	}
	if r, _ := http.Get(srv.URL + "/other"); r.StatusCode != http.StatusNotFound {
		t.Fatalf("status %d", r.StatusCode)
	}
	// no pacing at speed-up 0
	e2 := New(16, 0, func(time.Duration) { t.Fatal("must not sleep") })
	e2.Observe(metrics.Sample{T: time.Hour})
}
