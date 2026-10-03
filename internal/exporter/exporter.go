// Package exporter publishes the state of a replayed simulation as
// Prometheus text-format gauges on /metrics (Tier 2). It is written against
// the standard library only (the text exposition format is simple), paces
// the simulation against the wall clock through an injected sleep function,
// and knows nothing about the simulator itself.
package exporter

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

// Exporter holds the latest sample and serves it.
type Exporter struct {
	mu        sync.Mutex
	last      metrics.Sample
	samples   int
	totalGPUs int
	speedup   float64
	sleep     func(time.Duration)
	paced     time.Duration // virtual time already paced
}

// New returns an exporter for a cluster with totalGPUs GPUs. speedup is the
// ratio of virtual to wall time (0: no pacing); sleep is time.Sleep in the
// command and a recorder in tests.
func New(totalGPUs int, speedup float64, sleep func(time.Duration)) *Exporter {
	return &Exporter{totalGPUs: totalGPUs, speedup: speedup, sleep: sleep}
}

// Observe records a sample; with pacing it first sleeps for the virtual time
// elapsed since the previous sample divided by the speed-up.
func (e *Exporter) Observe(s metrics.Sample) {
	if e.speedup > 0 && s.T > e.paced {
		e.sleep(time.Duration(float64(s.T-e.paced) / e.speedup))
		e.paced = s.T
	}
	e.mu.Lock()
	e.last = s
	e.samples++
	e.mu.Unlock()
}

// ServeHTTP writes the gauges in the Prometheus text exposition format.
func (e *Exporter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/metrics" {
		http.NotFound(w, r)
		return
	}
	e.mu.Lock()
	s, n, total := e.last, e.samples, e.totalGPUs
	e.mu.Unlock()
	util := 0.0
	if total > 0 {
		util = float64(s.Alloc) / float64(total)
	}
	blocked := 0
	if s.Blocked {
		blocked = 1
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	gauges := []struct {
		name, help string
		value      float64
	}{
		{"gcs_sim_time_seconds", "Virtual time of the replayed simulation (simulated).", s.T.Seconds()},
		{"gcs_pending_jobs", "Jobs waiting in the queue (simulated).", float64(s.Pending)},
		{"gcs_running_jobs", "Jobs running (simulated).", float64(s.Running)},
		{"gcs_free_gpus", "Free GPUs (simulated).", float64(s.Free)},
		{"gcs_allocated_gpus", "Allocated GPUs, including restart overhead (simulated).", float64(s.Alloc)},
		{"gcs_utilization_ratio", "Allocated GPUs / total GPUs at this instant (simulated).", util},
		{"gcs_stranded_gpus", "Expected stranded free GPUs (simulated).", s.Stranded},
		{"gcs_fragmentation_blocked", "1 if a pending job fits the free totals but cannot be placed (simulated).", float64(blocked)},
		{"gcs_samples_total", "Scheduling instants observed.", float64(n)},
	}
	for _, g := range gauges {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", g.name, g.help, g.name, g.name, g.value)
	}
}
