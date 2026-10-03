package trace

import (
	"cmp"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/rng"
)

// HeliosHeader is the header of the Helios cluster_log.csv files
// (S-Lab-System-Group/HeliosData, CC-BY-4.0).
const HeliosHeader = "job_id,user,vc,gpu_num,cpu_num,node_num,state,submit_time,start_time,end_time,duration,queue"

// HeliosOptions select and convert an excerpt of a Helios cluster log.
type HeliosOptions struct {
	Start       time.Time     // first submit time of the excerpt (UTC wall time of the log)
	Length      time.Duration // excerpt length
	GPUsPerNode int           // node size of the simulated cluster (assumed 8)
	Estimate    EstimateConfig
	Seed        uint64 // seed of the synthetic estimates
}

// ConvertStats reports what the converter kept and dropped.
type ConvertStats struct {
	Rows, InWindow, NonGPU, ZeroDuration, Kept int
}

const heliosTime = "2006-01-02 15:04:05"

// ConvertHelios converts the GPU jobs of a Helios cluster log submitted in
// [Start, Start+Length) into schema-v1 jobs:
//
//   - tenant = vc, user = user, priority 1, not preemptible, no checkpoint,
//     no wait target, topology any (the log has none of these);
//   - workers: when gpu_num is divisible by node_num and the per-node share
//     fits a node (at most GPUsPerNode), node_num workers of gpu_num/node_num
//     GPUs; otherwise, when gpu_num exceeds a node, gpu_num/GPUsPerNode
//     workers of GPUsPerNode GPUs if divisible, else gpu_num workers of one
//     GPU; otherwise one worker of gpu_num GPUs (an assumption: the log does
//     not say how the GPUs were split);
//   - cpus per worker = ceil(cpu_num / workers), memory 0 (not in the log);
//   - runtime_s = duration (all final states are kept: the GPUs were held);
//   - submit_s = submit_time - Start;
//   - estimate_s = synthetic: the estimate model applied with Seed (the log
//     has no user estimates).
//
// Jobs without GPUs and jobs with zero duration are dropped and counted.
func ConvertHelios(r io.Reader, o HeliosOptions) ([]api.Job, ConvertStats, error) {
	var st ConvertStats
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = 12
	head, err := cr.Read()
	if err != nil {
		return nil, st, fmt.Errorf("helios: %w", err)
	}
	if strings.Join(head, ",") != HeliosHeader {
		return nil, st, errors.New("helios: unexpected header (want " + HeliosHeader + ")")
	}
	end := o.Start.Add(o.Length)
	type row struct {
		j      api.Job
		submit time.Time
	}
	var rows []row
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, st, fmt.Errorf("helios: %w", err)
		}
		st.Rows++
		submit, err := time.Parse(heliosTime, rec[7])
		if err != nil {
			return nil, st, fmt.Errorf("helios: job %s: submit_time: %w", rec[0], err)
		}
		if submit.Before(o.Start) || !submit.Before(end) {
			continue
		}
		st.InWindow++
		gpus, err1 := strconv.Atoi(rec[3])
		cpus, err2 := strconv.Atoi(rec[4])
		nodes, err3 := strconv.Atoi(rec[5])
		dur, err4 := strconv.Atoi(rec[10])
		if err := errors.Join(err1, err2, err3, err4); err != nil {
			return nil, st, fmt.Errorf("helios: job %s: %w", rec[0], err)
		}
		if gpus <= 0 {
			st.NonGPU++
			continue
		}
		if dur <= 0 {
			st.ZeroDuration++
			continue
		}
		workers, per := 1, gpus
		switch {
		case nodes >= 1 && gpus%nodes == 0 && gpus/nodes <= o.GPUsPerNode:
			workers, per = nodes, gpus/nodes
		case gpus > o.GPUsPerNode && gpus%o.GPUsPerNode == 0:
			workers, per = gpus/o.GPUsPerNode, o.GPUsPerNode
		case gpus > o.GPUsPerNode:
			workers, per = gpus, 1
		}
		j := api.Job{ID: "h" + rec[0], Tenant: rec[2], User: rec[1], Priority: 1, GPUs: per, Workers: workers,
			Topology: api.TopoAny, CPUs: (max(cpus, 0) + workers - 1) / workers, Runtime: time.Duration(dur) * time.Second}
		if j.User == "" {
			j.User = j.Tenant
		}
		rows = append(rows, row{j, submit})
	}
	slices.SortStableFunc(rows, func(a, b row) int {
		if c := a.submit.Compare(b.submit); c != 0 {
			return c
		}
		return cmp.Compare(a.j.ID, b.j.ID)
	})
	est := rng.New(o.Seed, "estimates")
	jobs := make([]api.Job, len(rows))
	for i, rw := range rows {
		j := rw.j
		j.Submit = rw.submit.Sub(o.Start).Truncate(time.Millisecond)
		j.Estimate = estimate(o.Estimate, est, j.Runtime)
		jobs[i] = j
	}
	st.Kept = len(jobs)
	return jobs, st, nil
}
