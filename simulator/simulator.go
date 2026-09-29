package simulator

import (
	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
)

func Run(jobs []api.Job, nodes []cluster.Node, p policy.PlacementPolicy) map[string]string {
	assignments := make(map[string]string)
	for _, job := range jobs {
		name, ok := p.Place(job, nodes)
		if !ok { continue }
		for i := range nodes {
			if nodes[i].Name == name { nodes[i].FreeGPUs -= job.GPUs; break }
		}
		assignments[job.ID] = name
	}
	return assignments
}
