package policy

import (
	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
)

type PlacementPolicy interface {
	Place(api.Job, []cluster.Node) (string, bool)
}

type FIFO struct{}

func (FIFO) Place(job api.Job, nodes []cluster.Node) (string, bool) {
	if !job.Valid() { return "", false }
	for _, node := range nodes {
		if node.FreeGPUs >= job.GPUs { return node.Name, true }
	}
	return "", false
}
