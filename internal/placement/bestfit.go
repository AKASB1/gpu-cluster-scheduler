package placement

import (
	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
)

func BestFit(job api.Job, nodes []cluster.Node) (string, bool) {
	if !job.Valid() { return "", false }
	best := ""
	waste := int(^uint(0) >> 1)
	for _, node := range nodes {
		remaining := node.FreeGPUs - job.GPUs
		if remaining >= 0 && remaining < waste {
			best, waste = node.Name, remaining
		}
	}
	return best, best != ""
}
