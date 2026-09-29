package simulator

import (
	"testing"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
)

func TestCapacity(t *testing.T) {
	jobs := []api.Job{{ID: "a", GPUs: 1}, {ID: "b", GPUs: 1}}
	nodes := []cluster.Node{{Name: "n", FreeGPUs: 1}}
	result := Run(jobs, nodes, policy.FIFO{})
	if result["a"] != "n" || len(result) != 1 { t.Fatalf("unexpected assignments: %v", result) }
}
