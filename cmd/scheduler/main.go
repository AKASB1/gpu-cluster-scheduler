package main

import (
	"fmt"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/cluster"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/policy"
	"github.com/AKASB1/gpu-cluster-scheduler/simulator"
)

func main() {
	jobs := []api.Job{{ID: "demo", GPUs: 1}}
	nodes := []cluster.Node{{Name: "local", FreeGPUs: 2}}
	fmt.Println(simulator.Run(jobs, nodes, policy.FIFO{}))
}
