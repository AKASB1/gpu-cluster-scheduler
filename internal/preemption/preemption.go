package preemption

import "github.com/AKASB1/gpu-cluster-scheduler/internal/api"

func MayPreempt(waiting, running api.Job) bool {
	return waiting.Priority > running.Priority
}
