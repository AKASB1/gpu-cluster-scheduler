package queue

import "github.com/AKASB1/gpu-cluster-scheduler/internal/api"

type Queue struct { jobs []api.Job }

func (q *Queue) Push(job api.Job) {
	q.jobs = append(q.jobs, job)
}

func (q *Queue) Pop() (api.Job, bool) {
	if len(q.jobs) == 0 { return api.Job{}, false }
	job := q.jobs[0]
	q.jobs = q.jobs[1:]
	return job, true
}
