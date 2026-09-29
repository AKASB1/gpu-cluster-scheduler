package api

type Job struct {
	ID string
	Tenant string
	GPUs int
	Priority int
}

func (j Job) Valid() bool { return j.ID != "" && j.GPUs > 0 }
