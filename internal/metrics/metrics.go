package metrics

func Utilization(allocated, capacity int) float64 {
	if capacity <= 0 || allocated < 0 || allocated > capacity { return 0 }
	return float64(allocated) / float64(capacity)
}
