package distance

import "math"

func Norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		fx := float64(x)
		sum += fx * fx
	}
	return math.Sqrt(sum)
}

func Angular(a, b []float32, normA, normB float64) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return 1 - dot/(normA*normB)
}
