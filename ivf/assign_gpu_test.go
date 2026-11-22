//go:build cuda

package ivf

import (
	"math"
	"math/rand"
	"testing"

	"vecsearch/distance"
)

func TestAssignGPUMatchesFloatScore(t *testing.T) {
	for _, k := range []int{32, 40} {
		t.Run("", func(t *testing.T) {
			checkAssignGPU(t, 1000, k, 100)
		})
	}
}

func checkAssignGPU(t *testing.T, n, k, dim int) {
	rng := rand.New(rand.NewSource(1))
	vecs := make([]float32, n*dim)
	centers := make([]float32, k*dim)
	for i := range vecs {
		vecs[i] = rng.Float32()*2 - 1
	}
	for i := range centers {
		centers[i] = rng.Float32()*2 - 1
	}
	centerNorms := make([]float64, k)
	for c := 0; c < k; c++ {
		centerNorms[c] = distance.Norm(centers[c*dim : (c+1)*dim])
	}

	got := make([]int, n)
	assignGPU(vecs, centers, centerNorms, got)
	releaseGPU()

	norms32 := make([]float32, k)
	for c, norm := range centerNorms {
		norms32[c] = float32(norm)
	}
	for i := 0; i < n; i++ {
		best, bestScore := 0, float32(math.Inf(-1))
		row := vecs[i*dim : (i+1)*dim]
		for c := 0; c < k; c++ {
			var dot float32
			center := centers[c*dim : (c+1)*dim]
			for d := 0; d < dim; d++ {
				dot += row[d] * center[d]
			}
			score := float32(math.Inf(-1))
			if norms32[c] != 0 {
				score = dot / norms32[c]
			}
			if score > bestScore {
				best, bestScore = c, score
			}
		}
		if got[i] != best {
			t.Fatalf("row %d: gpu %d, reference %d", i, got[i], best)
		}
	}
}
