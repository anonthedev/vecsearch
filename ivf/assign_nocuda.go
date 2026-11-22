//go:build !cuda

package ivf

func assignGPU(vecs, centers []float32, centerNorms []float64, assign []int) {
	panic("GPU assignment needs: go run -tags cuda")
}

func releaseGPU() {}
