//go:build cuda

package ivf

// #cgo LDFLAGS: ${SRCDIR}/assign_cuda.o -L/usr/local/cuda/lib64 -lcudart -lstdc++ -Wl,-rpath,/usr/local/cuda/lib64
// #include "assign.h"
import "C"
import "unsafe"

func assignGPU(vecs, centers []float32, centerNorms []float64, assign []int) {
	n := len(assign)
	k := len(centerNorms)
	dim := len(centers) / k
	norms := make([]float32, k)
	for i, norm := range centerNorms {
		norms[i] = float32(norm)
	}
	ids := make([]int32, n)

	if C.assign_upload((*C.float)(unsafe.Pointer(&vecs[0])), C.int(n), C.int(dim)) != 0 {
		panic(C.GoString(C.assign_last_error()))
	}
	if C.assign_run(
		(*C.float)(unsafe.Pointer(&centers[0])),
		(*C.float)(unsafe.Pointer(&norms[0])),
		(*C.int)(unsafe.Pointer(&ids[0])),
		C.int(n), C.int(k), C.int(dim),
	) != 0 {
		panic(C.GoString(C.assign_last_error()))
	}
	for i, id := range ids {
		assign[i] = int(id)
	}
}

func releaseGPU() {
	C.assign_release()
}
