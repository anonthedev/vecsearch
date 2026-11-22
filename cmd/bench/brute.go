package main

import (
	"container/heap"

	"vecsearch/distance"
)

type neighbor struct {
	dist float64
	row  int
}

type topK []neighbor

func (h topK) Len() int           { return len(h) }
func (h topK) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h topK) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *topK) Push(x any)        { *h = append(*h, x.(neighbor)) }
func (h *topK) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func vectorNorms(vecs []float32, dim int) []float64 {
	n := len(vecs) / dim
	norms := make([]float64, n)
	for i := 0; i < n; i++ {
		norms[i] = distance.Norm(vecs[i*dim : (i+1)*dim])
	}
	return norms
}

func searchBrute(train []float32, norms []float64, query []float32, k int) []int {
	dim := len(query)
	qn := distance.Norm(query)
	h := &topK{}
	for j := 0; j < len(norms); j++ {
		row := train[j*dim : (j+1)*dim]
		d := distance.Angular(query, row, qn, norms[j])
		if h.Len() < k {
			heap.Push(h, neighbor{d, j})
		} else if d < (*h)[0].dist {
			heap.Pop(h)
			heap.Push(h, neighbor{d, j})
		}
	}
	ids := make([]int, h.Len())
	for i := len(ids) - 1; i >= 0; i-- {
		ids[i] = heap.Pop(h).(neighbor).row
	}
	return ids
}
