package ivf

import "vecsearch/distance"

import "container/heap"

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


func Search(query, vecs []float32, norms []float64, centers []float32, inverted_idx [][]int, k int, nprobe int) []int {
	probes := &topK{}
    dim := len(query)
    qn := distance.Norm(query)
    
    for c := 0; c < len(inverted_idx); c++ {
        center := centers[c*dim : (c+1)*dim]
        d := distance.Angular(query, center, qn, distance.Norm(center))
        if probes.Len() < nprobe {
			heap.Push(probes, neighbor{d, c})
		} else if d < (*probes)[0].dist {
			heap.Pop(probes)
			heap.Push(probes, neighbor{d, c})
		}
    }

	h := &topK{}
	for _, probe := range *probes {
		for _, row := range inverted_idx[probe.row] {
			vec := vecs[row*dim : (row+1)*dim]
			d := distance.Angular(query, vec, qn, norms[row])
	
			if h.Len() < k {
				heap.Push(h, neighbor{d, row})
			} else if d < (*h)[0].dist {
				heap.Pop(h)
				heap.Push(h, neighbor{d, row})
			}
		}
	}

	ids := make([]int, h.Len())
	for i := len(ids) - 1; i >= 0; i-- {
		ids[i] = heap.Pop(h).(neighbor).row
	}

	return ids
}
