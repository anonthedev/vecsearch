package ivf

import (
	"log"
	"math"
	"math/rand"
	"time"
	"vecsearch/distance"
)

func Train(vecs []float32, norms []float64, dim, k, iters int, gpu bool) ([]float32, []int, [][]int) {
	n := len(vecs)/dim //no of rows

	centers := make([]float32, k*dim) //k centers, each of dim dimensions
	for center:=0; center<k; center++ {
		src := rand.Intn(n) * dim
		copy(centers[center*dim:(center+1)*dim], vecs[src:src+dim])
	}//assign random rows as initial centers

	assign := make([]int, n) //assign each row to a center values will range from 0 to k-1
	centersNorms := make([]float64, k) //norms of centers

	if gpu {
		log.Printf("kmeans assign on gpu")
	}

	for iter:=0; iter<iters; iter++{
		pass := time.Now()
		log.Printf("kmeans %d/%d", iter+1, iters)
		for center :=0; center<k; center++ {
			centersNorms[center] = distance.Norm(centers[center*dim:(center+1)*dim])
		}//creating norms of centers

		if gpu {
			assignGPU(vecs, centers, centersNorms, assign)
		} else {
			for i:=0; i<n; i++ {
				row := vecs[i*dim:(i+1)*dim]
				best, bestDist :=0, distance.Angular(row, centers[:dim], norms[i], centersNorms[0])
				//initial best is the first center, and bestDist is the distance to the first center

				//loop through all centers to find the best center for each row
				for center:=1; center<k; center++ {
					d := distance.Angular(row, centers[center*dim:(center+1)*dim], norms[i], centersNorms[center])
					if d<bestDist {
						best, bestDist = center, d
					}
				}
				//assign the best center to the row
				assign[i] = best
			}
		}

		log.Printf("kmeans %d/%d %s", iter+1, iters, time.Since(pass))
		if iter == iters-1 {
			break
		}

		sums := make([]float64, k*dim) //sum of vectos of each center
		counts := make([]int, k) //count of vectors in each center

		for i := 0; i<n; i++{
			c := assign[i]
			counts[c]++
			for d:=0; d<dim; d++{
				sums[c*dim+d] += float64(vecs[i*dim+d])
			}
		}

		for center := 0; center<k; center++{
			if counts[center] == 0{
				src := rand.Intn(n) * dim
				copy(centers[center*dim:(center+1)*dim], vecs[src:src+dim])
				continue
			}

			var length float64

			for d:=0; d<dim; d++{
				avg := sums[center*dim+d]/float64(counts[center])
				centers[center*dim+d] = float32(avg)
				length += avg*avg
			}
			length = math.Sqrt(length)
			if length == 0{
				continue
			}

			inv := float32(1/length)
			for d:=0; d<dim; d++{
				centers[center*dim+d] *= inv
			}
		}
	}

	if gpu {
		releaseGPU()
	}

	inverted_idx := make([][]int, k)

	for i:=0; i<n; i++{
		center := assign[i]
		inverted_idx[center] = append(inverted_idx[center], i)
	}

	return centers, assign, inverted_idx
}