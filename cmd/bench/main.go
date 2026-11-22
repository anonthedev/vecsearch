package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"vecsearch/bench"
	"vecsearch/ivf"
)

const (
	dim = 100
	nq  = 100
	k   = 10
)

func main() {
	load := flag.Bool("load", false, "load the saved ivf index")
	fresh := flag.Bool("fresh", false, "train a new ivf index and save it")
	gpu := flag.Bool("gpu", false, "assign k-means clusters on the GPU")
	flag.Parse()
	if *load && *fresh {
		log.Fatal("pass either --load or --fresh")
	}

	log.Printf("loading data")
	startBuild := time.Now()
	testVecs := readFile[float32](os.Open("data/test.bin"))
	trainVecs := readFile[float32](os.Open("data/train.bin"))
	neighborsVecs := readFile[int32](os.Open("data/neighbors.bin"))

	log.Printf("computing norms")
	normStart := time.Now()
	normTrainVecs := vectorNorms(trainVecs, dim)
	normTime := time.Since(normStart)
	log.Printf("norms %s", normTime)

	const indexPath = "data/ivf16384.bin"
	var centers []float32
	var lists [][]int
	_, statErr := os.Stat(indexPath)
	useLoad := *load || (!*fresh && statErr == nil)
	if useLoad {
		log.Printf("loading ivf %s", indexPath)
		centers, lists = ivf.Load(indexPath)
	} else {
		log.Printf("training ivf")
		centers, _, lists = ivf.Train(trainVecs, normTrainVecs, dim, 16384, 10, *gpu)
		log.Printf("saving ivf %s", indexPath)
		ivf.Save(indexPath, centers, lists, dim)
	}
	buildTime := time.Since(startBuild)
	log.Printf("build %s", buildTime)

	n := len(trainVecs) / dim
	bytes := len(trainVecs)*4 + len(normTrainVecs)*8

	bench.Run(bench.Config{
		Name:           "brute",
		NQ:             nq,
		K:              k,
		Dim:            dim,
		NeighborStride: dim,
		Test:           testVecs,
		Neighbors:      neighborsVecs,
		Build:          buildTime,
		Norms:          normTime,
		BytesPerVector: bytes / n,
	}, func(query []float32) []int {
		return searchBrute(trainVecs, normTrainVecs, query, k)
	})

	for _, nprobe := range []int{128, 256, 512, 1024, 2048} {
		bench.Run(bench.Config{
			Name:           fmt.Sprintf("ivf16384-%d", nprobe),
			NQ:             nq,
			K:              k,
			Dim:            dim,
			NeighborStride: dim,
			Test:           testVecs,
			Neighbors:      neighborsVecs,
			Build:          buildTime,
			Norms:          normTime,
			BytesPerVector: bytes / n,
		}, func(query []float32) []int {
			return ivf.Search(query, trainVecs, normTrainVecs, centers, lists, k, nprobe)
		})
	}
}

func readFile[T float32 | int32](file *os.File, err error) []T {
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var n, dims int32
	binary.Read(file, binary.LittleEndian, &n)
	binary.Read(file, binary.LittleEndian, &dims)

	vecs := make([]T, int(n)*int(dims))
	binary.Read(file, binary.LittleEndian, vecs)
	return vecs
}
