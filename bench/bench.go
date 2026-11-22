package bench

import (
	"fmt"
	"log"
	"os"
	"sort"
	"time"
)

type Config struct {
	Name           string
	NQ             int
	K              int
	Dim            int
	NeighborStride int
	Test           []float32
	Neighbors      []int32
	Build          time.Duration
	Norms          time.Duration
	BytesPerVector int
}

func Run(cfg Config, search func(query []float32) []int) {
	hits := 0
	latencies := make([]time.Duration, cfg.NQ)
	log.Printf("search %s, %d queries", cfg.Name, cfg.NQ)
	start := time.Now()
	for i := 0; i < cfg.NQ; i++ {
		qStart := time.Now()
		ids := search(cfg.Test[i*cfg.Dim : (i+1)*cfg.Dim])
		latencies[i] = time.Since(qStart)
		if (i+1)%10 == 0 {
			log.Printf("%s %d/%d", cfg.Name, i+1, cfg.NQ)
		}

		truth := cfg.Neighbors[i*cfg.NeighborStride : i*cfg.NeighborStride+cfg.K]
		for _, id := range ids {
			for _, t := range truth {
				if id == int(t) {
					hits++
					break
				}
			}
		}
	}
	elapsed := time.Since(start)
	sort.Slice(latencies, func(a, b int) bool {
		return latencies[a] < latencies[b]
	})

	recall := float64(hits) / float64(cfg.NQ*cfg.K)
	qps := float64(cfg.NQ) / elapsed.Seconds()

	fmt.Printf("queries:   %d\n", cfg.NQ)
	fmt.Printf("recall@10: %.3f\n", recall)
	fmt.Printf("latency:   %s\n", elapsed/time.Duration(cfg.NQ))
	fmt.Printf("p50:       %s\n", latencies[cfg.NQ/2-1])
	fmt.Printf("p99:       %s\n", latencies[cfg.NQ*99/100-1])
	fmt.Printf("qps:       %.1f\n", qps)
	fmt.Printf("norms:     %s\n", cfg.Norms)
	fmt.Printf("memory:    %d bytes/vector\n", cfg.BytesPerVector)
	fmt.Printf("build:     %s\n", cfg.Build)

	f, err := os.OpenFile("data/results.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	fmt.Fprintf(f, "%s,%.3f,%.1f\n", cfg.Name, recall, qps)
}
