# vecsearch

Nearest-neighbor search in Go, written to see where the time goes. There are two indexes: a brute-force scan, and IVF (inverted file). Distance is angular, `1 - cosine similarity`, with vector norms computed once up front.

## Data

[GloVe](https://nlp.stanford.edu/projects/glove/) 100-dimensional embeddings, the `glove-100-angular` set from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks):

| | |
|---|---|
| train | 1,183,514 vectors |
| test | 10,000 queries |
| dimension | 100 |
| ground truth | 100 neighbors per query |

`python-script/data.py` reads `python-script/data/glove-100-angular.hdf5` and writes `train.bin`, `test.bin`, `neighbors.bin`, and `distances.bin`. The benchmark reads those four files from `data/`. Both directories are gitignored.

Each `.bin` file is a little-endian header of two `int32`s (`count`, `dim`) followed by the payload: `float32` for vectors and distances, `int32` for neighbor ids.

## Search

Brute force scores the query against every training vector and keeps the 10 nearest with a heap.

IVF trains `nlist` centroids with k-means (10 iterations, centroids re-normalized), then stores each vector in the list of its nearest centroid. A query scores every centroid, opens the `nprobe` closest lists, and scans only those vectors. The index is saved under `data/` and reloaded on the next run.

Assignment can run on the GPU. Build the CUDA object, then pass `-tags cuda`:

```bash
nvcc -c -o ivf/assign_cuda.o ivf/assign.cu
go run -tags cuda ./cmd/bench --fresh --gpu
```

Search itself is still on the CPU.

## Benchmark

`cmd/bench` runs the first 100 test queries, `k = 10`, single-threaded. Recall@10 is the fraction of the dataset's top-10 neighbors that the search returns. The harness also prints mean latency, p50, p99, bytes per vector, and build time, and appends `name,recall,qps` to `data/results.csv`.

```bash
go run ./cmd/bench            # load data/ivf16384.bin if it exists, else train
go run ./cmd/bench --fresh    # train 16384 lists and save
go run ./cmd/bench --load     # require the saved index
```

The checked-in binary sweeps `nprobe` in `{128, 256, 512, 1024, 2048}` on 16384 lists. Earlier sweeps used 64, 2048, 4096, and 8192 lists. `python python-script/chart.py` plots every saved point.

## Results

100 queries, one thread. Recall@10 against the dataset's top 10. Brute force is the baseline: recall 1.000 at 8.8 QPS on the same run as the 16384 sweep. Across runs that number moved between about 6 and 12, so the speedups below are rough. Flat storage is 408 bytes per vector (the `float32` coordinates plus one `float64` norm). The inverted lists sit on top of that.

Fastest setting that still clears each recall target:

| recall at least | setting | lists opened | recall@10 | QPS | vs brute |
|---|---|---:|---:|---:|---:|
| 0.90 | 16384 lists, 256 probes | 1.6% | 0.919 | 119 | 13× |
| 0.95 | 16384 lists, 512 probes | 3.1% | 0.950 | 71 | 8× |
| 0.99 | 8192 lists, 1024 probes | 12.5% | 0.990 | 24 | 2.7× |
| 1.00 | brute force | all vectors | 1.000 | 8.8 | 1× |

"Lists opened" is `nprobe / nlist`. Lists are not equal size, so this is not the fraction of vectors scanned.

Holding recall near 0.99 means opening about an eighth of the lists, and the win over a full scan shrinks to about 3×. Pushing to 0.995 (4096 lists, 1024 probes, a quarter of the lists) is only 10 QPS, barely ahead of brute force. Nothing in the sweep reached recall 1 short of scanning every vector.

Two ends of the same curve show why the middle is the part that matters. One probe on 2048 lists does 2000 QPS at recall 0.41. Probing half of 64 lists (`nprobe` 32) reaches recall 0.996 at 7.4 QPS, which is slower than the full scan.
