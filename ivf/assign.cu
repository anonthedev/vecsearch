#include "assign.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static char errbuf[256];

static float* d_vecs;
static float* d_centers;
static float* d_norms;
static int* d_assign;
static int haveN;
static int haveDim;
static int haveK;

static int fail(cudaError_t st, const char* what) {
	if (st == cudaSuccess) {
		return 0;
	}
	snprintf(errbuf, sizeof errbuf, "%s: %s", what, cudaGetErrorString(st));
	return 1;
}

const char* assign_last_error(void) {
	return errbuf;
}

int assign_upload(const float* vecs, int n, int dim) {
	if (d_vecs && haveN == n && haveDim == dim) {
		return 0;
	}
	if (d_vecs) {
		cudaFree(d_vecs);
		d_vecs = 0;
	}
	size_t bytes = (size_t)n * (size_t)dim * sizeof(float);
	if (fail(cudaMalloc(&d_vecs, bytes), "cudaMalloc vecs")) {
		return 1;
	}
	if (fail(cudaMemcpy(d_vecs, vecs, bytes, cudaMemcpyHostToDevice), "cudaMemcpy vecs")) {
		return 1;
	}
	haveN = n;
	haveDim = dim;
	return 0;
}

__global__ void assignKernel(const float* vecs, const float* centers, const float* centerNorms, int* assign, int n, int k, int dim) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	const float* row = vecs + (size_t)i * dim;
	int best = 0;
	float bestScore = -INFINITY;
	for (int c = 0; c < k; c++) {
		const float* center = centers + (size_t)c * dim;
		float dot = 0.f;
		for (int d = 0; d < dim; d++) {
			dot += row[d] * center[d];
		}
		float norm = centerNorms[c];
		float score = norm == 0.f ? -INFINITY : dot / norm;
		if (score > bestScore) {
			bestScore = score;
			best = c;
		}
	}
	assign[i] = best;
}

// One thread per row. The row is loaded once. A block stages 32 centers in
// shared memory and every thread in the block scores that same chunk.
__global__ void assignKernelDim100(const float* vecs, const float* centers, const float* centerNorms, int* assign, int n, int k) {
	const int dim = 100;
	const int tile = 32;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float row[100];
	if (i < n) {
		const float* src = vecs + (size_t)i * dim;
		#pragma unroll
		for (int d = 0; d < dim; d++) {
			row[d] = src[d];
		}
	}

	__shared__ float tileVecs[32 * 100];
	__shared__ float tileNorms[32];

	int best = 0;
	float bestScore = -INFINITY;
	for (int c0 = 0; c0 < k; c0 += tile) {
		int ncent = k - c0;
		if (ncent > tile) {
			ncent = tile;
		}
		for (int j = threadIdx.x; j < ncent * dim; j += blockDim.x) {
			tileVecs[j] = centers[(size_t)c0 * dim + j];
		}
		if (threadIdx.x < ncent) {
			tileNorms[threadIdx.x] = centerNorms[c0 + threadIdx.x];
		}
		__syncthreads();

		if (i < n) {
			for (int c = 0; c < ncent; c++) {
				const float* center = tileVecs + c * dim;
				float dot = 0.f;
				#pragma unroll
				for (int d = 0; d < dim; d++) {
					dot += row[d] * center[d];
				}
				float norm = tileNorms[c];
				float score = norm == 0.f ? -INFINITY : dot / norm;
				if (score > bestScore) {
					bestScore = score;
					best = c0 + c;
				}
			}
		}
		__syncthreads();
	}

	if (i < n) {
		assign[i] = best;
	}
}

int assign_run(const float* centers, const float* centerNorms, int* assign, int n, int k, int dim) {
	size_t centerBytes = (size_t)k * (size_t)dim * sizeof(float);
	size_t normBytes = (size_t)k * sizeof(float);
	size_t assignBytes = (size_t)n * sizeof(int);
	if (haveK != k || haveDim != dim) {
		if (d_centers) {
			cudaFree(d_centers);
		}
		if (d_norms) {
			cudaFree(d_norms);
		}
		if (fail(cudaMalloc(&d_centers, centerBytes), "cudaMalloc centers")) {
			return 1;
		}
		if (fail(cudaMalloc(&d_norms, normBytes), "cudaMalloc norms")) {
			return 1;
		}
		haveK = k;
		haveDim = dim;
	}
	if (!d_assign || haveN != n) {
		if (d_assign) {
			cudaFree(d_assign);
		}
		if (fail(cudaMalloc(&d_assign, assignBytes), "cudaMalloc assign")) {
			return 1;
		}
		haveN = n;
	}
	if (fail(cudaMemcpy(d_centers, centers, centerBytes, cudaMemcpyHostToDevice), "cudaMemcpy centers")) {
		return 1;
	}
	if (fail(cudaMemcpy(d_norms, centerNorms, normBytes, cudaMemcpyHostToDevice), "cudaMemcpy norms")) {
		return 1;
	}

	int threads = 256;
	int blocks = (n + threads - 1) / threads;
	if (dim == 100) {
		assignKernelDim100<<<blocks, threads>>>(d_vecs, d_centers, d_norms, d_assign, n, k);
	} else {
		assignKernel<<<blocks, threads>>>(d_vecs, d_centers, d_norms, d_assign, n, k, dim);
	}
	if (fail(cudaGetLastError(), "assign kernel")) {
		return 1;
	}
	if (fail(cudaDeviceSynchronize(), "assign sync")) {
		return 1;
	}
	if (fail(cudaMemcpy(assign, d_assign, assignBytes, cudaMemcpyDeviceToHost), "cudaMemcpy assign")) {
		return 1;
	}
	return 0;
}

void assign_release(void) {
	cudaFree(d_vecs);
	cudaFree(d_centers);
	cudaFree(d_norms);
	cudaFree(d_assign);
	d_vecs = 0;
	d_centers = 0;
	d_norms = 0;
	d_assign = 0;
	haveN = 0;
	haveDim = 0;
	haveK = 0;
}
