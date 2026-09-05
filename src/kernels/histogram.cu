#include "cuda_utils.cuh"
#include "device_math.cuh"
#include "ops.hpp"

#include <stdexcept>

__constant__ unsigned char c_lut[256];

namespace {

constexpr int kBlock = 256;
constexpr int kBins  = 256;

// Grid-stride so the launch shape is independent of image size.
__global__ void k_hist_global(const unsigned char* __restrict__ src, int n,
                              unsigned int* __restrict__ hist) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&hist[src[i]], 1u);
}

// Each block accumulates into its own shared copy and contributes at most 256
// global atomics. Most pixels land in a few hot bins, so keeping that
// contention in shared memory is the win.
__global__ void k_hist_shared(const unsigned char* __restrict__ src, int n,
                              unsigned int* __restrict__ hist) {
    __shared__ unsigned int sh[kBins];
    for (int i = threadIdx.x; i < kBins; i += blockDim.x) sh[i] = 0u;
    __syncthreads();

    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&sh[src[i]], 1u);
    __syncthreads();

    for (int i = threadIdx.x; i < kBins; i += blockDim.x)
        if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

__global__ void k_apply_lut(const unsigned char* __restrict__ src,
                            unsigned char* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = c_lut[src[i]];
}

// Enough blocks to fill the device; the grid-stride loop covers the rest.
int grid_for(int n) {
    const int want = (n + kBlock - 1) / kBlock;
    return want < 1024 ? (want < 1 ? 1 : want) : 1024;
}

}  // namespace

namespace gpu {

void histogram_global(const unsigned char* d_src, int n, unsigned int* d_hist) {
    k_hist_global<<<grid_for(n), kBlock>>>(d_src, n, d_hist);
}

void histogram_shared(const unsigned char* d_src, int n, unsigned int* d_hist) {
    k_hist_shared<<<grid_for(n), kBlock>>>(d_src, n, d_hist);
}

void upload_lut(const std::vector<unsigned char>& lut) {
    if (lut.size() != kBins) throw std::runtime_error("LUT must have 256 entries");
    CUDA_CHECK(cudaMemcpyToSymbol(c_lut, lut.data(), kBins));
}

void apply_lut(const unsigned char* d_src, unsigned char* d_dst, int n) {
    k_apply_lut<<<(n + kBlock - 1) / kBlock, kBlock>>>(d_src, d_dst, n);
}

}  // namespace gpu
