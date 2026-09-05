#include "cuda_utils.cuh"
#include "device_math.cuh"
#include "ops.hpp"

#include <stdexcept>

// Every thread in a warp reads the same tap at once, which constant memory
// broadcasts in a single cycle.
__constant__ float c_gauss[2 * kMaxRadius + 1];

namespace {

constexpr int kTile = 16;   // output tile edge for the 2-D kernels
constexpr int kHBlk = 128;  // threads per row, horizontal separable pass
constexpr int kVBlkX = 32, kVBlkY = 8;

// Variant 1: every tap straight from global memory.
__global__ void k_blur_naive(const unsigned char* __restrict__ src,
                             unsigned char* __restrict__ dst,
                             int W, int H, int r) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const int   gy = clampi(y + dy, 0, H - 1);
        const float wy = c_gauss[dy + r];
        for (int dx = -r; dx <= r; ++dx) {
            const int gx = clampi(x + dx, 0, W - 1);
            acc += wy * c_gauss[dx + r] * src[static_cast<size_t>(gy) * W + gx];
        }
    }
    dst[static_cast<size_t>(y) * W + x] = to_u8(acc);
}

// Variant 2: tile plus halo in shared memory. Neighbouring outputs overlap by
// 2r taps, so one staged load replaces (2r+1)^2 global reads per pixel.
__global__ void k_blur_shared(const unsigned char* __restrict__ src,
                              unsigned char* __restrict__ dst,
                              int W, int H, int r) {
    extern __shared__ float s[];
    const int sw = kTile + 2 * r;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x0 = blockIdx.x * kTile, y0 = blockIdx.y * kTile;

    // sw*sw elements fetched by kTile*kTile threads.
    for (int i = ty * kTile + tx; i < sw * sw; i += kTile * kTile) {
        const int sy = i / sw, sx = i - sy * sw;
        const int gy = clampi(y0 + sy - r, 0, H - 1);
        const int gx = clampi(x0 + sx - r, 0, W - 1);
        s[i] = src[static_cast<size_t>(gy) * W + gx];
    }
    __syncthreads();

    const int x = x0 + tx, y = y0 + ty;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const float  wy  = c_gauss[dy + r];
        const float* row = &s[(ty + r + dy) * sw + (tx + r)];
        for (int dx = -r; dx <= r; ++dx) acc += wy * c_gauss[dx + r] * row[dx];
    }
    dst[static_cast<size_t>(y) * W + x] = to_u8(acc);
}

// Variant 3: two 1-D passes. The Gaussian is separable, so per-pixel work
// drops from (2r+1)^2 to 2*(2r+1) - 289 multiply-adds against 34 at r=8.
__global__ void k_blur_h(const unsigned char* __restrict__ src,
                         float* __restrict__ tmp, int W, int H, int r) {
    extern __shared__ float s[];
    const int tx = threadIdx.x;
    const int x0 = blockIdx.x * kHBlk;
    const int y  = blockIdx.y;
    const int sw = kHBlk + 2 * r;

    for (int i = tx; i < sw; i += kHBlk)
        s[i] = src[static_cast<size_t>(y) * W + clampi(x0 + i - r, 0, W - 1)];
    __syncthreads();

    const int x = x0 + tx;
    if (x >= W) return;
    float acc = 0.0f;
    for (int k = -r; k <= r; ++k) acc += c_gauss[k + r] * s[tx + r + k];
    tmp[static_cast<size_t>(y) * W + x] = acc;
}

__global__ void k_blur_v(const float* __restrict__ tmp,
                         unsigned char* __restrict__ dst, int W, int H, int r) {
    extern __shared__ float s[];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x  = blockIdx.x * kVBlkX + tx;
    const int y0 = blockIdx.y * kVBlkY;
    const int sh = kVBlkY + 2 * r;
    const int xc = min(x, W - 1);   // clamp so halo columns stay in range

    for (int i = ty; i < sh; i += kVBlkY)
        s[i * kVBlkX + tx] = tmp[static_cast<size_t>(clampi(y0 + i - r, 0, H - 1)) * W + xc];
    __syncthreads();

    const int y = y0 + ty;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int k = -r; k <= r; ++k) acc += c_gauss[k + r] * s[(ty + r + k) * kVBlkX + tx];
    dst[static_cast<size_t>(y) * W + x] = to_u8(acc);
}

}  // namespace

namespace gpu {

void upload_gaussian(const std::vector<float>& weights) {
    if (weights.size() > static_cast<size_t>(2 * kMaxRadius + 1))
        throw std::runtime_error("gaussian kernel exceeds constant-memory slot");
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, weights.data(), weights.size() * sizeof(float)));
}

void blur_naive(const unsigned char* d_src, unsigned char* d_dst, int w, int h, int r) {
    const dim3 block(kTile, kTile);
    const dim3 grid((w + kTile - 1) / kTile, (h + kTile - 1) / kTile);
    k_blur_naive<<<grid, block>>>(d_src, d_dst, w, h, r);
}

void blur_shared(const unsigned char* d_src, unsigned char* d_dst, int w, int h, int r) {
    const dim3 block(kTile, kTile);
    const dim3 grid((w + kTile - 1) / kTile, (h + kTile - 1) / kTile);
    const int  sw    = kTile + 2 * r;
    const size_t smem = static_cast<size_t>(sw) * sw * sizeof(float);
    k_blur_shared<<<grid, block, smem>>>(d_src, d_dst, w, h, r);
}

void blur_separable(const unsigned char* d_src, float* d_tmp,
                    unsigned char* d_dst, int w, int h, int r) {
    const dim3 hgrid((w + kHBlk - 1) / kHBlk, h);
    k_blur_h<<<hgrid, kHBlk, (kHBlk + 2 * r) * sizeof(float)>>>(d_src, d_tmp, w, h, r);

    const dim3 vblock(kVBlkX, kVBlkY);
    const dim3 vgrid((w + kVBlkX - 1) / kVBlkX, (h + kVBlkY - 1) / kVBlkY);
    k_blur_v<<<vgrid, vblock, static_cast<size_t>(kVBlkX) * (kVBlkY + 2 * r) * sizeof(float)>>>(
        d_tmp, d_dst, w, h, r);
}

}  // namespace gpu
