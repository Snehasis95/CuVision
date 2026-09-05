#include "cuda_utils.cuh"
#include "device_math.cuh"
#include "ops.hpp"

namespace {

constexpr int kTile = 16;
constexpr int kHalo = 1;                 // 3x3 stencil
constexpr int kSh   = kTile + 2 * kHalo; // 18

__device__ __forceinline__ float sobel_mag(const float p[3][3]) {
    const float gx = -p[0][0] - 2.0f * p[1][0] - p[2][0]
                     + p[0][2] + 2.0f * p[1][2] + p[2][2];
    const float gy = -p[0][0] - 2.0f * p[0][1] - p[0][2]
                     + p[2][0] + 2.0f * p[2][1] + p[2][2];
    return sqrtf(gx * gx + gy * gy);
}

__global__ void k_sobel_naive(const unsigned char* __restrict__ src,
                              unsigned char* __restrict__ dst, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float p[3][3];
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            p[dy + 1][dx + 1] = src[static_cast<size_t>(clampi(y + dy, 0, H - 1)) * W
                                    + clampi(x + dx, 0, W - 1)];

    dst[static_cast<size_t>(y) * W + x] = to_u8(sobel_mag(p));
}

// Same arithmetic, but each input pixel is fetched once per block rather than
// up to nine times.
__global__ void k_sobel_shared(const unsigned char* __restrict__ src,
                               unsigned char* __restrict__ dst, int W, int H) {
    __shared__ float s[kSh][kSh];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x0 = blockIdx.x * kTile, y0 = blockIdx.y * kTile;

    for (int i = ty * kTile + tx; i < kSh * kSh; i += kTile * kTile) {
        const int sy = i / kSh, sx = i - sy * kSh;
        const int gy = clampi(y0 + sy - kHalo, 0, H - 1);
        const int gx = clampi(x0 + sx - kHalo, 0, W - 1);
        s[sy][sx] = src[static_cast<size_t>(gy) * W + gx];
    }
    __syncthreads();

    const int x = x0 + tx, y = y0 + ty;
    if (x >= W || y >= H) return;

    float p[3][3];
    for (int dy = 0; dy < 3; ++dy)
        for (int dx = 0; dx < 3; ++dx) p[dy][dx] = s[ty + dy][tx + dx];

    dst[static_cast<size_t>(y) * W + x] = to_u8(sobel_mag(p));
}

}  // namespace

namespace gpu {

void sobel_naive(const unsigned char* d_src, unsigned char* d_dst, int w, int h) {
    const dim3 block(kTile, kTile);
    const dim3 grid((w + kTile - 1) / kTile, (h + kTile - 1) / kTile);
    k_sobel_naive<<<grid, block>>>(d_src, d_dst, w, h);
}

void sobel_shared(const unsigned char* d_src, unsigned char* d_dst, int w, int h) {
    const dim3 block(kTile, kTile);
    const dim3 grid((w + kTile - 1) / kTile, (h + kTile - 1) / kTile);
    k_sobel_shared<<<grid, block>>>(d_src, d_dst, w, h);
}

}  // namespace gpu
