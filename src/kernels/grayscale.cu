#include "cuda_utils.cuh"
#include "device_math.cuh"
#include "ops.hpp"

namespace {

constexpr int kBlock = 256;

// Baseline. Consecutive threads read at 3-byte stride, so a warp's loads
// straddle sectors and cost more transactions than the byte count needs.
__global__ void k_gray_naive(const unsigned char* __restrict__ src,
                             unsigned char* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = to_u8(luma(src[3 * i + 0], src[3 * i + 1], src[3 * i + 2]));
}

// 4 px * 3 ch = 12 bytes: three 32-bit loads, one 32-bit store. cudaMalloc is
// 256-byte aligned, so the uchar4 reinterpret is safe.
__global__ void k_gray_vec4(const uchar4* __restrict__ src4,
                            uchar4* __restrict__ dst4,
                            const unsigned char* __restrict__ src,
                            unsigned char* __restrict__ dst,
                            int n) {
    const int q  = blockIdx.x * blockDim.x + threadIdx.x;
    const int nq = n >> 2;

    if (q < nq) {
        const uchar4 a = src4[3 * q + 0];   // r0 g0 b0 r1
        const uchar4 b = src4[3 * q + 1];   // g1 b1 r2 g2
        const uchar4 c = src4[3 * q + 2];   // b2 r3 g3 b3
        uchar4 out;
        out.x = to_u8(luma(a.x, a.y, a.z));
        out.y = to_u8(luma(a.w, b.x, b.y));
        out.z = to_u8(luma(b.z, b.w, c.x));
        out.w = to_u8(luma(c.y, c.z, c.w));
        dst4[q] = out;
    } else if (q == nq) {
        // Tail of at most three pixels.
        for (int p = nq * 4; p < n; ++p)
            dst[p] = to_u8(luma(src[3 * p + 0], src[3 * p + 1], src[3 * p + 2]));
    }
}

}  // namespace

namespace gpu {

void grayscale_naive(const unsigned char* d_rgb, unsigned char* d_gray, int w, int h) {
    const int n = w * h;
    k_gray_naive<<<(n + kBlock - 1) / kBlock, kBlock>>>(d_rgb, d_gray, n);
}

void grayscale_vec4(const unsigned char* d_rgb, unsigned char* d_gray, int w, int h) {
    const int n       = w * h;
    const int threads = (n >> 2) + 1;   // +1 covers the tail thread
    k_gray_vec4<<<(threads + kBlock - 1) / kBlock, kBlock>>>(
        reinterpret_cast<const uchar4*>(d_rgb),
        reinterpret_cast<uchar4*>(d_gray),
        d_rgb, d_gray, n);
}

}  // namespace gpu
