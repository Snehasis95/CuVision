#pragma once
#include <cuda_runtime.h>

// Must match the host clamp_u8 exactly or the two diverge.
__device__ __forceinline__ unsigned char to_u8(float v) {
    return static_cast<unsigned char>(fminf(fmaxf(v + 0.5f, 0.0f), 255.0f));
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return min(max(v, lo), hi);
}

// ITU-R BT.601 luma.
__device__ __forceinline__ float luma(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}
