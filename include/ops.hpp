#pragma once
#include "image.hpp"
#include <vector>

// Bounds the __constant__ weight array and the shared-memory tile.
constexpr int kMaxRadius = 15;

// Normalised 1-D Gaussian, length 2*radius+1. sigma <= 0 picks radius/2.
std::vector<float> gaussian_weights(int radius, float sigma);

// CPU reference implementations. These validate the kernels, not compete.
namespace cpu {
Image grayscale(const Image& rgb);
Image gaussian_blur(const Image& gray, int radius, float sigma);
Image sobel(const Image& gray);
std::vector<unsigned int> histogram(const Image& gray);
std::vector<unsigned char> equalize_lut(const std::vector<unsigned int>& hist, size_t total);
Image apply_lut(const Image& gray, const std::vector<unsigned char>& lut);
}  // namespace cpu

// GPU launchers. Raw device pointers, so the benchmark allocates once and
// times kernels without cudaMalloc/cudaMemcpy in the way.
namespace gpu {

// naive: 1 px/thread. vec4: 4 px/thread, so 12 bytes are three 32-bit loads.
void grayscale_naive(const unsigned char* d_rgb, unsigned char* d_gray, int w, int h);
void grayscale_vec4 (const unsigned char* d_rgb, unsigned char* d_gray, int w, int h);

// Call before any blur.
void upload_gaussian(const std::vector<float>& weights);

// Single-channel Gaussian blur.
//   naive     - (2r+1)^2 global loads per pixel
//   shared    - same maths, tile + halo in shared memory
//   separable - two 1-D passes, float intermediate
void blur_naive    (const unsigned char* d_src, unsigned char* d_dst, int w, int h, int radius);
void blur_shared   (const unsigned char* d_src, unsigned char* d_dst, int w, int h, int radius);
void blur_separable(const unsigned char* d_src, float* d_tmp,
                    unsigned char* d_dst, int w, int h, int radius);

// 3x3 Sobel magnitude on a single channel.
void sobel_naive (const unsigned char* d_src, unsigned char* d_dst, int w, int h);
void sobel_shared(const unsigned char* d_src, unsigned char* d_dst, int w, int h);

// global: atomics straight to global memory. shared: privatised per block.
void histogram_global(const unsigned char* d_src, int n, unsigned int* d_hist);
void histogram_shared(const unsigned char* d_src, int n, unsigned int* d_hist);

// 256-entry LUT into __constant__ memory, then one lookup per pixel.
void upload_lut(const std::vector<unsigned char>& lut);
void apply_lut(const unsigned char* d_src, unsigned char* d_dst, int n);

}  // namespace gpu

// Whole-image wrappers used by the CLI.
Image run_grayscale(const Image& rgb);
Image run_blur(const Image& gray, int radius, float sigma);
Image run_sobel(const Image& gray);
Image run_equalize(const Image& gray);
Image run_pipeline(const Image& rgb, int radius, float sigma);

void print_device_info();
void run_benchmark(int width, int height, int radius, float sigma, int iters);
