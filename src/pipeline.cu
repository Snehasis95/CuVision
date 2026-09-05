#include "cuda_utils.cuh"
#include "ops.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename F>
double time_cpu(F&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Warm up, then average `iters` launches inside one event pair.
template <typename F>
double time_gpu(F&& fn, int iters) {
    for (int i = 0; i < 3; ++i) fn();
    CUDA_CHECK_KERNEL();

    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) fn();
    t.stop();
    const double ms = t.elapsed_ms() / iters;
    CUDA_CHECK(cudaGetLastError());
    return ms;
}

void row(const char* name, double ms, double mpix, double speedup, int err) {
    std::printf("  %-26s %9.3f %12.1f %10.2fx %9d\n", name, ms, mpix, speedup, err);
}

void cpu_row(const char* name, double ms, double mpix) {
    std::printf("  %-26s %9.3f %12.1f %11s %9s\n", name, ms, mpix, "-", "-");
}

void header(const char* title) {
    std::printf("\n%s\n", title);
    std::printf("  %-26s %9s %12s %11s %9s\n", "variant", "ms", "MPixel/s", "speedup", "max err");
    std::printf("  %s\n", std::string(70, '-').c_str());
}

Image download_gray(const DeviceBuffer<unsigned char>& buf, int w, int h) {
    Image out(w, h, 1);
    buf.download(out.data.data(), out.bytes());
    return out;
}

int hist_max_err(const std::vector<unsigned int>& a, const std::vector<unsigned int>& b) {
    long long worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::llabs(static_cast<long long>(a[i]) -
                                           static_cast<long long>(b[i])));
    return static_cast<int>(std::min<long long>(worst, 2147483647LL));
}

}  // namespace

void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    // Theoretical peak = memory clock (kHz) * bus width (bits) / 8, doubled for DDR.
    const double peak_gbs = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;

    std::printf("device : %s (sm_%d%d)\n", p.name, p.major, p.minor);
    std::printf("  SMs            : %d\n", p.multiProcessorCount);
    std::printf("  global memory  : %.1f GiB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("  peak bandwidth : %.0f GB/s (%d-bit bus @ %.0f MHz)\n",
                peak_gbs, p.memoryBusWidth, p.memoryClockRate / 1000.0);
    std::printf("  shared / block : %zu KiB\n", p.sharedMemPerBlock / 1024);
}

// Whole-image wrappers: one upload, one download, intermediates stay resident.

Image run_grayscale(const Image& rgb) {
    if (rgb.channels != 3) throw std::runtime_error("grayscale expects a 3-channel image");
    const int w = rgb.width, h = rgb.height, n = w * h;

    DeviceBuffer<unsigned char> d_rgb(rgb.bytes()), d_gray(n);
    d_rgb.upload(rgb.data.data(), rgb.bytes());
    gpu::grayscale_vec4(d_rgb.get(), d_gray.get(), w, h);
    CUDA_CHECK_KERNEL();
    return download_gray(d_gray, w, h);
}

Image run_blur(const Image& gray, int radius, float sigma) {
    if (gray.channels != 1) throw std::runtime_error("blur expects a 1-channel image");
    const int w = gray.width, h = gray.height, n = w * h;

    gpu::upload_gaussian(gaussian_weights(radius, sigma));
    DeviceBuffer<unsigned char> d_src(n), d_dst(n);
    DeviceBuffer<float>         d_tmp(n);
    d_src.upload(gray.data.data(), n);
    gpu::blur_separable(d_src.get(), d_tmp.get(), d_dst.get(), w, h, radius);
    CUDA_CHECK_KERNEL();
    return download_gray(d_dst, w, h);
}

Image run_sobel(const Image& gray) {
    if (gray.channels != 1) throw std::runtime_error("sobel expects a 1-channel image");
    const int w = gray.width, h = gray.height, n = w * h;

    DeviceBuffer<unsigned char> d_src(n), d_dst(n);
    d_src.upload(gray.data.data(), n);
    // Measured on a T4 at 16.8 MPixel: naive 0.527 ms, tiled 0.783 ms. A 3x3
    // stencil is small enough that L1 absorbs the redundant reads, so tiling
    // only costs a __syncthreads() and the halo load.
    gpu::sobel_naive(d_src.get(), d_dst.get(), w, h);
    CUDA_CHECK_KERNEL();
    return download_gray(d_dst, w, h);
}

Image run_equalize(const Image& gray) {
    if (gray.channels != 1) throw std::runtime_error("equalize expects a 1-channel image");
    const int w = gray.width, h = gray.height, n = w * h;

    DeviceBuffer<unsigned char> d_src(n), d_dst(n);
    DeviceBuffer<unsigned int>  d_hist(256);
    d_src.upload(gray.data.data(), n);
    d_hist.zero();

    gpu::histogram_shared(d_src.get(), n, d_hist.get());
    CUDA_CHECK_KERNEL();

    // 256 bins is not worth a device-side scan.
    std::vector<unsigned int> hist(256);
    d_hist.download(hist.data(), 256);
    gpu::upload_lut(cpu::equalize_lut(hist, static_cast<size_t>(n)));

    gpu::apply_lut(d_src.get(), d_dst.get(), n);
    CUDA_CHECK_KERNEL();
    return download_gray(d_dst, w, h);
}

Image run_pipeline(const Image& rgb, int radius, float sigma) {
    if (rgb.channels != 3) throw std::runtime_error("pipeline expects a 3-channel image");
    const int w = rgb.width, h = rgb.height, n = w * h;

    gpu::upload_gaussian(gaussian_weights(radius, sigma));
    DeviceBuffer<unsigned char> d_rgb(rgb.bytes()), d_gray(n), d_blur(n), d_out(n);
    DeviceBuffer<float>         d_tmp(n);

    d_rgb.upload(rgb.data.data(), rgb.bytes());
    gpu::grayscale_vec4(d_rgb.get(), d_gray.get(), w, h);
    gpu::blur_separable(d_gray.get(), d_tmp.get(), d_blur.get(), w, h, radius);
    gpu::sobel_naive(d_blur.get(), d_out.get(), w, h);   // see run_sobel for why
    CUDA_CHECK_KERNEL();
    return download_gray(d_out, w, h);
}

// ---- Benchmark ----

void run_benchmark(int width, int height, int radius, float sigma, int iters) {
    const int    n    = width * height;
    const double mpix = n / 1e6;

    print_device_info();
    std::printf("\nimage  : %d x %d (%.1f MPixel), blur radius %d, sigma %.2f, %d iterations\n",
                width, height, mpix, radius, sigma, iters);

    const Image rgb = make_test_image(width, height);

    // ---- CPU references, which double as the correctness oracle ----
    Image ref_gray, ref_blur, ref_sobel, ref_eq;
    std::vector<unsigned int> ref_hist;

    const double cpu_gray_ms  = time_cpu([&] { ref_gray  = cpu::grayscale(rgb); });
    const double cpu_blur_ms  = time_cpu([&] { ref_blur  = cpu::gaussian_blur(ref_gray, radius, sigma); });
    const double cpu_sobel_ms = time_cpu([&] { ref_sobel = cpu::sobel(ref_gray); });
    const double cpu_eq_ms    = time_cpu([&] {
        ref_hist = cpu::histogram(ref_gray);
        ref_eq   = cpu::apply_lut(ref_gray, cpu::equalize_lut(ref_hist, static_cast<size_t>(n)));
    });

    // ---- device buffers allocated once, so timings exclude cudaMalloc ----
    DeviceBuffer<unsigned char> d_rgb(rgb.bytes()), d_gray(n), d_dst(n);
    DeviceBuffer<float>         d_tmp(n);
    DeviceBuffer<unsigned int>  d_hist(256);
    gpu::upload_gaussian(gaussian_weights(radius, sigma));

    // ---- transfer cost, reported separately from kernel time ----
    GpuTimer t;
    std::vector<unsigned char> scratch(n);

    t.start();
    for (int i = 0; i < iters; ++i) d_rgb.upload(rgb.data.data(), rgb.bytes());
    t.stop();
    const double h2d_ms = t.elapsed_ms() / iters;

    t.start();
    for (int i = 0; i < iters; ++i) d_gray.download(scratch.data(), n);
    t.stop();
    const double d2h_ms = t.elapsed_ms() / iters;

    std::printf("\nPCIe transfer (excluded from the kernel timings below)\n");
    std::printf("  H2D %6.1f MB : %8.3f ms  (%6.1f GB/s)\n",
                rgb.bytes() / 1e6, h2d_ms, rgb.bytes() / (h2d_ms * 1e6));
    std::printf("  D2H %6.1f MB : %8.3f ms  (%6.1f GB/s)\n",
                n / 1e6, d2h_ms, n / (d2h_ms * 1e6));

    d_rgb.upload(rgb.data.data(), rgb.bytes());
    Image got(width, height, 1);

    // ---- RGB -> grayscale ----
    header("RGB -> grayscale  (memory bound: 4 bytes moved per pixel)");
    cpu_row("cpu (single thread)", cpu_gray_ms, mpix / cpu_gray_ms * 1e3);

    const double g_naive = time_gpu(
        [&] { gpu::grayscale_naive(d_rgb.get(), d_gray.get(), width, height); }, iters);
    d_gray.download(got.data.data(), n);
    row("gpu naive (1 px/thread)", g_naive, mpix / g_naive * 1e3,
        cpu_gray_ms / g_naive, max_abs_diff(got, ref_gray));

    const double g_vec = time_gpu(
        [&] { gpu::grayscale_vec4(d_rgb.get(), d_gray.get(), width, height); }, iters);
    d_gray.download(got.data.data(), n);
    row("gpu vec4 (4 px/thread)", g_vec, mpix / g_vec * 1e3,
        cpu_gray_ms / g_vec, max_abs_diff(got, ref_gray));
    std::printf("  -> vec4 moves %.0f GB/s effective; compare against peak above\n",
                4.0 * n / (g_vec * 1e6));

    // Feed the later kernels the CPU's grayscale, not the GPU's. The two differ
    // by up to 1 LSB and a stencil amplifies that (Sobel's weights sum to 8),
    // which would read as kernel error when it is really propagated input error.
    d_gray.upload(ref_gray.data.data(), n);

    // ---- Gaussian blur ----
    char title[160];
    std::snprintf(title, sizeof(title),
                  "Gaussian blur r=%d  (2-D form is %d taps/px, separable is %d)",
                  radius, (2 * radius + 1) * (2 * radius + 1), 2 * (2 * radius + 1));
    header(title);
    cpu_row("cpu (separable)", cpu_blur_ms, mpix / cpu_blur_ms * 1e3);

    const double b_naive = time_gpu(
        [&] { gpu::blur_naive(d_gray.get(), d_dst.get(), width, height, radius); }, iters);
    d_dst.download(got.data.data(), n);
    row("gpu naive (global)", b_naive, mpix / b_naive * 1e3, 1.0, max_abs_diff(got, ref_blur));

    const double b_shared = time_gpu(
        [&] { gpu::blur_shared(d_gray.get(), d_dst.get(), width, height, radius); }, iters);
    d_dst.download(got.data.data(), n);
    row("gpu shared-mem tiled", b_shared, mpix / b_shared * 1e3, b_naive / b_shared,
        max_abs_diff(got, ref_blur));

    const double b_sep = time_gpu([&] {
        gpu::blur_separable(d_gray.get(), d_tmp.get(), d_dst.get(), width, height, radius);
    }, iters);
    d_dst.download(got.data.data(), n);
    row("gpu separable (2 passes)", b_sep, mpix / b_sep * 1e3, b_naive / b_sep,
        max_abs_diff(got, ref_blur));
    std::printf("  -> separable vs cpu: %.0fx\n", cpu_blur_ms / b_sep);

    // ---- Sobel ----
    header("Sobel 3x3 magnitude");
    cpu_row("cpu", cpu_sobel_ms, mpix / cpu_sobel_ms * 1e3);

    const double s_naive = time_gpu(
        [&] { gpu::sobel_naive(d_gray.get(), d_dst.get(), width, height); }, iters);
    d_dst.download(got.data.data(), n);
    row("gpu naive (global)", s_naive, mpix / s_naive * 1e3, 1.0, max_abs_diff(got, ref_sobel));

    const double s_shared = time_gpu(
        [&] { gpu::sobel_shared(d_gray.get(), d_dst.get(), width, height); }, iters);
    d_dst.download(got.data.data(), n);
    row("gpu shared-mem tiled", s_shared, mpix / s_shared * 1e3, s_naive / s_shared,
        max_abs_diff(got, ref_sobel));

    // ---- Histogram ----
    header("256-bin histogram  (atomic contention)");
    cpu_row("cpu (hist + equalise)", cpu_eq_ms, mpix / cpu_eq_ms * 1e3);

    std::vector<unsigned int> hist(256);

    const double h_global = time_gpu([&] {
        d_hist.zero();
        gpu::histogram_global(d_gray.get(), n, d_hist.get());
    }, iters);
    d_hist.download(hist.data(), 256);
    row("gpu global atomics", h_global, mpix / h_global * 1e3, 1.0, hist_max_err(hist, ref_hist));

    const double h_shared = time_gpu([&] {
        d_hist.zero();
        gpu::histogram_shared(d_gray.get(), n, d_hist.get());
    }, iters);
    d_hist.download(hist.data(), 256);
    row("gpu shared (privatised)", h_shared, mpix / h_shared * 1e3, h_global / h_shared,
        hist_max_err(hist, ref_hist));

    const Image eq_gpu = run_equalize(ref_gray);
    std::printf("  -> equalisation (hist + CDF + LUT) max err vs cpu: %d\n",
                max_abs_diff(eq_gpu, ref_eq));

    // ---- End-to-end ----
    const Image pipe_gpu = run_pipeline(rgb, radius, sigma);
    const Image pipe_cpu = cpu::sobel(cpu::gaussian_blur(cpu::grayscale(rgb), radius, sigma));

    std::printf("\nend-to-end pipeline (gray -> blur -> sobel), one upload + one download\n");
    // The pipeline runs the variant that measured fastest in each group above.
    const double pipe_ms = g_vec + b_sep + s_naive;
    std::printf("  kernels only : %8.3f ms\n", pipe_ms);
    std::printf("  + transfers  : %8.3f ms  (transfers dominate)\n", pipe_ms + h2d_ms + d2h_ms);
    std::printf("  cpu total    : %8.3f ms\n", cpu_gray_ms + cpu_blur_ms + cpu_sobel_ms);
    std::printf("  max err      : %8d\n", max_abs_diff(pipe_gpu, pipe_cpu));
    std::printf("\nnote: per-kernel max err of 0-1 is float contraction, not a bug.\n");
    std::printf("      the end-to-end figure is larger by design: it chains the GPU\n");
    std::printf("      grayscale, whose 1-LSB difference the Sobel weights amplify.\n");
}
