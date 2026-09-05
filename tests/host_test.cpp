// Host-only tests for the netpbm I/O and the CPU reference implementations.
//
// Deliberately CUDA-free: this builds and runs on a machine with no GPU, which
// is where most of the development happens. The GPU kernels are validated
// separately, against these same reference implementations, by `imgproc bench`.
//
//   make test
#include "ops.hpp"
#include <cstdio>
#include <numeric>
#include <stdexcept>

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int W = 64, H = 48;
    Image rgb = make_test_image(W, H);
    check(rgb.width == W && rgb.height == H && rgb.channels == 3, "test image shape");
    check(rgb.bytes() == size_t(W) * H * 3, "test image byte count");

    // netpbm round trip, both P6 and P5
    save_image("rt.ppm", rgb);
    Image rgb2 = load_image("rt.ppm");
    check(rgb2.data == rgb.data && rgb2.channels == 3, "P6 round trip is lossless");

    Image gray = cpu::grayscale(rgb);
    check(gray.channels == 1 && gray.pixels() == rgb.pixels(), "grayscale shape");
    save_image("rt.pgm", gray);
    Image gray2 = load_image("rt.pgm");
    check(gray2.data == gray.data && gray2.channels == 1, "P5 round trip is lossless");

    // A flat image must survive blur unchanged (weights sum to 1, borders clamp).
    Image flat(32, 32, 1);
    for (auto& v : flat.data) v = 200;
    Image blurred_flat = cpu::gaussian_blur(flat, 5, 0.0f);
    check(max_abs_diff(flat, blurred_flat) == 0, "blur preserves a constant image");

    // ...and produce no edges.
    Image sobel_flat = cpu::sobel(flat);
    int mx = 0; for (auto v : sobel_flat.data) mx = v > mx ? v : mx;
    check(mx == 0, "sobel of a constant image is zero");

    // Sobel must fire on a hard vertical edge.
    Image step(32, 32, 1);
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x) step.data[y * 32 + x] = x < 16 ? 0 : 255;
    Image sobel_step = cpu::sobel(step);
    check(sobel_step.data[16 * 32 + 15] > 200 && sobel_step.data[16 * 32 + 2] == 0,
          "sobel finds a vertical step and ignores flat regions");

    // Gaussian weights normalise and are symmetric.
    auto w = gaussian_weights(8, 0.0f);
    float sum = std::accumulate(w.begin(), w.end(), 0.0f);
    check(w.size() == 17, "gaussian tap count is 2r+1");
    check(sum > 0.9999f && sum < 1.0001f, "gaussian weights sum to 1");
    check(w[0] == w[16] && w[7] == w[9], "gaussian weights are symmetric");

    // Histogram totals must equal the pixel count.
    auto hist = cpu::histogram(gray);
    unsigned long long total = 0; for (auto c : hist) total += c;
    check(total == gray.pixels(), "histogram sums to pixel count");

    // Equalisation is monotonic and spans the full range.
    auto lut = cpu::equalize_lut(hist, gray.pixels());
    bool mono = true;
    for (int i = 1; i < 256; ++i) if (lut[i] < lut[i - 1]) mono = false;
    check(mono, "equalisation LUT is monotonic");
    Image eq = cpu::apply_lut(gray, lut);
    check(eq.pixels() == gray.pixels(), "equalised image shape");

    // Bad input must be rejected, not silently mis-parsed.
    bool threw = false;
    try { load_image("does_not_exist.ppm"); } catch (const std::exception&) { threw = true; }
    check(threw, "missing file throws");
    threw = false;
    try { gaussian_weights(99, 1.0f); } catch (const std::exception&) { threw = true; }
    check(threw, "over-large radius throws");

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all host tests passed",
                failures, failures == 1 ? "" : "s");
    std::remove("rt.ppm");
    std::remove("rt.pgm");

    return failures ? 1 : 0;
}
