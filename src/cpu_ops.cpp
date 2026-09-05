#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

unsigned char clamp_u8(float v) {
    if (v <= 0.0f)   return 0;
    if (v >= 255.0f) return 255;
    return static_cast<unsigned char>(v + 0.5f);
}

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

std::vector<float> gaussian_weights(int radius, float sigma) {
    if (radius < 0 || radius > kMaxRadius)
        throw std::runtime_error("radius must be in [0, " + std::to_string(kMaxRadius) + "]");
    if (sigma <= 0.0f) sigma = std::max(0.5f, radius / 2.0f);

    std::vector<float> w(2 * radius + 1);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        w[i + radius] = std::exp(-(i * i) / (2.0f * sigma * sigma));
        sum += w[i + radius];
    }
    for (float& v : w) v /= sum;
    return w;
}

namespace cpu {

Image grayscale(const Image& rgb) {
    if (rgb.channels != 3) throw std::runtime_error("grayscale expects a 3-channel image");
    Image out(rgb.width, rgb.height, 1);
    for (size_t p = 0; p < rgb.pixels(); ++p) {
        const float r = rgb.data[p * 3 + 0];
        const float g = rgb.data[p * 3 + 1];
        const float b = rgb.data[p * 3 + 2];
        out.data[p] = clamp_u8(0.299f * r + 0.587f * g + 0.114f * b);  // BT.601 luma
    }
    return out;
}

// Same two-pass order as the GPU version, so the two agree to rounding.
Image gaussian_blur(const Image& gray, int radius, float sigma) {
    if (gray.channels != 1) throw std::runtime_error("blur expects a 1-channel image");
    const std::vector<float> w = gaussian_weights(radius, sigma);
    const int W = gray.width, H = gray.height;

    std::vector<float> tmp(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k)
                acc += w[k + radius] * gray.data[static_cast<size_t>(y) * W + clampi(x + k, 0, W - 1)];
            tmp[static_cast<size_t>(y) * W + x] = acc;
        }

    Image out(W, H, 1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k)
                acc += w[k + radius] * tmp[static_cast<size_t>(clampi(y + k, 0, H - 1)) * W + x];
            out.data[static_cast<size_t>(y) * W + x] = clamp_u8(acc);
        }
    return out;
}

Image sobel(const Image& gray) {
    if (gray.channels != 1) throw std::runtime_error("sobel expects a 1-channel image");
    const int W = gray.width, H = gray.height;
    Image out(W, H, 1);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float p[3][3];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    p[dy + 1][dx + 1] = gray.data[static_cast<size_t>(clampi(y + dy, 0, H - 1)) * W
                                                  + clampi(x + dx, 0, W - 1)];

            const float gx = -p[0][0] - 2.0f * p[1][0] - p[2][0]
                             + p[0][2] + 2.0f * p[1][2] + p[2][2];
            const float gy = -p[0][0] - 2.0f * p[0][1] - p[0][2]
                             + p[2][0] + 2.0f * p[2][1] + p[2][2];
            out.data[static_cast<size_t>(y) * W + x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    return out;
}

std::vector<unsigned int> histogram(const Image& gray) {
    std::vector<unsigned int> h(256, 0);
    for (unsigned char v : gray.data) ++h[v];
    return h;
}

std::vector<unsigned char> equalize_lut(const std::vector<unsigned int>& hist, size_t total) {
    std::vector<unsigned char> lut(256, 0);
    if (total == 0) return lut;

    // cdf_min subtracted so the darkest occupied bin maps to 0.
    unsigned long long cdf = 0, cdf_min = 0;
    for (unsigned int c : hist) { if (c) { cdf_min = c; break; } }

    const double denom = static_cast<double>(total) - static_cast<double>(cdf_min);
    for (int i = 0; i < 256; ++i) {
        cdf += hist[i];
        const double v = denom > 0.0
            ? (static_cast<double>(cdf) - static_cast<double>(cdf_min)) / denom * 255.0
            : 0.0;
        lut[i] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, v)) + 0.5);
    }
    return lut;
}

Image apply_lut(const Image& gray, const std::vector<unsigned char>& lut) {
    Image out(gray.width, gray.height, gray.channels);
    for (size_t i = 0; i < gray.data.size(); ++i) out.data[i] = lut[gray.data[i]];
    return out;
}

}  // namespace cpu
