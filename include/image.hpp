#pragma once
#include <string>
#include <vector>
#include <cstddef>

// Interleaved 8-bit image. channels is 1 (gray) or 3 (RGB).
struct Image {
    int width = 0, height = 0, channels = 0;
    std::vector<unsigned char> data;

    Image() = default;
    Image(int w, int h, int c)
        : width(w), height(h), channels(c),
          data(static_cast<size_t>(w) * static_cast<size_t>(h) * c) {}

    size_t pixels() const { return static_cast<size_t>(width) * height; }
    size_t bytes()  const { return data.size(); }
    bool   empty()  const { return data.empty(); }
};

// Binary netpbm: P6 for 3-channel, P5 for 1-channel.
Image load_image(const std::string& path);
void  save_image(const std::string& path, const Image& img);

// Synthetic RGB test pattern, so the benchmarks need no sample photo.
Image make_test_image(int w, int h);

// Largest per-byte difference between two same-sized images.
int max_abs_diff(const Image& a, const Image& b);
