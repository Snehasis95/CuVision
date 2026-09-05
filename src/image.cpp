#include "image.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

// Netpbm allows whitespace and '#' comments between header tokens.
std::string read_token(std::istream& in) {
    std::string tok;
    char c;
    while (in.get(c)) {
        if (c == '#') {                       // comment runs to end of line
            while (in.get(c) && c != '\n') {}
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!tok.empty()) return tok;
            continue;
        }
        tok.push_back(c);
    }
    return tok;
}

int read_int(std::istream& in, const std::string& path) {
    const std::string tok = read_token(in);
    if (tok.empty()) throw std::runtime_error("truncated header in " + path);
    return std::atoi(tok.c_str());
}

unsigned char clamp_u8(float v) {
    if (v <= 0.0f)   return 0;
    if (v >= 255.0f) return 255;
    return static_cast<unsigned char>(v + 0.5f);
}

}  // namespace

Image load_image(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    const std::string magic = read_token(in);
    int channels;
    if      (magic == "P6") channels = 3;
    else if (magic == "P5") channels = 1;
    else throw std::runtime_error("unsupported format '" + magic + "' in " + path +
                                  " (expected binary P5/P6)");

    const int w      = read_int(in, path);
    const int h      = read_int(in, path);
    const int maxval = read_int(in, path);
    if (w <= 0 || h <= 0) throw std::runtime_error("bad dimensions in " + path);
    if (maxval != 255)    throw std::runtime_error("only 8-bit (maxval 255) supported in " + path);
    // read_token already ate the separating whitespace byte.

    Image img(w, h, channels);
    in.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.bytes()));
    if (static_cast<size_t>(in.gcount()) != img.bytes())
        throw std::runtime_error("truncated pixel data in " + path);
    return img;
}

void save_image(const std::string& path, const Image& img) {
    if (img.channels != 1 && img.channels != 3)
        throw std::runtime_error("can only write 1- or 3-channel images");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);

    out << (img.channels == 3 ? "P6" : "P5") << "\n"
        << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              static_cast<std::streamsize>(img.bytes()));
    if (!out) throw std::runtime_error("write failed for " + path);
}

Image make_test_image(int w, int h) {
    Image img(w, h, 3);
    const float cx = w * 0.5f, cy = h * 0.5f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = x - cx, dy = y - cy;
            const float radius = std::sqrt(dx * dx + dy * dy);

            // Rings for the edge detector to find, ramp for a smooth region.
            const float rings = 127.0f * (0.5f + 0.5f * std::sin(radius * 0.15f));
            const float ramp  = 255.0f * (static_cast<float>(x + y) / (w + h));

            // Deterministic, so the pattern reproduces across machines.
            const unsigned int hsh = static_cast<unsigned int>(x * 73856093u ^ y * 19349663u);
            const float noise = static_cast<float>(hsh & 31u) - 16.0f;

            const size_t i = (static_cast<size_t>(y) * w + x) * 3;
            img.data[i + 0] = clamp_u8(rings + noise);
            img.data[i + 1] = clamp_u8(ramp * 0.7f + noise);
            img.data[i + 2] = clamp_u8(255.0f - ramp * 0.5f + rings * 0.3f + noise);
        }
    }
    return img;
}

int max_abs_diff(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels)
        throw std::runtime_error("max_abs_diff: image shapes differ");

    int worst = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const int d = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (d > worst) worst = d;
    }
    return worst;
}
