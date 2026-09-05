#include "ops.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {

void usage() {
    std::printf(
        "gpu image processing engine (CUDA)\n"
        "\n"
        "usage:\n"
        "  imgproc info\n"
        "  imgproc gen      <out.ppm> [w=1920] [h=1080]\n"
        "  imgproc gray     <in.ppm>  <out.pgm>\n"
        "  imgproc blur     <in>      <out.pgm> [radius=5] [sigma=0]\n"
        "  imgproc sobel    <in>      <out.pgm>\n"
        "  imgproc equalize <in>      <out.pgm>\n"
        "  imgproc pipeline <in.ppm>  <out.pgm> [radius=5] [sigma=0]\n"
        "  imgproc bench    [w=4096] [h=4096] [radius=8] [iters=50]\n"
        "\n"
        "notes:\n"
        "  * images are binary netpbm: P6 for RGB (.ppm), P5 for gray (.pgm)\n"
        "  * sigma <= 0 means radius/2\n"
        "  * `gen` writes a synthetic test image so nothing needs downloading\n");
}

int arg_int(int argc, char** argv, int idx, int fallback) {
    return idx < argc ? std::atoi(argv[idx]) : fallback;
}

float arg_float(int argc, char** argv, int idx, float fallback) {
    return idx < argc ? static_cast<float>(std::atof(argv[idx])) : fallback;
}

// Most ops are single-channel; convert first if handed colour.
Image as_gray(const Image& img) {
    return img.channels == 3 ? run_grayscale(img) : img;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { usage(); return 1; }
    const std::string cmd = argv[1];

    if (cmd == "info") {
        print_device_info();
        return 0;
    }

    if (cmd == "bench") {
        const int w = arg_int(argc, argv, 2, 4096);
        const int h = arg_int(argc, argv, 3, 4096);
        const int r = arg_int(argc, argv, 4, 8);
        const int it = arg_int(argc, argv, 5, 50);
        run_benchmark(w, h, r, 0.0f, it);
        return 0;
    }

    if (cmd == "gen") {
        if (argc < 3) { usage(); return 1; }
        const int w = arg_int(argc, argv, 3, 1920);
        const int h = arg_int(argc, argv, 4, 1080);
        const Image img = make_test_image(w, h);
        save_image(argv[2], img);
        std::printf("wrote %s (%d x %d RGB)\n", argv[2], w, h);
        return 0;
    }

    if (argc < 4) { usage(); return 1; }
    const std::string in_path = argv[2], out_path = argv[3];
    const Image src = load_image(in_path);
    std::printf("loaded %s (%d x %d, %d channel%s)\n",
                in_path.c_str(), src.width, src.height, src.channels,
                src.channels == 1 ? "" : "s");

    Image out;
    if (cmd == "gray") {
        if (src.channels != 3) { std::fprintf(stderr, "gray needs a 3-channel .ppm\n"); return 1; }
        out = run_grayscale(src);
    } else if (cmd == "blur") {
        out = run_blur(as_gray(src), arg_int(argc, argv, 4, 5), arg_float(argc, argv, 5, 0.0f));
    } else if (cmd == "sobel") {
        out = run_sobel(as_gray(src));
    } else if (cmd == "equalize") {
        out = run_equalize(as_gray(src));
    } else if (cmd == "pipeline") {
        if (src.channels != 3) { std::fprintf(stderr, "pipeline needs a 3-channel .ppm\n"); return 1; }
        out = run_pipeline(src, arg_int(argc, argv, 4, 5), arg_float(argc, argv, 5, 0.0f));
    } else {
        std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
        usage();
        return 1;
    }

    save_image(out_path, out);
    std::printf("wrote %s (%d x %d, %d channel%s)\n",
                out_path.c_str(), out.width, out.height, out.channels,
                out.channels == 1 ? "" : "s");
    return 0;

} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
