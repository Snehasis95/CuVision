# CuVision

A CUDA image-processing engine built to make the cost of a memory-access
pattern visible. Every operation ships with a single-threaded CPU reference and
two or three GPU kernels that compute exactly the same result by different
means — naive global loads, shared-memory tiling, vectorised loads, separable
passes, privatised atomics. `imgproc bench` runs them side by side and prints
throughput next to the maximum per-pixel error against the CPU oracle, so a
"faster" kernel that quietly changes its output cannot hide.

No third-party dependencies, no sample images to download. Images are binary
netpbm (P6 for RGB, P5 for grayscale) and the test pattern is generated.

## Operations

| Op | Kernel variants |
|---|---|
| RGB → grayscale | `naive` (1 px/thread, 3-byte stride) · `vec4` (4 px/thread, three 32-bit loads) |
| Gaussian blur | `naive` ((2r+1)² global taps) · `shared` (tile + halo) · `separable` (two 1-D passes, float intermediate) |
| Sobel 3×3 magnitude | `naive` · `shared` (tile + 1-px halo) |
| 256-bin histogram | `global` (atomics to global memory) · `shared` (per-block privatised bins) |
| Histogram equalisation | shared histogram → CDF on the host → 256-entry LUT in `__constant__` memory |

Gaussian taps and the equalisation LUT live in `__constant__` memory: a warp
reads the same entry at once, which broadcasts in a single cycle.

## Build

Needs a CUDA toolkit and an NVIDIA GPU.

```sh
make                      # -arch=native, needs CUDA >= 11.5
make ARCH=-arch=sm_75     # older toolkits
```

Or with CMake (defaults to `CMAKE_CUDA_ARCHITECTURES=75;80;86;89` — Turing
through Ada; Volta is omitted because CUDA 13 dropped it):

```sh
cmake -B build && cmake --build build -j
```

`-lineinfo` is on in both builds so Nsight Compute can map stalls back to
source lines. Fast math is deliberately off, so GPU results stay comparable
with the CPU reference.

## Usage

```
imgproc info                                       # device, SM count, peak bandwidth
imgproc gen      <out.ppm> [w=1920] [h=1080]       # synthetic RGB test image
imgproc gray     <in.ppm>  <out.pgm>
imgproc blur     <in>      <out.pgm> [radius=5] [sigma=0]
imgproc sobel    <in>      <out.pgm>
imgproc equalize <in>      <out.pgm>
imgproc pipeline <in.ppm>  <out.pgm> [radius=5] [sigma=0]
imgproc bench    [w=4096] [h=4096] [radius=8] [iters=50]
```

`sigma <= 0` means `radius/2`. Radius is capped at 15 (`kMaxRadius`), which
bounds the `__constant__` weight array and the shared-memory tile. The
single-op commands convert colour input to grayscale first where needed.

```sh
mkdir -p data
make demo     # gen -> pipeline -> equalize
make bench    # 4096x4096, radius 8, 50 iterations
```

The `pipeline` command chains grayscale → blur → Sobel with one upload and one
download; intermediates never leave the device.

## Benchmark

`imgproc bench` allocates device buffers once, warms up, then averages `iters`
launches inside a single `cudaEvent` pair, so the numbers exclude `cudaMalloc`
and host launch overhead. PCIe transfer cost is measured and reported
separately — for a single pass over an image it dominates every kernel in the
table.

Each row carries a `max err` column: the largest absolute per-byte difference
against the CPU implementation. Values of 0–1 are float contraction, not bugs.
The end-to-end figure is larger by design, because it chains the *GPU*
grayscale and Sobel's weights (summing to 8) amplify a 1-LSB input difference —
which is why the per-kernel comparisons feed the CPU's grayscale to the later
stages instead.

Not every tiled kernel wins. On a T4 at 16.8 MPixel, the naive Sobel measured
0.527 ms against 0.783 ms tiled: a 3×3 stencil is small enough that L1 absorbs
the redundant reads, so tiling only buys a `__syncthreads()` and a halo load.
`run_sobel` and `run_pipeline` use the naive variant for that reason.

## Tests

```sh
make test
```

Host-only — no CUDA, no GPU, builds with plain `g++`/`clang++`. Covers netpbm
round-tripping, Gaussian weight normalisation and symmetry, blur preserving a
constant image, Sobel firing on a step edge and not on flat regions, histogram
totals, LUT monotonicity, and rejection of bad input. The GPU kernels are
validated against these same reference implementations by `imgproc bench`.

## Layout

```
include/
  image.hpp         Image struct, netpbm I/O, test pattern, max_abs_diff
  ops.hpp           CPU reference + GPU launcher declarations
  cuda_utils.cuh    CUDA_CHECK, GpuTimer, RAII DeviceBuffer
  device_math.cuh   shared device-side helpers
src/
  main.cu           CLI
  pipeline.cu       whole-image wrappers, device info, benchmark harness
  image.cpp         netpbm parse/write
  cpu_ops.cpp       reference implementations (the correctness oracle)
  kernels/          grayscale.cu blur.cu sobel.cu histogram.cu
tests/host_test.cpp
```
