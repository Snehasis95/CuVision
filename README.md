# CuVision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X1xrtIFfLz6XMWF-eg_I4g-DyRC9HYmo#scrollTo=u1HEejI1B74P)

A CUDA image-processing engine built to make the cost of a memory-access
pattern visible. Every operation ships with a single-threaded CPU reference and
two or three GPU kernels that compute exactly the same result by different
means — naive global loads, shared-memory tiling, vectorised loads, separable
passes, privatised atomics. `imgproc bench` runs them side by side and prints
throughput next to the maximum per-pixel error against the CPU oracle, so a
"faster" kernel that quietly changes its output cannot hide.

No third-party dependencies, no sample images to download. Images are binary
netpbm (P6 for RGB, P5 for grayscale) and the test pattern is generated.

No NVIDIA card handy? The badge above opens
[`notebooks/CuVision.ipynb`](notebooks/CuVision.ipynb) on a free Colab GPU — it
clones, builds, runs the tests, renders each operation's output, and walks
through reading the benchmark table.

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

Needs a CUDA toolkit and an NVIDIA GPU. To skip this entirely, run the
[Colab notebook](https://colab.research.google.com/github/Snehasis95/CuVision/blob/main/notebooks/CuVision.ipynb).

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
separately.

Each row carries a `max err` column: the largest absolute per-byte difference
against the CPU implementation. Values of 0–1 are float contraction, not bugs.
The end-to-end figure is larger by design, because it chains the *GPU*
grayscale and Sobel's weights (summing to 8) amplify a 1-LSB input difference —
which is why the per-kernel comparisons feed the CPU's grayscale to the later
stages instead.

### Measured: Tesla T4 (sm_75), 4096×4096, r=8, 50 iterations

Full run in [`results/t4-4096.txt`](results/t4-4096.txt). Peak bandwidth
320 GB/s; the `% peak` column is minimum DRAM traffic over measured time.

| Op | Variant | ms | vs CPU | % peak | max err |
|---|---|---:|---:|---:|---:|
| grayscale | cpu | 61.040 | — | — | — |
| | gpu naive | 0.490 | 125× | 43% | 1 |
| | **gpu vec4** | **0.251** | **243×** | **84%** | 1 |
| blur r=8 | cpu | 897.994 | — | — | — |
| | gpu naive | 14.987 | 60× | — | 1 |
| | gpu shared tiled | 11.291 | 80× | — | 1 |
| | **gpu separable** | **1.844** | **487×** | 28% | 1 |
| sobel | cpu | 178.356 | — | — | — |
| | **gpu naive** | **0.503** | **355×** | 21% | 0 |
| | gpu shared tiled | 0.747 | 239× | 14% | 0 |
| histogram | cpu | 36.893 | — | — | — |
| | gpu global atomics | 4.619 | 8× | 1% | 0 |
| | **gpu shared privatised** | **0.183** | **202×** | 29% | 0 |

### What the numbers say

**Coalescing is worth 1.95×, with no change in arithmetic.** `naive` and `vec4`
compute identical luma; the only difference is that consecutive threads read at
a 3-byte stride versus three aligned 32-bit loads. Naive reaches 137 GB/s (43%
of peak), vec4 reaches 267 GB/s (84%) — and 84/43 ≈ 1.95, which is exactly the
measured speedup. The access pattern *is* the whole result.

**The separable blur wins by algorithm, not by memory.** 289 taps/px against
34 predicts 8.5×; measured is 8.13×. That near-proportionality says the 2-D
kernels are bound by load-issue throughput, not DRAM — which is also why
shared-memory tiling bought only 1.33×. Separable still sits at just 28% of
peak, so the float intermediate (4 bytes/px, written and
read back) is the next thing to attack.

**Tiling loses on Sobel — and the bandwidth column explains why.** Naive moves
its minimum 33.6 MB in 0.503 ms, or 67 GB/s: only 21% of peak, so DRAM was
never the bottleneck. What bounds it is nine load instructions per output
pixel. Shared-memory tiling reduces *DRAM traffic* — which was already free —
while keeping the same nine loads, and adds a halo fetch plus a
`__syncthreads()`. It is 1.49× slower as a result. `run_sobel` and
`run_pipeline` use the naive variant for that reason.

**Privatisation is the single biggest kernel win: 25.3×.** Every pixel of a
16.8 MPixel image contends for 256 global counters in the naive version. Giving
each block its own shared copy caps global traffic at 256 atomics per block.
At 0.183 ms it is still only 29% of peak, so multi-copy privatisation (several
sub-histograms per block) has room left.

**Transfers dominate everything.** 14.0 ms of the 16.6 ms end-to-end is PCIe,
against 2.6 ms of kernels. And at 4.8 GB/s H2D the transfers themselves are
leaving ~2.5× on the table: that is pageable-memory throughput, roughly a third
of what PCIe 3.0 ×16 sustains with pinned buffers. Pinned memory and
compute/transfer overlap are worth more here than any remaining kernel tuning.

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
results/t4-4096.txt        captured benchmark run (Tesla T4)
notebooks/CuVision.ipynb   end-to-end walkthrough on a Colab GPU
```
