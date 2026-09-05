NVCC  ?= nvcc
# -arch=native needs CUDA >= 11.5 (Colab is fine). Override for older toolkits:
#   make ARCH=-arch=sm_75
ARCH  ?= -arch=native

NVCCFLAGS = -O3 -std=c++17 -Iinclude -lineinfo $(ARCH)

SRC = src/main.cu \
      src/pipeline.cu \
      src/image.cpp \
      src/cpu_ops.cpp \
      src/kernels/grayscale.cu \
      src/kernels/blur.cu \
      src/kernels/sobel.cu \
      src/kernels/histogram.cu

imgproc: $(SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Quick end-to-end smoke test: generate an image, run the pipeline, check sizes.
.PHONY: demo bench test clean
demo: imgproc
	./imgproc gen data/test.ppm 1920 1080
	./imgproc pipeline data/test.ppm data/edges.pgm 5
	./imgproc equalize data/test.ppm data/equalized.pgm
	@ls -l data/

bench: imgproc
	./imgproc bench 4096 4096 8 50

clean:
	rm -f imgproc host_test data/*.ppm data/*.pgm

# Host-only tests: no CUDA, no GPU. Builds with plain g++/clang++.
CXX ?= g++
test: tests/host_test.cpp src/image.cpp src/cpu_ops.cpp
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Iinclude $^ -o host_test
	./host_test
