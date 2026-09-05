cmake_minimum_required(VERSION 3.18)
project(imgproc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Turing through Ada. Volta (70) is deliberately absent: CUDA 13 dropped it, and
# listing it makes the build fail outright on a current toolkit. Override for a
# specific card with:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

add_executable(imgproc
  src/main.cu
  src/pipeline.cu
  src/image.cpp
  src/cpu_ops.cpp
  src/kernels/grayscale.cu
  src/kernels/blur.cu
  src/kernels/sobel.cu
  src/kernels/histogram.cu
)

target_include_directories(imgproc PRIVATE include)

# -lineinfo lets Nsight Compute map stalls back to source lines. Fast math is
# deliberately off so the GPU results stay comparable with the CPU reference.
target_compile_options(imgproc PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
