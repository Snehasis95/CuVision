#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",             \
                         cudaGetErrorName(err_), __FILE__, __LINE__,           \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Syncs, so a bad access surfaces at the kernel that caused it.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// cudaEvent records on the stream, so this excludes host launch overhead.
class GpuTimer {
public:
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop()  { CUDA_CHECK(cudaEventRecord(stop_)); }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { alloc(n); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; o.n_ = 0; }

    void alloc(size_t n) {
        release();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
        n_ = n;
    }
    void release() { if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; n_ = 0; } }

    void upload(const T* host, size_t n)   { CUDA_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice)); }
    void download(T* host, size_t n) const { CUDA_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost)); }
    void zero()                            { CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T))); }

    T*       get()       { return ptr_; }
    const T* get() const { return ptr_; }
    size_t   count() const { return n_; }

private:
    T*     ptr_ = nullptr;
    size_t n_   = 0;
};
