// Mô phỏng CUDA Runtime API tối thiểu cho chế độ kiểm thử trên CPU
#pragma once
#include "pfshim_core.h"
#include <chrono>
#include <cstdlib>
#include <cstring>

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum cudaDeviceAttr { cudaDevAttrClockRate, cudaDevAttrMemoryClockRate, cudaDevAttrGlobalMemoryBusWidth };
struct CUstream_st {};
struct CUevent_st { std::chrono::steady_clock::time_point t; };
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;
struct cudaDeviceProp {
    char name[256]; int major, minor, multiProcessorCount, maxThreadsPerMultiProcessor, warpSize, l2CacheSize;
    size_t totalGlobalMem, sharedMemPerBlock;
};
inline const char* cudaGetErrorString(cudaError_t e) { return e ? "shim error" : "no error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* c) { *c = 1; return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    memset(p, 0, sizeof(*p));
    strcpy(p->name, "CPU-SHIM (emulated, NOT a GPU)");
    p->major = 0; p->minor = 0; p->multiProcessorCount = 1; p->maxThreadsPerMultiProcessor = 1024; p->warpSize = 32;
    p->totalGlobalMem = 8ull << 30; p->sharedMemPerBlock = 48 << 10; p->l2CacheSize = 0;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) { *v = 0; return cudaSuccess; }
inline cudaError_t cudaDriverGetVersion(int* v) { *v = 0; return cudaSuccess; }
inline cudaError_t cudaRuntimeGetVersion(int* v) { *v = 0; return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { *f = 8ull << 30; *t = 8ull << 30; return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, size_t s) {
    *p = malloc(s ? s : 1);
    if (*p) memset(*p, 0xCD, s);          // rác cố ý: phát hiện đọc bộ nhớ chưa khởi tạo
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t s) { return cudaMalloc((void**)p, s); }
inline cudaError_t cudaMallocHost(void** p, size_t s) { return cudaMalloc(p, s); }
template <class T> inline cudaError_t cudaMallocHost(T** p, size_t s) { return cudaMalloc((void**)p, s); }
inline cudaError_t cudaFree(void* p) { free(p); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) { memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t = nullptr) { memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = new CUstream_st; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { delete s; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new CUevent_st; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr) { e->t = std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count(); return cudaSuccess;
}
