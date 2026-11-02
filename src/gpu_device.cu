// =============================================================================
//  gpu_device.cu - Thông tin GPU, khởi động (warm-up), thí nghiệm truyền dữ liệu
//  Host -> Device: bộ nhớ thường (pageable) vs bộ nhớ ghim (pinned) vs pinned + đa stream
// =============================================================================
#include "gpu_sssp.h"
#include "cuda_check.cuh"

#include <algorithm>
#include <cstring>

const char* toString(GpuAlgo a) {
    switch (a) {
        case GpuAlgo::NaiveRelax: return "Naive-Relax";
        case GpuAlgo::Frontier:   return "Frontier";
        case GpuAlgo::DeltaStep:  return "Delta-Step";
    }
    return "?";
}
const char* toString(Mapping m) {
    switch (m) {
        case Mapping::Thread: return "thread";
        case Mapping::Warp:   return "warp";
        case Mapping::Hybrid: return "hybrid";
    }
    return "?";
}
const char* toString(AtomicMode a) {
    switch (a) {
        case AtomicMode::MinRBA:     return "atomicMin+RBA";
        case AtomicMode::CasRBA:     return "atomicCAS+RBA";
        case AtomicMode::MinNoCheck: return "atomicMin(no-check)";
    }
    return "?";
}

GpuDeviceInfo gpuQueryDevice(int dev) {
    GpuDeviceInfo info;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= dev) {
        cudaGetLastError();
        return info;
    }
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    info.ok = true;
    info.name = p.name;
    info.ccMajor = p.major;
    info.ccMinor = p.minor;
    info.smCount = p.multiProcessorCount;
    info.globalMem = p.totalGlobalMem;
    info.sharedPerBlock = p.sharedMemPerBlock;
    info.l2Bytes = (size_t)p.l2CacheSize;
    info.maxThreadsPerSM = p.maxThreadsPerMultiProcessor;
    info.warpSize = p.warpSize;
    // clockRate / memoryClockRate đã bị loại khỏi cudaDeviceProp ở CUDA 13 -> dùng thuộc tính
    int clk = 0, memClk = 0, bus = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memClk, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaGetLastError();
    info.clockMHz = clk / 1000;
    info.memClockMHz = memClk / 1000;
    info.busWidth = bus;
    info.peakBandwidthGBs = 2.0 * (double)memClk * 1e3 * (bus / 8.0) / 1e9;   // DDR
    cudaDriverGetVersion(&info.driverVersion);
    cudaRuntimeGetVersion(&info.runtimeVersion);
    return info;
}

void gpuWarmup() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));                  // khởi tạo CUDA context (tốn ~100ms lần đầu)
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// 2.4.4: So sánh 3 cách nạp đồ thị CSR lên VRAM
// -----------------------------------------------------------------------------
TransferStudy gpuTransferStudy(const CSRGraph& g, int reps) {
    TransferStudy r;
    const size_t bRow = (size_t)(g.n + 1) * 4, bCol = (size_t)g.m * 4, bW = (size_t)g.m * 4;
    r.bytes = (double)(bRow + bCol + bW);
    u32 *dRow, *dCol, *dW;
    CUDA_CHECK(cudaMalloc(&dRow, bRow));
    CUDA_CHECK(cudaMalloc(&dCol, bCol));
    CUDA_CHECK(cudaMalloc(&dW, bW));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float ms = 0.f;

    // (a) Pageable: driver phải chép qua bộ đệm ghim trung gian -> chậm hơn
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, 0));
        CUDA_CHECK(cudaMemcpy(dRow, g.row_offsets.data(), bRow, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCol, g.col_indices.data(), bCol, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW, g.weights.data(), bW, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1, 0));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        r.pageableMs += ms / reps;
    }

    // (b) Pinned (page-locked): DMA trực tiếp, cho phép cudaMemcpyAsync thực sự bất đồng bộ
    u32 *hRow, *hCol, *hW;
    CUDA_CHECK(cudaMallocHost(&hRow, bRow));
    CUDA_CHECK(cudaMallocHost(&hCol, bCol));
    CUDA_CHECK(cudaMallocHost(&hW, bW));
    memcpy(hRow, g.row_offsets.data(), bRow);
    memcpy(hCol, g.col_indices.data(), bCol);
    memcpy(hW, g.weights.data(), bW);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, 0));
        CUDA_CHECK(cudaMemcpyAsync(dRow, hRow, bRow, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(dCol, hCol, bCol, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(dW, hW, bW, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaEventRecord(e1, 0));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        r.pinnedMs += ms / reps;
    }

    // (c) Pinned + 3 CUDA Streams: chia khối (chunk) xen kẽ giữa các stream
    cudaStream_t st[3];
    for (auto& s : st) CUDA_CHECK(cudaStreamCreate(&s));
    const size_t CH = 8u << 20;   // 8 MB / chunk
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, 0));
        int sid = 0;
        auto chunked = [&](void* dst, const void* src, size_t bytes) {
            for (size_t off = 0; off < bytes; off += CH) {
                size_t len = std::min(CH, bytes - off);
                CUDA_CHECK(cudaMemcpyAsync((char*)dst + off, (const char*)src + off, len,
                                           cudaMemcpyHostToDevice, st[sid]));
                sid = (sid + 1) % 3;
            }
        };
        chunked(dRow, hRow, bRow);
        chunked(dCol, hCol, bCol);
        chunked(dW, hW, bW);
        // Stream mặc định (legacy) chờ mọi stream "blocking" -> e1 đo được toàn bộ
        CUDA_CHECK(cudaEventRecord(e1, 0));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        r.pinnedStreamsMs += ms / reps;
    }
    for (auto& s : st) CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFreeHost(hRow));
    CUDA_CHECK(cudaFreeHost(hCol));
    CUDA_CHECK(cudaFreeHost(hW));
    CUDA_CHECK(cudaFree(dRow));
    CUDA_CHECK(cudaFree(dCol));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return r;
}
