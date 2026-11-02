// =============================================================================
//  gpu_sssp.h - Giao diện Host cho các bộ giải đường đi ngắn nhất trên GPU
//  (không lộ kiểu CUDA ra ngoài -> main.cpp biên dịch bằng trình biên dịch C++ thường)
// =============================================================================
#pragma once
#include "graph.h"
#include <string>
#include <vector>

// ---- Thuật toán (Step 2 & Step 3) ----
enum class GpuAlgo {
    NaiveRelax,   // Kernel 1: Bellman-Ford đồng bộ mảng khoảng cách (thread/đỉnh, cờ active + d_changed)
    Frontier,     // Kernel 2a: Duyệt biên sóng F_curr -> F_next (data-driven Bellman-Ford-Moore)
    DeltaStep     // Kernel 2b: Delta-Stepping (bucket, light/heavy edges)
};

// ---- Ánh xạ luồng (2.2.3 / 2.4.3) ----
enum class Mapping {
    Thread,       // 1 luồng / đỉnh biên
    Warp,         // 1 warp (32 luồng) / đỉnh biên
    Hybrid        // Cân bằng tải động 3 tầng: CTA (shared memory) / Warp / Thread theo bậc đỉnh
};

// ---- Chiến lược cập nhật nguyên tử (Step 4) ----
enum class AtomicMode {
    MinRBA,       // Read-Before-Atomic + atomicMin 64-bit trên cặp (dist<<32 | parent)  [mặc định]
    CasRBA,       // Read-Before-Atomic + vòng lặp atomicCAS 64-bit
    MinNoCheck    // atomicMin "mù" (không đọc trước) - dùng để đo mức tranh chấp
};

const char* toString(GpuAlgo a);
const char* toString(Mapping m);
const char* toString(AtomicMode a);

struct GpuOptions {
    GpuAlgo algo = GpuAlgo::DeltaStep;
    Mapping mapping = Mapping::Hybrid;
    AtomicMode atomic = AtomicMode::MinRBA;
    u32 delta = 0;          // 0 => tự chọn: Delta = 32 * avgW / avgDeg  (Davidson et al. 2014)
    int blockSize = 256;
    // Số kernel nới lỏng phát liên tiếp giữa 2 lần đồng bộ Host-Device (Frontier / Delta).
    // 1 = đồng bộ sau mỗi vòng (cách truyền thống). >1: kernel tự đọc kích thước biên trên GPU.
    u32 syncEvery = 8;
};

// Bộ đếm thống kê thu từ kernel (gom theo warp rồi mới atomicAdd -> chi phí không đáng kể)
struct GpuStats {
    u64 edgesScanned = 0;       // số cạnh đã xét (khối lượng công việc)
    u64 atomicsIssued = 0;      // số lệnh atomic cập nhật khoảng cách đã phát ra
    u64 updates = 0;            // số lần khoảng cách thực sự giảm
    u64 pushes = 0;             // số lần đưa đỉnh vào hàng đợi biên
    u32 iterations = 0;         // số vòng nới lỏng có việc (biên khác rỗng)
    u32 hostSyncs = 0;          // số lần Host phải chờ GPU (cudaStreamSynchronize) trong vòng lặp
    u32 launches = 0;           // số kernel nới lỏng đã phát (kể cả kernel rỗng khi chạy theo lô)
    u32 buckets = 0;            // số bucket đã xử lý (Delta-Stepping)
    u32 delta = 0;              // Delta thực tế đã dùng
};

struct GpuRunResult {
    double ms = 0.0;            // thời gian GPU (cudaEvent), gồm khởi tạo + vòng lặp + đồng bộ host
    GpuStats stats;
    std::vector<u32> dist;      // SSSP: toàn bộ mảng; A*: rỗng
    std::vector<u32> parent;
    u32 targetDist = INF_DIST;  // A*
    std::vector<u32> path;      // A*: đường đi src -> dst
};

struct GpuDeviceInfo {
    bool ok = false;
    std::string name;
    int ccMajor = 0, ccMinor = 0, smCount = 0, clockMHz = 0, memClockMHz = 0, busWidth = 0;
    int driverVersion = 0, runtimeVersion = 0, maxThreadsPerSM = 0, warpSize = 32;
    size_t globalMem = 0, sharedPerBlock = 0, l2Bytes = 0;
    double peakBandwidthGBs = 0.0;
};
GpuDeviceInfo gpuQueryDevice(int dev = 0);
void gpuWarmup();

// Thí nghiệm truyền dữ liệu Host -> Device (2.4.4)
struct TransferStudy {
    double bytes = 0;
    double pageableMs = 0, pinnedMs = 0, pinnedStreamsMs = 0;
};
TransferStudy gpuTransferStudy(const CSRGraph& g, int reps);

// -----------------------------------------------------------------------------
// GpuSolver: giữ đồ thị CSR trên VRAM + toàn bộ bộ đệm làm việc (cấp phát 1 lần)
// -----------------------------------------------------------------------------
class GpuSolver {
public:
    // maxBatch: số nguồn tối đa xử lý đồng thời ở chế độ Multi-Source Batching
    GpuSolver(const CSRGraph& g, u32 maxBatch = 1);
    ~GpuSolver();
    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    double uploadMs() const;          // thời gian nạp đồ thị Host -> Device (pinned + 3 streams)
    size_t deviceBytes() const;       // tổng bộ nhớ VRAM đã cấp phát
    u32 autoDelta() const;

    // SSSP một nguồn (Dijkstra song song)
    GpuRunResult sssp(u32 src, const GpuOptions& opt, bool copyBack = true);
    // Điểm - điểm A* song song (cắt tỉa theo cận trên dist[target] + heuristic)
    GpuRunResult astar(u32 src, u32 dst, const GpuOptions& opt);
    // Đa nguồn: K nguồn trong 1 biên chung (phần tử = k*n + v), thuật toán Frontier
    GpuRunResult multiSource(const std::vector<u32>& srcs, const GpuOptions& opt,
                             std::vector<std::vector<u32>>* distOut);

private:
    struct Impl;
    Impl* p_;
};
