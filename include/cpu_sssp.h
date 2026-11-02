// =============================================================================
//  cpu_sssp.h - CPU Baseline: Dijkstra / A* tuần tự chuẩn (std::priority_queue)
// =============================================================================
#pragma once
#include "graph.h"

// Kết quả SSSP (một nguồn tới mọi đỉnh)
struct SSSPResult {
    std::vector<u32> dist;     // dist[v], INF_DIST nếu không tới được
    std::vector<u32> parent;   // đỉnh cha trên cây đường đi ngắn nhất
    double ms = 0.0;           // thời gian thực thi (ms)
    u64 edgesScanned = 0;      // số cạnh đã xét (khối lượng công việc)
    u64 settled = 0;           // số đỉnh được "chốt"
};

// Kết quả truy vấn điểm - điểm (A*)
struct P2PResult {
    u32 dist = INF_DIST;
    std::vector<u32> path;     // src ... dst (rỗng nếu không có đường)
    double ms = 0.0;
    u64 edgesScanned = 0;
    u64 settled = 0;
};

// Dijkstra chuẩn với hàng đợi ưu tiên nhị phân (lazy deletion): O((n + m) log n)
SSSPResult cpuDijkstra(const CSRGraph& g, u32 src);

// A* với heuristic Euclid chấp nhận được (heuristic.h). useHeuristic=false => Dijkstra dừng sớm.
P2PResult cpuAStar(const CSRGraph& g, u32 src, u32 dst, bool useHeuristic = true);

// Nhiều nguồn: chạy Dijkstra độc lập cho từng nguồn, song song bằng OpenMP (threads<=0: mặc định)
struct MultiSourceCpuResult {
    std::vector<std::vector<u32>> dist;  // chỉ lưu khi keepDist=true
    double ms = 0.0;
    u64 edgesScanned = 0;
    int threads = 1;
};
MultiSourceCpuResult cpuMultiSource(const CSRGraph& g, const std::vector<u32>& srcs, int threads, bool keepDist);

// Truy vết đường đi từ mảng parent (rỗng nếu đứt chuỗi hoặc có chu trình)
std::vector<u32> reconstructPath(const std::vector<u32>& parent, u32 src, u32 dst);

int cpuMaxThreads();
