// =============================================================================
//  cpu_sssp.cpp - CPU Baseline (đơn luồng) + đa nguồn OpenMP
// =============================================================================
#include "cpu_sssp.h"
#include "heuristic.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

int cpuMaxThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// -----------------------------------------------------------------------------
// Dijkstra: khóa ưu tiên đóng gói 64-bit (dist << 32 | v) -> so sánh 1 lệnh,
// hàng đợi nhị phân std::priority_queue (min-heap qua std::greater).
// -----------------------------------------------------------------------------
SSSPResult cpuDijkstra(const CSRGraph& g, u32 src) {
    SSSPResult r;
    HostTimer t; t.start();
    r.dist.assign(g.n, INF_DIST);
    r.parent.assign(g.n, NO_PARENT);
    std::priority_queue<u64, std::vector<u64>, std::greater<u64>> pq;
    r.dist[src] = 0;
    pq.push((u64)src);
    const u32* ro = g.row_offsets.data();
    const u32* ci = g.col_indices.data();
    const u32* wt = g.weights.data();
    while (!pq.empty()) {
        u64 top = pq.top(); pq.pop();
        u32 d = (u32)(top >> 32), u = (u32)(top & 0xFFFFFFFFu);
        if (d > r.dist[u]) continue;           // bản ghi cũ (lazy deletion)
        r.settled++;
        for (u32 k = ro[u]; k < ro[u + 1]; ++k) {
            u32 v = ci[k];
            u64 nd = (u64)d + wt[k];
            r.edgesScanned++;
            if (nd < r.dist[v]) {                // nới lỏng cạnh (edge relaxation)
                r.dist[v] = (u32)nd;
                r.parent[v] = u;
                pq.push((nd << 32) | v);
            }
        }
    }
    r.ms = t.elapsedMs();
    return r;
}

// -----------------------------------------------------------------------------
// A*: f(v) = g(v) + h(v). Dừng khi lấy đích ra khỏi hàng đợi.
// Heuristic nhất quán => mỗi đỉnh được chốt tối đa 1 lần; vẫn cho phép mở lại
// (lazy deletion) để an toàn ngay cả khi heuristic chỉ chấp nhận được.
// -----------------------------------------------------------------------------
P2PResult cpuAStar(const CSRGraph& g, u32 src, u32 dst, bool useHeuristic) {
    P2PResult r;
    HostTimer t; t.start();
    const bool useH = useHeuristic && g.hasCoords() && g.hScale > 0.0;
    const double xt = useH ? g.x[dst] : 0.0, yt = useH ? g.y[dst] : 0.0;
    auto H = [&](u32 v) -> u64 { return useH ? heuristicValue(g.x[v], g.y[v], xt, yt, g.hScale) : 0u; };

    std::vector<u32> dist(g.n, INF_DIST), parent(g.n, NO_PARENT);
    using Item = std::pair<u64, u32>;      // (f, v)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    dist[src] = 0;
    pq.push({H(src), src});
    const u32* ro = g.row_offsets.data();
    const u32* ci = g.col_indices.data();
    const u32* wt = g.weights.data();
    while (!pq.empty()) {
        Item it = pq.top(); pq.pop();
        u32 u = it.second;
        u64 du = dist[u];
        if (it.first > du + H(u)) continue;     // bản ghi cũ
        r.settled++;
        if (u == dst) break;                     // đích đã tối ưu
        for (u32 k = ro[u]; k < ro[u + 1]; ++k) {
            u32 v = ci[k];
            u64 nd = du + wt[k];
            r.edgesScanned++;
            if (nd < dist[v]) {
                dist[v] = (u32)nd;
                parent[v] = u;
                pq.push({nd + H(v), v});
            }
        }
    }
    r.dist = dist[dst];
    if (r.dist != INF_DIST) r.path = reconstructPath(parent, src, dst);
    r.ms = t.elapsedMs();
    return r;
}

MultiSourceCpuResult cpuMultiSource(const CSRGraph& g, const std::vector<u32>& srcs, int threads, bool keepDist) {
    MultiSourceCpuResult r;
    if (threads <= 0) threads = cpuMaxThreads();
    r.threads = threads;
    if (keepDist) r.dist.resize(srcs.size());
    u64 edges = 0;
    const int K = (int)srcs.size();
    HostTimer t; t.start();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(+ : edges)
#endif
    for (int i = 0; i < K; ++i) {
        SSSPResult s = cpuDijkstra(g, srcs[i]);
        edges += s.edgesScanned;
        if (keepDist) r.dist[i] = std::move(s.dist);
    }
    r.ms = t.elapsedMs();
    r.edgesScanned = edges;
    return r;
}

std::vector<u32> reconstructPath(const std::vector<u32>& parent, u32 src, u32 dst) {
    std::vector<u32> path;
    if (dst >= parent.size()) return path;
    u32 v = dst;
    for (size_t steps = 0; steps <= parent.size(); ++steps) {
        path.push_back(v);
        if (v == src) { std::reverse(path.begin(), path.end()); return path; }
        v = parent[v];
        if (v == NO_PARENT || v >= parent.size()) break;
    }
    return {};   // đứt chuỗi hoặc chu trình
}
