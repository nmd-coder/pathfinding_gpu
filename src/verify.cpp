// =============================================================================
//  verify.cpp - Kiểm thử đúng đắn GPU vs CPU
// =============================================================================
#include "verify.h"
#include "cpu_sssp.h"

#include <algorithm>
#include <cstdio>

void VerifyReport::merge(const VerifyReport& o) {
    distChecked += o.distChecked;  distMismatch += o.distMismatch;
    treeChecked += o.treeChecked;  treeErrors += o.treeErrors;
    pathsChecked += o.pathsChecked; pathErrors += o.pathErrors;
    if (firstError.empty()) firstError = o.firstError;
}

// Trọng số nhỏ nhất của cung a->b (INF nếu không tồn tại)
static u32 minArcWeight(const CSRGraph& g, u32 a, u32 b) {
    u32 best = INF_DIST;
    for (u32 k = g.row_offsets[a]; k < g.row_offsets[a + 1]; ++k)
        if (g.col_indices[k] == b) best = std::min(best, g.weights[k]);
    return best;
}

bool validatePath(const CSRGraph& g, const std::vector<u32>& path, u32 src, u32 dst, u32 expected, std::string* err) {
    char buf[256];
    if (expected == INF_DIST) {
        if (!path.empty()) { if (err) *err = "path returned for unreachable target"; return false; }
        return true;
    }
    if (path.empty() || path.front() != src || path.back() != dst) {
        snprintf(buf, sizeof(buf), "path %u->%u: empty or wrong endpoints", src, dst);
        if (err) *err = buf;
        return false;
    }
    u64 sum = 0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        u32 w = minArcWeight(g, path[i], path[i + 1]);
        if (w == INF_DIST) {
            snprintf(buf, sizeof(buf), "path %u->%u: arc (%u,%u) does not exist", src, dst, path[i], path[i + 1]);
            if (err) *err = buf;
            return false;
        }
        sum += w;
    }
    if (sum != expected) {
        snprintf(buf, sizeof(buf), "path %u->%u: cost %llu != shortest %u", src, dst, (unsigned long long)sum, expected);
        if (err) *err = buf;
        return false;
    }
    return true;
}

VerifyReport verifySSSP(const CSRGraph& g, u32 src, const std::vector<u32>& refDist,
                        const std::vector<u32>& dist, const std::vector<u32>& parent,
                        u32 samplePaths, u64 seed) {
    VerifyReport r;
    char buf[256];
    // (1) Khoảng cách
    for (u32 v = 0; v < g.n; ++v) {
        r.distChecked++;
        if (dist[v] != refDist[v]) {
            if (r.distMismatch++ == 0) {
                snprintf(buf, sizeof(buf), "dist[%u]: gpu=%u cpu=%u", v, dist[v], refDist[v]);
                r.firstError = buf;
            }
        }
    }
    // (2) Cây đường đi ngắn nhất
    for (u32 v = 0; v < g.n; ++v) {
        if (v == src || dist[v] == INF_DIST) continue;
        r.treeChecked++;
        u32 p = parent[v];
        bool ok = p < g.n && dist[p] != INF_DIST;
        if (ok) {
            ok = false;
            for (u32 k = g.row_offsets[p]; k < g.row_offsets[p + 1]; ++k)
                if (g.col_indices[k] == v && (u64)dist[p] + g.weights[k] == dist[v]) { ok = true; break; }
        }
        if (!ok && r.treeErrors++ == 0 && r.firstError.empty()) {
            snprintf(buf, sizeof(buf), "parent[%u]=%u violates dist[p]+w==dist[v]", v, p);
            r.firstError = buf;
        }
    }
    // (3) Truy vết một số đường đi
    Rng rng(seed);
    for (u32 i = 0, tries = 0; i < samplePaths && tries < samplePaths * 50u; ++tries) {
        u32 t = rng.below(g.n);
        if (refDist[t] == INF_DIST) continue;
        ++i;
        r.pathsChecked++;
        std::vector<u32> path = reconstructPath(parent, src, t);
        std::string e;
        if (!validatePath(g, path, src, t, refDist[t], &e)) {
            if (r.pathErrors++ == 0 && r.firstError.empty()) r.firstError = e;
        }
    }
    return r;
}

VerifyReport verifyP2P(const CSRGraph& g, u32 src, u32 dst, u32 refDist, u32 dist, const std::vector<u32>& path) {
    VerifyReport r;
    char buf[256];
    r.distChecked = 1;
    if (dist != refDist) {
        r.distMismatch = 1;
        snprintf(buf, sizeof(buf), "A* %u->%u: gpu=%u ref=%u", src, dst, dist, refDist);
        r.firstError = buf;
    }
    r.pathsChecked = 1;
    std::string e;
    if (!validatePath(g, path, src, dst, refDist, &e)) {
        r.pathErrors = 1;
        if (r.firstError.empty()) r.firstError = e;
    }
    return r;
}
