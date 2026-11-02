// =============================================================================
//  verify.h - Kiểm thử tính đúng đắn (Correctness Verification) GPU vs CPU
// =============================================================================
//  3 lớp kiểm tra:
//   (1) Khoảng cách: dist_gpu[v] == dist_cpu[v] với MỌI đỉnh v (so sánh nguyên, chính xác tuyệt đối)
//   (2) Cây đường đi ngắn nhất: với mọi v != src tới được, p = parent[v] phải thỏa
//       tồn tại cạnh (p,v) và dist[p] + w(p,v) == dist[v]  (w >= 1 => không có chu trình)
//   (3) Truy vết đường đi: dựng lại đường src->t từ parent cho các đích mẫu, kiểm tra
//       từng cạnh tồn tại và tổng trọng số == dist_cpu[t]
// =============================================================================
#pragma once
#include "graph.h"
#include <string>

struct VerifyReport {
    u64 distChecked = 0, distMismatch = 0;
    u64 treeChecked = 0, treeErrors = 0;
    u64 pathsChecked = 0, pathErrors = 0;
    std::string firstError;

    bool pass() const { return distMismatch == 0 && treeErrors == 0 && pathErrors == 0; }
    u64 totalChecks() const { return distChecked + treeChecked + pathsChecked; }
    u64 totalErrors() const { return distMismatch + treeErrors + pathErrors; }
    double passRate() const {
        u64 t = totalChecks();
        return t ? 100.0 * (double)(t - totalErrors()) / (double)t : 100.0;
    }
    void merge(const VerifyReport& o);
};

// SSSP: đối chiếu toàn bộ mảng dist + kiểm tra cây parent + truy vết samplePaths đường đi
VerifyReport verifySSSP(const CSRGraph& g, u32 src, const std::vector<u32>& refDist,
                        const std::vector<u32>& dist, const std::vector<u32>& parent,
                        u32 samplePaths, u64 seed);

// Điểm - điểm (A*): khoảng cách đích + tính hợp lệ của đường đi trả về
VerifyReport verifyP2P(const CSRGraph& g, u32 src, u32 dst, u32 refDist, u32 dist, const std::vector<u32>& path);

// Kiểm tra 1 đường đi: các cạnh liên tiếp tồn tại, tổng trọng số == expected
bool validatePath(const CSRGraph& g, const std::vector<u32>& path, u32 src, u32 dst, u32 expected, std::string* err);
