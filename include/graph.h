// =============================================================================
//  graph.h - STEP 1: Biểu diễn đồ thị dạng danh sách kề nén CSR
// =============================================================================
//  CSR (Compressed Sparse Row) gồm 3 mảng 1 chiều liên tục:
//    row_offsets[n+1] : cạnh ra của đỉnh u nằm trong đoạn [row_offsets[u], row_offsets[u+1])
//    col_indices[m]   : đỉnh đích của từng cạnh
//    weights[m]       : trọng số (nguyên, >= 1) của từng cạnh
//  Ví dụ: 0->1(5), 0->2(3), 1->2(1), 2->0(7)
//    row_offsets = [0, 2, 3, 4]
//    col_indices = [1, 2, 2, 0]
//    weights     = [5, 3, 1, 7]
//  Bộ nhớ: (n+1)*4 + m*8 byte, truy cập tuần tự -> coalesced trên GPU.
//
//  Ngoài ra: mỗi hàng được SẮP XẾP THEO TRỌNG SỐ TĂNG DẦN để Delta-Stepping
//  tách cạnh nhẹ/nặng (light/heavy) chỉ bằng 1 phép tìm kiếm nhị phân / đỉnh
//  (light: [row_offsets[u], lightEnd[u]),  heavy: [lightEnd[u], row_offsets[u+1])).
// =============================================================================
#pragma once
#include "common.h"
#include <string>
#include <vector>

// Danh sách cạnh thô (đầu vào cho bộ dựng CSR)
struct EdgeList {
    std::string name;
    u32 n = 0;
    std::vector<u32> src, dst, w;
    std::vector<double> x, y;     // tọa độ (rỗng nếu đồ thị không có tọa độ)
};

struct CSRGraph {
    std::string name;
    u32 n = 0;                    // số đỉnh
    u32 m = 0;                    // số cạnh có hướng (arc)
    std::vector<u32> row_offsets; // n+1
    std::vector<u32> col_indices; // m
    std::vector<u32> weights;     // m
    std::vector<double> x, y;     // tọa độ phẳng cho A* (có thể rỗng)
    double hScale = 0.0;          // hệ số heuristic chấp nhận được (0 => A* suy biến thành Dijkstra)

    // Thống kê
    u32 minW = 0, maxW = 0, maxDeg = 0;
    double avgW = 0.0, avgDeg = 0.0;

    bool hasCoords() const { return !x.empty(); }
    u32 degree(u32 u) const { return row_offsets[u + 1] - row_offsets[u]; }
    size_t bytesCSR() const { return (size_t)(n + 1) * 4 + (size_t)m * 8; }
};

// ----- Dựng CSR từ danh sách cạnh (counting sort theo đỉnh nguồn, O(n + m)) -----
CSRGraph buildCSR(EdgeList&& el);

// ----- Bộ sinh đồ thị -----
// Lưới đường bộ 2D (giống mạng giao thông đô thị): rows x cols đỉnh, có tọa độ,
// cạnh 4-láng giềng (xóa ngẫu nhiên ~5%), một ít đường chéo, các "trục chính" rẻ hơn.
EdgeList genGridRoad(u32 rows, u32 cols, u64 seed);
// R-MAT (Graph500, a=0.57 b=0.19 c=0.19): bậc phân bố lũy thừa, không có tọa độ.
EdgeList genRMAT(u32 scale, u32 edgeFactor, u32 maxW, u64 seed);

// ----- Bộ đọc dữ liệu thực -----
// DIMACS 9th Challenge (.gr: "a u v w", .co: "v id x y"), ví dụ USA-road-d.NY
bool loadDIMACS(const std::string& grPath, const std::string& coPath, EdgeList& out);
// SNAP edge list ("u v" hoặc "u v w", dòng '#' là chú thích). Thiếu trọng số => sinh ngẫu nhiên.
bool loadSNAP(const std::string& path, bool symmetrize, u32 maxW, u64 seed, EdgeList& out);

// ----- Cache nhị phân để nạp nhanh đồ thị lớn -----
bool saveBinary(const CSRGraph& g, const std::string& path);
bool loadBinary(const std::string& path, CSRGraph& g);

// Chọn ngẫu nhiên các đỉnh nguồn có bậc ra > 0
std::vector<u32> pickSources(const CSRGraph& g, u32 count, u64 seed);
void printGraphInfo(const CSRGraph& g);
