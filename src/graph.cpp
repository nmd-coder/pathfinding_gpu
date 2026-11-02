// =============================================================================
//  graph.cpp - STEP 1: Sinh / đọc đồ thị quy mô lớn và dựng cấu trúc CSR
// =============================================================================
#include "graph.h"
#include "heuristic.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------
// Dựng CSR: counting sort theo đỉnh nguồn -> O(n + m), sau đó sắp xếp mỗi hàng
// theo (trọng số, đỉnh đích) để phục vụ phân tách light/heavy của Delta-Stepping.
// -----------------------------------------------------------------------------
CSRGraph buildCSR(EdgeList&& el) {
    CSRGraph g;
    g.name = el.name;
    g.n = el.n;
    const size_t mIn = el.src.size();

    // Đếm bậc ra (bỏ khuyên u->u)
    std::vector<u32> deg(g.n + 1, 0);
    size_t mKeep = 0;
    for (size_t i = 0; i < mIn; ++i)
        if (el.src[i] != el.dst[i]) { deg[el.src[i]]++; mKeep++; }
    if (mKeep >= 0xFFFFFFFFull) throw std::runtime_error("Graph too large: m must be < 2^32");
    g.m = (u32)mKeep;

    // Prefix sum (exclusive scan) -> row_offsets
    g.row_offsets.assign(g.n + 1, 0);
    for (u32 u = 0; u < g.n; ++u) g.row_offsets[u + 1] = g.row_offsets[u] + deg[u];

    // Phân phối cạnh vào vị trí (scatter)
    g.col_indices.resize(g.m);
    g.weights.resize(g.m);
    std::vector<u32> cursor(g.row_offsets.begin(), g.row_offsets.end() - 1);
    for (size_t i = 0; i < mIn; ++i) {
        u32 s = el.src[i], d = el.dst[i];
        if (s == d) continue;
        u32 pos = cursor[s]++;
        g.col_indices[pos] = d;
        g.weights[pos] = std::max<u32>(1u, el.w[i]);  // trọng số >= 1 (đảm bảo cây cha không có chu trình)
    }
    // Giải phóng danh sách cạnh thô sớm để tiết kiệm RAM
    std::vector<u32>().swap(el.src);
    std::vector<u32>().swap(el.dst);
    std::vector<u32>().swap(el.w);

    // Sắp xếp mỗi hàng theo (w, dst) tăng dần
    const int N = (int)g.n;
#pragma omp parallel
    {
        std::vector<u64> tmp;
#pragma omp for schedule(dynamic, 4096)
        for (int ui = 0; ui < N; ++ui) {
            u32 b = g.row_offsets[ui], e = g.row_offsets[ui + 1];
            if (e - b < 2) continue;
            tmp.resize(e - b);
            for (u32 k = b; k < e; ++k) tmp[k - b] = ((u64)g.weights[k] << 32) | g.col_indices[k];
            std::sort(tmp.begin(), tmp.end());
            for (u32 k = b; k < e; ++k) {
                g.weights[k] = (u32)(tmp[k - b] >> 32);
                g.col_indices[k] = (u32)(tmp[k - b] & 0xFFFFFFFFu);
            }
        }
    }

    // Thống kê
    g.minW = g.m ? 0xFFFFFFFFu : 0; g.maxW = 0;
    double sumW = 0.0;
    for (u32 k = 0; k < g.m; ++k) {
        g.minW = std::min(g.minW, g.weights[k]);
        g.maxW = std::max(g.maxW, g.weights[k]);
        sumW += g.weights[k];
    }
    g.avgW = g.m ? sumW / g.m : 0.0;
    g.maxDeg = 0;
    for (u32 u = 0; u < g.n; ++u) g.maxDeg = std::max(g.maxDeg, g.degree(u));
    g.avgDeg = g.n ? (double)g.m / g.n : 0.0;

    // Tọa độ + hệ số heuristic chấp nhận được (xem heuristic.h)
    g.x = std::move(el.x);
    g.y = std::move(el.y);
    if (g.hasCoords()) {
        double scale = std::numeric_limits<double>::infinity();
        for (u32 u = 0; u < g.n; ++u)
            for (u32 k = g.row_offsets[u]; k < g.row_offsets[u + 1]; ++k) {
                u32 v = g.col_indices[k];
                double dx = g.x[u] - g.x[v], dy = g.y[u] - g.y[v];
                double e = std::sqrt(dx * dx + dy * dy);
                if (e > 1e-12) scale = std::min(scale, g.weights[k] / e);
            }
        g.hScale = std::isfinite(scale) ? scale : 0.0;
    }
    return g;
}

// -----------------------------------------------------------------------------
// Lưới đường bộ 2D có tọa độ (mô phỏng mạng giao thông đô thị).
//  - Khoảng cách lưới 100 đơn vị, tọa độ dao động ±25 (đường phố không thẳng tắp)
//  - Đường thường: w = |uv| * U(1.15, 1.9)  (đường cong, đèn đỏ, ...)
//  - Trục chính (mỗi 16 hàng/cột): w = |uv| * U(1.0, 1.15)  -> đường đi ngắn nhất ưu tiên trục chính
//  - Xóa ngẫu nhiên 5% đoạn đường thường, thêm 5% đường chéo mỗi hướng
// -----------------------------------------------------------------------------
EdgeList genGridRoad(u32 rows, u32 cols, u64 seed) {
    EdgeList el;
    el.name = "grid_" + std::to_string(rows) + "x" + std::to_string(cols);
    el.n = rows * cols;
    el.x.resize(el.n);
    el.y.resize(el.n);
    Rng rng(seed);
    for (u32 r = 0; r < rows; ++r)
        for (u32 c = 0; c < cols; ++c) {
            u32 id = r * cols + c;
            el.x[id] = c * 100.0 + rng.uniform(-25.0, 25.0);
            el.y[id] = r * 100.0 + rng.uniform(-25.0, 25.0);
        }
    size_t est = (size_t)el.n * 4 + 16;
    el.src.reserve(est); el.dst.reserve(est); el.w.reserve(est);

    auto addRoad = [&](u32 a, u32 b, bool arterial) {
        double dx = el.x[a] - el.x[b], dy = el.y[a] - el.y[b];
        double e = std::sqrt(dx * dx + dy * dy);
        double f = arterial ? rng.uniform(1.0, 1.15) : rng.uniform(1.15, 1.9);
        u32 w = (u32)std::max<long long>(1, std::llround(e * f));
        el.src.push_back(a); el.dst.push_back(b); el.w.push_back(w);   // đường 2 chiều
        el.src.push_back(b); el.dst.push_back(a); el.w.push_back(w);
    };
    for (u32 r = 0; r < rows; ++r)
        for (u32 c = 0; c < cols; ++c) {
            u32 id = r * cols + c;
            if (c + 1 < cols) {
                bool art = (r % 16 == 0);
                if (art || rng.uniform() < 0.95) addRoad(id, id + 1, art);
            }
            if (r + 1 < rows) {
                bool art = (c % 16 == 0);
                if (art || rng.uniform() < 0.95) addRoad(id, id + cols, art);
            }
            if (r + 1 < rows && c + 1 < cols && rng.uniform() < 0.05) addRoad(id, id + cols + 1, false);
            if (r + 1 < rows && c > 0 && rng.uniform() < 0.05) addRoad(id, id + cols - 1, false);
        }
    return el;
}

// -----------------------------------------------------------------------------
// R-MAT (Chakrabarti et al. 2004), tham số Graph500. Vô hướng, khử cạnh lặp,
// hoán vị ngẫu nhiên nhãn đỉnh để tránh cục bộ nhân tạo.
// -----------------------------------------------------------------------------
EdgeList genRMAT(u32 scale, u32 edgeFactor, u32 maxW, u64 seed) {
    EdgeList el;
    el.name = "rmat_s" + std::to_string(scale) + "_ef" + std::to_string(edgeFactor);
    el.n = 1u << scale;
    const u64 target = (u64)el.n * edgeFactor;
    const double A = 0.57, B = 0.19, C = 0.19;
    Rng rng(seed);

    std::vector<u32> perm(el.n);
    std::iota(perm.begin(), perm.end(), 0u);
    for (u32 i = el.n - 1; i > 0; --i) std::swap(perm[i], perm[rng.below(i + 1)]);

    std::vector<u64> keys;
    keys.reserve(target);
    for (u64 e = 0; e < target; ++e) {
        u32 u = 0, v = 0;
        for (u32 b = 0; b < scale; ++b) {
            double r = rng.uniform();
            u32 bu = 0, bv = 0;
            if (r < A) {}
            else if (r < A + B) bv = 1;
            else if (r < A + B + C) bu = 1;
            else { bu = 1; bv = 1; }
            u = (u << 1) | bu; v = (v << 1) | bv;
        }
        u = perm[u]; v = perm[v];
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        keys.push_back(((u64)u << 32) | v);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    el.src.reserve(keys.size() * 2); el.dst.reserve(keys.size() * 2); el.w.reserve(keys.size() * 2);
    for (u64 k : keys) {
        u32 u = (u32)(k >> 32), v = (u32)(k & 0xFFFFFFFFu);
        u32 w = 1 + (u32)(hash64(k ^ (seed * 0x9E37ull)) % maxW);
        el.src.push_back(u); el.dst.push_back(v); el.w.push_back(w);
        el.src.push_back(v); el.dst.push_back(u); el.w.push_back(w);
    }
    return el;
}

// -----------------------------------------------------------------------------
// Đọc file văn bản nhanh (nạp toàn bộ vào bộ nhớ rồi tự phân tích số)
// -----------------------------------------------------------------------------
static bool readWholeFile(const std::string& path, std::string& buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    f.seekg(0, std::ios::beg);
    buf.resize((size_t)sz);
    if (sz > 0) f.read(&buf[0], sz);
    return (bool)f || f.eof();
}
static inline void skipSpaces(const char*& p, const char* e) { while (p < e && (*p == ' ' || *p == '\t' || *p == '\r')) ++p; }
static inline void skipLine(const char*& p, const char* e) { while (p < e && *p != '\n') ++p; if (p < e) ++p; }
static inline bool readI64(const char*& p, const char* e, long long& out) {
    skipSpaces(p, e);
    bool neg = false;
    if (p < e && (*p == '-' || *p == '+')) { neg = (*p == '-'); ++p; }
    if (p >= e || *p < '0' || *p > '9') return false;
    long long v = 0;
    while (p < e && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
    out = neg ? -v : v;
    return true;
}

bool loadDIMACS(const std::string& grPath, const std::string& coPath, EdgeList& out) {
    std::string buf;
    if (!readWholeFile(grPath, buf)) { fprintf(stderr, "Cannot open %s\n", grPath.c_str()); return false; }
    out = EdgeList();
    {
        size_t slash = grPath.find_last_of("/\\");
        out.name = (slash == std::string::npos) ? grPath : grPath.substr(slash + 1);
    }
    const char* p = buf.data();
    const char* e = p + buf.size();
    while (p < e) {
        if (*p == 'p') {                       // p sp <n> <m>
            p++; skipSpaces(p, e);
            while (p < e && *p != ' ' && *p != '\t') ++p;   // bỏ "sp"
            long long n = 0, m = 0;
            readI64(p, e, n); readI64(p, e, m);
            out.n = (u32)n;
            out.src.reserve((size_t)m); out.dst.reserve((size_t)m); out.w.reserve((size_t)m);
            skipLine(p, e);
        } else if (*p == 'a') {                // a <u> <v> <w>  (đỉnh đánh số từ 1)
            p++;
            long long u, v, w;
            if (readI64(p, e, u) && readI64(p, e, v) && readI64(p, e, w)) {
                out.src.push_back((u32)(u - 1)); out.dst.push_back((u32)(v - 1)); out.w.push_back((u32)w);
            }
            skipLine(p, e);
        } else skipLine(p, e);
    }
    if (out.n == 0) { fprintf(stderr, "Invalid DIMACS .gr (missing 'p sp' line)\n"); return false; }

    if (!coPath.empty()) {
        std::string cb;
        if (!readWholeFile(coPath, cb)) { fprintf(stderr, "Cannot open %s\n", coPath.c_str()); return false; }
        std::vector<double> lon(out.n, 0.0), lat(out.n, 0.0);
        const char* q = cb.data();
        const char* qe = q + cb.size();
        double latSum = 0.0; u64 cnt = 0;
        while (q < qe) {
            if (*q == 'v') {                   // v <id> <x=lon*1e6> <y=lat*1e6>
                q++;
                long long id, X, Y;
                if (readI64(q, qe, id) && readI64(q, qe, X) && readI64(q, qe, Y) && id >= 1 && (u64)id <= out.n) {
                    lon[id - 1] = (double)X; lat[id - 1] = (double)Y; latSum += (double)Y; cnt++;
                }
            }
            skipLine(q, qe);
        }
        // Phép chiếu equirectangular: x = lon*cos(lat0), y = lat -> mặt phẳng Euclid
        // (bất đẳng thức tam giác đúng, hScale tự hiệu chỉnh đơn vị theo dữ liệu)
        double lat0 = cnt ? (latSum / cnt) * 1e-6 * 3.14159265358979323846 / 180.0 : 0.0;
        double cs = std::cos(lat0);
        out.x.resize(out.n); out.y.resize(out.n);
        for (u32 i = 0; i < out.n; ++i) { out.x[i] = lon[i] * cs; out.y[i] = lat[i]; }
    }
    return true;
}

bool loadSNAP(const std::string& path, bool symmetrize, u32 maxW, u64 seed, EdgeList& out) {
    std::string buf;
    if (!readWholeFile(path, buf)) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }
    out = EdgeList();
    {
        size_t slash = path.find_last_of("/\\");
        out.name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    }
    const char* p = buf.data();
    const char* e = p + buf.size();
    u32 maxId = 0;
    while (p < e) {
        skipSpaces(p, e);
        if (p >= e) break;
        if (*p == '#' || *p == '%' || *p == '\n') { skipLine(p, e); continue; }
        long long u, v, w = -1;
        if (!readI64(p, e, u) || !readI64(p, e, v)) { skipLine(p, e); continue; }
        const char* save = p;
        if (!readI64(p, e, w)) { p = save; w = -1; }
        skipLine(p, e);
        if (u < 0 || v < 0) continue;
        if (w <= 0) {   // không có trọng số: sinh đối xứng, tái lập được
            u64 a = (u64)std::min(u, v), b = (u64)std::max(u, v);
            w = 1 + (long long)(hash64((a << 32 | b) ^ seed) % maxW);
        }
        out.src.push_back((u32)u); out.dst.push_back((u32)v); out.w.push_back((u32)w);
        if (symmetrize) { out.src.push_back((u32)v); out.dst.push_back((u32)u); out.w.push_back((u32)w); }
        maxId = std::max(maxId, (u32)std::max(u, v));
    }
    out.n = maxId + 1;
    return true;
}

// -----------------------------------------------------------------------------
// Cache nhị phân
// -----------------------------------------------------------------------------
static const char kMagic[8] = {'P', 'F', 'C', 'S', 'R', '0', '0', '1'};

bool saveBinary(const CSRGraph& g, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    u32 nameLen = (u32)g.name.size();
    u32 hasXY = g.hasCoords() ? 1u : 0u;
    fwrite(kMagic, 1, 8, f);
    fwrite(&g.n, 4, 1, f); fwrite(&g.m, 4, 1, f); fwrite(&hasXY, 4, 1, f); fwrite(&nameLen, 4, 1, f);
    fwrite(g.name.data(), 1, nameLen, f);
    fwrite(g.row_offsets.data(), 4, (size_t)g.n + 1, f);
    fwrite(g.col_indices.data(), 4, g.m, f);
    fwrite(g.weights.data(), 4, g.m, f);
    if (hasXY) { fwrite(g.x.data(), 8, g.n, f); fwrite(g.y.data(), 8, g.n, f); }
    fclose(f);
    return true;
}

bool loadBinary(const std::string& path, CSRGraph& g) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char mg[8];
    u32 hasXY = 0, nameLen = 0;
    bool ok = fread(mg, 1, 8, f) == 8 && memcmp(mg, kMagic, 8) == 0;
    ok = ok && fread(&g.n, 4, 1, f) == 1 && fread(&g.m, 4, 1, f) == 1 &&
         fread(&hasXY, 4, 1, f) == 1 && fread(&nameLen, 4, 1, f) == 1;
    if (!ok) { fclose(f); return false; }
    g.name.resize(nameLen);
    if (nameLen) ok = fread(&g.name[0], 1, nameLen, f) == nameLen;
    // Đọc lại dưới dạng EdgeList để tái sử dụng buildCSR (tính thống kê + hScale)
    std::vector<u32> ro((size_t)g.n + 1), ci(g.m), w(g.m);
    ok = ok && fread(ro.data(), 4, ro.size(), f) == ro.size();
    ok = ok && fread(ci.data(), 4, ci.size(), f) == ci.size();
    ok = ok && fread(w.data(), 4, w.size(), f) == w.size();
    EdgeList el;
    el.name = g.name; el.n = g.n;
    if (ok && hasXY) {
        el.x.resize(g.n); el.y.resize(g.n);
        ok = fread(el.x.data(), 8, g.n, f) == g.n && fread(el.y.data(), 8, g.n, f) == g.n;
    }
    fclose(f);
    if (!ok) return false;
    el.src.resize(g.m); el.dst = std::move(ci); el.w = std::move(w);
    for (u32 u = 0; u < g.n; ++u)
        for (u32 k = ro[u]; k < ro[u + 1]; ++k) el.src[k] = u;
    g = buildCSR(std::move(el));
    return true;
}

std::vector<u32> pickSources(const CSRGraph& g, u32 count, u64 seed) {
    std::vector<u32> s;
    Rng rng(seed);
    for (u32 tries = 0; s.size() < count && tries < count * 1000u + 1000u; ++tries) {
        u32 v = rng.below(g.n);
        if (g.degree(v) > 0) s.push_back(v);
    }
    while (s.size() < count) s.push_back(0);
    return s;
}

void printGraphInfo(const CSRGraph& g) {
    printf("%s  Graph  %s%s\n", term::bold(), g.name.c_str(), term::reset());
    printf("    Vertices (n)      : %u\n", g.n);
    printf("    Directed arcs (m) : %u   (avg degree %.2f, max degree %u)\n", g.m, g.avgDeg, g.maxDeg);
    printf("    Weights           : [%u .. %u], mean %.1f\n", g.minW, g.maxW, g.avgW);
    printf("    Coordinates       : %s", g.hasCoords() ? "yes" : "no (A* -> h = 0)");
    if (g.hasCoords()) printf("   (admissible h-scale = %.6g)", g.hScale);
    printf("\n    CSR size          : %.2f MB  (row_offsets + col_indices + weights)\n", g.bytesCSR() / 1048576.0);
}
