// =============================================================================
//  gpu_sssp.cu - Điều phối phía Host cho các thuật toán GPU
//    * Cấp phát VRAM 1 lần, nạp CSR bằng pinned memory + 3 CUDA streams
//    * Vòng lặp Host cho Kernel 1 (Naive), Kernel 2a (Frontier), Kernel 2b (Delta-Stepping)
//    * Đo thời gian bằng cudaEvent_t trên cùng stream
// =============================================================================
#include "gpu_kernels.cuh"
#include "cpu_sssp.h"   // reconstructPath

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace pf;

namespace {
// Vị trí các bộ đếm trong mảng d_ctr (đọc về Host bằng 1 lệnh memcpy mỗi vòng lặp)
enum Ctr : u32 { C_NEAR0 = 0, C_NEAR1 = 1, C_FAR0 = 2, C_FAR1 = 3, C_R = 4, C_CHANGED = 5, C_COUNT = 8 };

inline u32 divUp(size_t a, u32 b) { return (u32)((a + b - 1) / b); }
inline size_t maskWords(size_t bits) { return (bits + 31) / 32; }

// Chuyển tham số runtime -> tham số template của kernel
template <int SEL, bool ASTAR, bool MARK_R>
void launchRelaxT(const RelaxCtx& c, AtomicMode am, Mapping mp, int bs, cudaStream_t s) {
    const u32 grid = divUp(c.inSize, (u32)bs);
#define PF_LAUNCH(AMV, MPV) relaxFrontierKernel<SEL, ASTAR, AMV, MPV, MARK_R><<<grid, bs, 0, s>>>(c)
#define PF_BY_MAP(AMV)                                                   \
    switch (mp) {                                                        \
        case Mapping::Thread: PF_LAUNCH(AMV, Mapping::Thread); break;    \
        case Mapping::Warp:   PF_LAUNCH(AMV, Mapping::Warp); break;      \
        default:              PF_LAUNCH(AMV, Mapping::Hybrid); break;    \
    }
    switch (am) {
        case AtomicMode::CasRBA:     PF_BY_MAP(AtomicMode::CasRBA); break;
        case AtomicMode::MinNoCheck: PF_BY_MAP(AtomicMode::MinNoCheck); break;
        default:                     PF_BY_MAP(AtomicMode::MinRBA); break;
    }
#undef PF_BY_MAP
#undef PF_LAUNCH
    CUDA_CHECK_LAUNCH();
}

template <bool ASTAR>
void launchNaiveT(const RelaxCtx& c, u8* aIn, u8* aOut, u32* changed, AtomicMode am, int bs, cudaStream_t s) {
    const u32 grid = divUp(c.g.n, (u32)bs);
    switch (am) {
        case AtomicMode::CasRBA:
            naiveRelaxKernel<ASTAR, AtomicMode::CasRBA><<<grid, bs, 0, s>>>(c, aIn, aOut, changed); break;
        case AtomicMode::MinNoCheck:
            naiveRelaxKernel<ASTAR, AtomicMode::MinNoCheck><<<grid, bs, 0, s>>>(c, aIn, aOut, changed); break;
        default:
            naiveRelaxKernel<ASTAR, AtomicMode::MinRBA><<<grid, bs, 0, s>>>(c, aIn, aOut, changed); break;
    }
    CUDA_CHECK_LAUNCH();
}
}  // namespace

// =============================================================================
struct GpuSolver::Impl {
    const CSRGraph& g;
    u32 n, m, K;
    DevGraph dg{};
    u32 *d_row = nullptr, *d_col = nullptr, *d_w = nullptr, *d_lightEnd = nullptr;
    double *d_x = nullptr, *d_y = nullptr;
    u32 lightDelta = 0;

    u64* d_state = nullptr;       // [K*n] (dist<<32 | parent)
    u32* d_h = nullptr;           // [n] heuristic
    u8* d_act[2] = {nullptr, nullptr};          // cờ active (Kernel 1)
    u32* d_q[2] = {nullptr, nullptr};           // F_curr / F_next  [K*n]
    u32* d_qMask[2] = {nullptr, nullptr};       // bitmask của F_curr / F_next
    u32* d_far[2] = {nullptr, nullptr};         // far pile (Delta-Stepping) [n]
    u32* d_farMask = nullptr;
    u32* d_r = nullptr;                          // tập R [n]
    u32* d_rMask = nullptr;
    u32* d_ctr = nullptr;                        // bộ đếm hàng đợi + cờ hội tụ
    unsigned long long* d_minKey = nullptr;
    unsigned long long* d_stats = nullptr;
    u32* d_srcs = nullptr;

    u32* h_ctr = nullptr;                        // pinned mirror của d_ctr
    unsigned long long* h_minKey = nullptr;

    cudaStream_t s{};
    cudaEvent_t ev0{}, ev1{};
    double uploadMs = 0.0;
    size_t bytes = 0;

    template <class T>
    T* dalloc(size_t count) {
        T* p = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&p, std::max<size_t>(1, count) * sizeof(T)));
        bytes += count * sizeof(T);
        return p;
    }

    Impl(const CSRGraph& graph, u32 maxBatch) : g(graph), n(graph.n), m(graph.m), K(std::max<u32>(1, maxBatch)) {
        if ((u64)K * n >= 0xFFFFFFFFull) throw std::runtime_error("maxBatch * n must be < 2^32");
        CUDA_CHECK(cudaSetDevice(0));
        // ---- Ước lượng bộ nhớ cần thiết & kiểm tra VRAM còn trống ----
        const size_t Kn = (size_t)K * n;
        size_t need = (size_t)(n + 1) * 4 + (size_t)m * 8 + (size_t)n * 4      // CSR + lightEnd
                    + (g.hasCoords() ? (size_t)n * 16 : 0)                       // tọa độ
                    + Kn * 8 + (size_t)n * 4 + (size_t)n * 2                     // state, h, active
                    + 2 * Kn * 4 + 2 * maskWords(Kn) * 4                          // F_curr/F_next + mask
                    + 3 * (size_t)n * 4 + 2 * maskWords(n) * 4;                  // far x2, R + mask
        size_t freeB = 0, totalB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
        if (need > freeB) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Not enough GPU memory: need %.1f MB, free %.1f MB (reduce graph or batch)",
                     need / 1048576.0, freeB / 1048576.0);
            throw std::runtime_error(buf);
        }
        CUDA_CHECK(cudaStreamCreate(&s));
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));

        // ---- STEP 1: Nạp 3 mảng CSR lên VRAM (pinned staging + 3 streams song song) ----
        HostTimer t; t.start();
        d_row = dalloc<u32>((size_t)n + 1);
        d_col = dalloc<u32>(m);
        d_w = dalloc<u32>(m);
        d_lightEnd = dalloc<u32>(n);
        {
            const size_t bRow = (size_t)(n + 1) * 4, bCol = (size_t)m * 4, bW = (size_t)m * 4;
            char* stage = nullptr;
            CUDA_CHECK(cudaMallocHost((void**)&stage, bRow + bCol + bW));
            memcpy(stage, g.row_offsets.data(), bRow);
            memcpy(stage + bRow, g.col_indices.data(), bCol);
            memcpy(stage + bRow + bCol, g.weights.data(), bW);
            cudaStream_t st[3];
            for (auto& x : st) CUDA_CHECK(cudaStreamCreate(&x));
            CUDA_CHECK(cudaMemcpyAsync(d_row, stage, bRow, cudaMemcpyHostToDevice, st[0]));
            CUDA_CHECK(cudaMemcpyAsync(d_col, stage + bRow, bCol, cudaMemcpyHostToDevice, st[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_w, stage + bRow + bCol, bW, cudaMemcpyHostToDevice, st[2]));
            for (auto& x : st) { CUDA_CHECK(cudaStreamSynchronize(x)); CUDA_CHECK(cudaStreamDestroy(x)); }
            CUDA_CHECK(cudaFreeHost(stage));
        }
        if (g.hasCoords()) {
            d_x = dalloc<double>(n);
            d_y = dalloc<double>(n);
            CUDA_CHECK(cudaMemcpy(d_x, g.x.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_y, g.y.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
        }
        uploadMs = t.elapsedMs();
        dg = DevGraph{d_row, d_col, d_w, d_lightEnd, n, m};

        // ---- Bộ đệm làm việc ----
        d_state = dalloc<u64>(Kn);
        d_h = dalloc<u32>(n);
        d_act[0] = dalloc<u8>(n);
        d_act[1] = dalloc<u8>(n);
        for (int i = 0; i < 2; ++i) {
            d_q[i] = dalloc<u32>(Kn);
            d_qMask[i] = dalloc<u32>(maskWords(Kn));
            d_far[i] = dalloc<u32>(n);
        }
        d_farMask = dalloc<u32>(maskWords(n));
        d_r = dalloc<u32>(n);
        d_rMask = dalloc<u32>(maskWords(n));
        d_ctr = dalloc<u32>(C_COUNT);
        d_minKey = dalloc<unsigned long long>(1);
        d_stats = dalloc<unsigned long long>(4);
        d_srcs = dalloc<u32>(K);
        CUDA_CHECK(cudaMallocHost((void**)&h_ctr, C_COUNT * sizeof(u32)));
        CUDA_CHECK(cudaMallocHost((void**)&h_minKey, sizeof(unsigned long long)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~Impl() {
        void* ptrs[] = {d_row, d_col, d_w, d_lightEnd, d_x, d_y, d_state, d_h, d_act[0], d_act[1],
                        d_q[0], d_q[1], d_qMask[0], d_qMask[1], d_far[0], d_far[1], d_farMask,
                        d_r, d_rMask, d_ctr, d_minKey, d_stats, d_srcs};
        for (void* p : ptrs) if (p) cudaFree(p);
        if (h_ctr) cudaFreeHost(h_ctr);
        if (h_minKey) cudaFreeHost(h_minKey);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaStreamDestroy(s);
    }

    u32 autoDelta() const {
        // Davidson et al. (IPDPS 2014): Delta ~ warpSize * avg_w / avg_deg
        double d = 32.0 * g.avgW / std::max(1.0, g.avgDeg);
        return (u32)std::max<long long>(1, std::llround(d));
    }

    void ensureLightEnd(u32 delta) {
        if (delta == lightDelta) return;
        lightEndKernel<<<divUp(n, 256), 256, 0, s>>>(d_row, d_w, n, delta, d_lightEnd);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaStreamSynchronize(s));
        lightDelta = delta;
    }

    void readCounters() {
        CUDA_CHECK(cudaMemcpyAsync(h_ctr, d_ctr, C_COUNT * sizeof(u32), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
    void zeroCounter(u32 i) { CUDA_CHECK(cudaMemsetAsync(d_ctr + i, 0, sizeof(u32), s)); }

    RelaxCtx makeCtx(bool astar, u32 target) const {
        RelaxCtx c{};
        c.g = dg;
        c.state = d_state;
        c.h = astar ? d_h : nullptr;
        c.target = target;
        c.stats = d_stats;
        c.bound = ~0ull;
        return c;
    }

    void launchRelax(int sel, bool astar, const RelaxCtx& c, const GpuOptions& o) {
        if (c.inSize == 0) return;
        if (sel == SEL_ALL) {
            if (astar) launchRelaxT<SEL_ALL, true, false>(c, o.atomic, o.mapping, o.blockSize, s);
            else       launchRelaxT<SEL_ALL, false, false>(c, o.atomic, o.mapping, o.blockSize, s);
        } else if (sel == SEL_LIGHT) {
            if (astar) launchRelaxT<SEL_LIGHT, true, true>(c, o.atomic, o.mapping, o.blockSize, s);
            else       launchRelaxT<SEL_LIGHT, false, true>(c, o.atomic, o.mapping, o.blockSize, s);
        } else {
            if (astar) launchRelaxT<SEL_HEAVY, true, false>(c, o.atomic, o.mapping, o.blockSize, s);
            else       launchRelaxT<SEL_HEAVY, false, false>(c, o.atomic, o.mapping, o.blockSize, s);
        }
    }

    // ------------------------------------------------------------------------
    // Kernel 1: Naive Parallel Relaxation (đồng bộ hóa mảng khoảng cách)
    // ------------------------------------------------------------------------
    void runNaive(bool astar, u32 dst, const GpuOptions& o, GpuStats& st) {
        CUDA_CHECK(cudaMemsetAsync(d_act[0], 0, n, s));
        CUDA_CHECK(cudaMemsetAsync(d_act[1], 0, n, s));
        seedKernel<<<1, 32, 0, s>>>(d_state, d_srcs, 1, n, DevQueue{nullptr, nullptr, nullptr}, d_act[0]);
        CUDA_CHECK_LAUNCH();
        RelaxCtx c = makeCtx(astar, dst);
        int cur = 0;
        while (true) {
            zeroCounter(C_CHANGED);                                   // d_changed = 0
            if (astar) launchNaiveT<true>(c, d_act[cur], d_act[cur ^ 1], d_ctr + C_CHANGED, o.atomic, o.blockSize, s);
            else       launchNaiveT<false>(c, d_act[cur], d_act[cur ^ 1], d_ctr + C_CHANGED, o.atomic, o.blockSize, s);
            readCounters();
            st.iterations++;
            cur ^= 1;
            if (h_ctr[C_CHANGED] == 0) break;                         // hội tụ
            if (st.iterations > n + 1) throw std::runtime_error("Naive relaxation did not converge");
        }
    }

    // ------------------------------------------------------------------------
    // Kernel 2a: Frontier-based (F_curr -> F_next), hỗ trợ batch K nguồn
    // ------------------------------------------------------------------------
    void runFrontier(u32 k, bool astar, u32 dst, const GpuOptions& o, GpuStats& st) {
        const size_t words = maskWords((size_t)k * n);
        CUDA_CHECK(cudaMemsetAsync(d_qMask[0], 0, words * 4, s));
        CUDA_CHECK(cudaMemsetAsync(d_qMask[1], 0, words * 4, s));
        seedKernel<<<divUp(k, 128), 128, 0, s>>>(d_state, d_srcs, k, n, DevQueue{d_q[0], d_ctr + C_NEAR0, d_qMask[0]}, nullptr);
        CUDA_CHECK_LAUNCH();
        u32 in = 0, cnt = k;
        while (cnt > 0) {
            const u32 out = in ^ 1u;
            zeroCounter(C_NEAR0 + out);
            RelaxCtx c = makeCtx(astar, dst);
            c.in = d_q[in];
            c.inSize = cnt;
            c.inMask = d_qMask[in];
            c.nearQ = DevQueue{d_q[out], d_ctr + C_NEAR0 + out, d_qMask[out]};
            c.farQ = c.nearQ;          // bound = ~0 => không bao giờ dùng far
            c.rQ = DevQueue{nullptr, nullptr, nullptr};
            launchRelax(SEL_ALL, astar, c, o);
            readCounters();
            cnt = h_ctr[C_NEAR0 + out];
            in = out;
            st.iterations++;
            if (st.iterations > n + 16) throw std::runtime_error("Frontier relaxation did not converge");
        }
    }

    // ------------------------------------------------------------------------
    // Kernel 2b: Delta-Stepping (Meyer & Sanders 2003) trên GPU
    //  bucket i chứa đỉnh có khóa trong [i*Delta, (i+1)*Delta) (khóa = dist, hoặc dist+h với A*)
    //  Pha nhẹ : lặp nới cạnh nhẹ (w <= Delta) của bucket tới khi bucket rỗng, ghi tập R
    //  Pha nặng: nới cạnh nặng (w > Delta) của R đúng 1 lần (kết quả rơi vào bucket sau)
    //  Chuyển bucket: tìm khóa nhỏ nhất trong far pile -> nhảy thẳng tới bucket không rỗng
    // ------------------------------------------------------------------------
    void runDelta(u32 src, bool astar, u32 dst, u32 delta, const GpuOptions& o, GpuStats& st) {
        const size_t words = maskWords(n);
        CUDA_CHECK(cudaMemsetAsync(d_qMask[0], 0, words * 4, s));
        CUDA_CHECK(cudaMemsetAsync(d_qMask[1], 0, words * 4, s));
        CUDA_CHECK(cudaMemsetAsync(d_farMask, 0, words * 4, s));
        CUDA_CHECK(cudaMemsetAsync(d_rMask, 0, words * 4, s));
        seedKernel<<<1, 32, 0, s>>>(d_state, d_srcs, 1, n, DevQueue{d_q[0], d_ctr + C_NEAR0, d_qMask[0]}, nullptr);
        CUDA_CHECK_LAUNCH();

        u64 key0 = 0;
        if (astar && g.hasCoords() && g.hScale > 0.0)
            key0 = heuristicValue(g.x[src], g.y[src], g.x[dst], g.y[dst], g.hScale);
        u64 bound = (key0 / delta + 1) * (u64)delta;
        u32 in = 0, fcur = 0;
        u32 nearCnt = 1, farCnt = 0, rCnt = 0;
        const u32 bs = (u32)o.blockSize;
        const u64 iterCap = 4ull * n + 1000;

        auto ctxFor = [&](const u32* inData, u32 inSize, u32* inMask, u32 nearIdx) {
            RelaxCtx c = makeCtx(astar, dst);
            c.in = inData;
            c.inSize = inSize;
            c.inMask = inMask;
            c.nearQ = DevQueue{d_q[nearIdx], d_ctr + C_NEAR0 + nearIdx, d_qMask[nearIdx]};
            c.farQ = DevQueue{d_far[fcur], d_ctr + C_FAR0 + fcur, d_farMask};
            c.rQ = DevQueue{d_r, d_ctr + C_R, d_rMask};
            c.bound = bound;
            return c;
        };

        while (true) {
            // ======== Xử lý bucket hiện tại ========
            while (true) {
                // Pha nhẹ: lặp tới khi bucket rỗng
                while (nearCnt > 0) {
                    const u32 out = in ^ 1u;
                    zeroCounter(C_NEAR0 + out);
                    RelaxCtx c = ctxFor(d_q[in], nearCnt, d_qMask[in], out);
                    launchRelax(SEL_LIGHT, astar, c, o);
                    readCounters();
                    nearCnt = h_ctr[C_NEAR0 + out];
                    farCnt = h_ctr[C_FAR0 + fcur];
                    rCnt = h_ctr[C_R];
                    in = out;
                    if (++st.iterations > iterCap) throw std::runtime_error("Delta-stepping did not converge");
                }
                if (rCnt == 0) break;
                // Pha nặng: nới cạnh nặng của R (xóa bit R khi xử lý)
                RelaxCtx c = ctxFor(d_r, rCnt, d_rMask, in);
                launchRelax(SEL_HEAVY, astar, c, o);
                zeroCounter(C_R);
                readCounters();
                nearCnt = h_ctr[C_NEAR0 + in];   // chỉ khác 0 với A* (khóa có thêm h)
                farCnt = h_ctr[C_FAR0 + fcur];
                rCnt = 0;
                ++st.iterations;
                if (nearCnt == 0) break;
            }
            st.buckets++;
            if (farCnt == 0) break;

            // ======== Chọn bucket kế tiếp (nhảy qua bucket rỗng) ========
            CUDA_CHECK(cudaMemsetAsync(d_minKey, 0xFF, sizeof(unsigned long long), s));
            if (astar) farMinKernel<true><<<divUp(farCnt, bs), bs, 0, s>>>(d_far[fcur], farCnt, d_state, d_h, dst, bound, d_minKey);
            else       farMinKernel<false><<<divUp(farCnt, bs), bs, 0, s>>>(d_far[fcur], farCnt, d_state, d_h, dst, bound, d_minKey);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpyAsync(h_minKey, d_minKey, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            const unsigned long long minKey = *h_minKey;
            if (minKey == ~0ull) break;                         // không còn ứng viên -> xong
            const u64 newBound = (minKey / delta + 1) * (u64)delta;

            const u32 fnext = fcur ^ 1u;
            zeroCounter(C_FAR0 + fnext);
            const DevQueue nearQ{d_q[in], d_ctr + C_NEAR0 + in, d_qMask[in]};
            const DevQueue farOut{d_far[fnext], d_ctr + C_FAR0 + fnext, d_farMask};
            if (astar) farPartitionKernel<true><<<divUp(farCnt, bs), bs, 0, s>>>(d_far[fcur], farCnt, d_state, d_h, dst, bound, newBound, nearQ, farOut);
            else       farPartitionKernel<false><<<divUp(farCnt, bs), bs, 0, s>>>(d_far[fcur], farCnt, d_state, d_h, dst, bound, newBound, nearQ, farOut);
            CUDA_CHECK_LAUNCH();
            readCounters();
            fcur = fnext;
            bound = newBound;
            nearCnt = h_ctr[C_NEAR0 + in];
            farCnt = h_ctr[C_FAR0 + fcur];
        }
    }

    // ------------------------------------------------------------------------
    // Điểm vào chung: khởi tạo + chạy + đo thời gian bằng cudaEvent
    // ------------------------------------------------------------------------
    GpuRunResult run(const u32* srcs, u32 k, bool astar, u32 dst, GpuOptions o, bool copyBack) {
        if (o.blockSize < 32 || o.blockSize > 1024 || (o.blockSize % 32) != 0) o.blockSize = 256;
        if (k == 0 || k > K) throw std::runtime_error("invalid batch size");
        for (u32 i = 0; i < k; ++i) if (srcs[i] >= n) throw std::runtime_error("source out of range");
        if (astar && dst >= n) throw std::runtime_error("target out of range");
        if (k > 1) o.algo = GpuAlgo::Frontier;               // batch đa nguồn dùng Frontier
        GpuRunResult r;
        const u32 delta = o.delta ? o.delta : autoDelta();
        if (o.algo == GpuAlgo::DeltaStep) ensureLightEnd(delta);
        r.stats.delta = (o.algo == GpuAlgo::DeltaStep) ? delta : 0;

        CUDA_CHECK(cudaMemsetAsync(d_stats, 0, 4 * sizeof(unsigned long long), s));
        CUDA_CHECK(cudaMemsetAsync(d_ctr, 0, C_COUNT * sizeof(u32), s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        CUDA_CHECK(cudaEventRecord(ev0, s));                 // ===== bắt đầu đo =====
        const size_t total = (size_t)k * n;
        const u32 initGrid = std::min<u32>(divUp(total, 256), 8192u);
        initStateKernel<<<initGrid, 256, 0, s>>>(d_state, total);
        CUDA_CHECK_LAUNCH();
        if (astar) {
            if (g.hasCoords() && g.hScale > 0.0) {
                heuristicKernel<<<divUp(n, 256), 256, 0, s>>>(d_x, d_y, n, g.x[dst], g.y[dst], g.hScale, d_h);
                CUDA_CHECK_LAUNCH();
            } else {
                CUDA_CHECK(cudaMemsetAsync(d_h, 0, (size_t)n * 4, s));   // không tọa độ: h = 0
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(d_srcs, srcs, k * sizeof(u32), cudaMemcpyHostToDevice, s));

        switch (o.algo) {
            case GpuAlgo::NaiveRelax: runNaive(astar, dst, o, r.stats); break;
            case GpuAlgo::Frontier:   runFrontier(k, astar, dst, o, r.stats); break;
            case GpuAlgo::DeltaStep:  runDelta(srcs[0], astar, dst, delta, o, r.stats); break;
        }
        CUDA_CHECK(cudaEventRecord(ev1, s));                 // ===== kết thúc đo =====
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        r.ms = ms;

        unsigned long long hs[4];
        CUDA_CHECK(cudaMemcpy(hs, d_stats, sizeof(hs), cudaMemcpyDeviceToHost));
        r.stats.edgesScanned = hs[0];
        r.stats.atomicsIssued = hs[1];
        r.stats.updates = hs[2];
        r.stats.pushes = hs[3];

        if (copyBack || astar) {
            std::vector<u64> st((size_t)k * n);
            CUDA_CHECK(cudaMemcpy(st.data(), d_state, st.size() * sizeof(u64), cudaMemcpyDeviceToHost));
            if (astar) {
                std::vector<u32> parent(n);
                for (u32 v = 0; v < n; ++v) parent[v] = (u32)(st[v] & 0xFFFFFFFFu);
                r.targetDist = (u32)(st[dst] >> 32);
                if (r.targetDist != INF_DIST) r.path = reconstructPath(parent, srcs[0], dst);
            } else if (k == 1) {
                r.dist.resize(n);
                r.parent.resize(n);
                for (u32 v = 0; v < n; ++v) {
                    r.dist[v] = (u32)(st[v] >> 32);
                    r.parent[v] = (u32)(st[v] & 0xFFFFFFFFu);
                }
            } else {
                stateCache.swap(st);
            }
        }
        return r;
    }
    std::vector<u64> stateCache;   // batch: giữ lại để tách theo từng nguồn
};

// =============================================================================
GpuSolver::GpuSolver(const CSRGraph& g, u32 maxBatch) : p_(new Impl(g, maxBatch)) {}
GpuSolver::~GpuSolver() { delete p_; }
double GpuSolver::uploadMs() const { return p_->uploadMs; }
size_t GpuSolver::deviceBytes() const { return p_->bytes; }
u32 GpuSolver::autoDelta() const { return p_->autoDelta(); }

GpuRunResult GpuSolver::sssp(u32 src, const GpuOptions& opt, bool copyBack) {
    return p_->run(&src, 1, false, 0, opt, copyBack);
}

GpuRunResult GpuSolver::astar(u32 src, u32 dst, const GpuOptions& opt) {
    return p_->run(&src, 1, true, dst, opt, false);
}

GpuRunResult GpuSolver::multiSource(const std::vector<u32>& srcs, const GpuOptions& opt,
                                    std::vector<std::vector<u32>>* distOut) {
    const u32 k = (u32)srcs.size();
    GpuRunResult r = p_->run(srcs.data(), k, false, 0, opt, distOut != nullptr);
    if (distOut) {
        distOut->assign(k, std::vector<u32>(p_->n));
        if (k == 1) {
            (*distOut)[0] = r.dist;
        } else {
            for (u32 i = 0; i < k; ++i)
                for (u32 v = 0; v < p_->n; ++v)
                    (*distOut)[i][v] = (u32)(p_->stateCache[(size_t)i * p_->n + v] >> 32);
            std::vector<u64>().swap(p_->stateCache);
        }
    }
    return r;
}
