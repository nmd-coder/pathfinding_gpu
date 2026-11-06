// =============================================================================
//  main.cpp - Chương trình thực nghiệm: kiểm thử đúng đắn + Benchmark GPU vs CPU
//  Đề tài: Thuật toán Pathfinding song song quy mô lớn trên bản đồ đồ thị
//          (GPU-Accelerated Parallel Dijkstra / A*)
// =============================================================================
//  Chế độ (--mode):
//    demo        : trọn bộ kịch bản cho video demo (info, verify, bench, atomic, delta, multisource)
//    all         : demo + mapping + transfer + R-MAT + đồ thị lớn hơn
//    info        : thông tin phần cứng
//    verify      : ma trận kiểm thử đúng đắn (mọi thuật toán x ánh xạ x chế độ atomic)
//    bench       : benchmark SSSP + A* theo nhiều kích thước đồ thị
//    atomic      : thí nghiệm giảm tranh chấp atomic (Read-Before-Atomic, CAS, no-check)
//    delta       : quét tham số Delta
//    compaction  : dựng biên kế tiếp bằng atomic gộp theo warp vs Prefix Sum (CUB)
//    sync        : số kernel phát liên tiếp giữa 2 lần đồng bộ Host-Device (giảm overhead đồng bộ)
//    mapping     : Thread / Warp / Hybrid (cân bằng tải)
//    transfer    : pageable vs pinned vs pinned+streams
//    multisource : nhiều nguồn: GPU batch vs CPU 1 luồng vs CPU OpenMP
//    single      : 1 truy vấn, in đường đi
//  Chạy "pathfinder --help" để xem toàn bộ tham số.
// =============================================================================
#include "common.h"
#include "cpu_sssp.h"
#include "gpu_sssp.h"
#include "graph.h"
#include "heuristic.h"
#include "report.h"
#include "verify.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// Cấu hình dòng lệnh
// -----------------------------------------------------------------------------
struct Config {
    std::string mode = "demo";
    std::string graph = "grid";        // grid | rmat | dimacs | snap | bin
    u64 n = 0;                         // số đỉnh (grid / rmat) cho các chế độ 1 đồ thị
    u32 ef = 8;                        // edge factor R-MAT
    std::string gr, co, file, saveBin;
    bool symmetrize = true;
    std::string family = "grid";       // bench: grid | rmat | both
    std::vector<u64> sizes;            // bench: danh sách kích thước
    u32 queries = 3;                   // số truy vấn (nguồn) mỗi đồ thị
    u64 seed = 2026;
    u32 src = 0xFFFFFFFFu, dst = 0xFFFFFFFFu;
    std::string algo = "delta";        // single: naive | frontier | delta
    bool astar = false;                // single: dùng A*
    GpuOptions gpu;
    u32 batch = 16;                    // multisource: K tối đa
    int threads = 0;                   // CPU OpenMP
    u32 samplePaths = 64;              // số đường đi truy vết mỗi truy vấn
    std::string outDir = "results";
    bool color = true;
    bool printPath = false;
};

static void usage() {
    printf(
        "Usage: pathfinder [--mode MODE] [options]\n"
        "  --mode demo|all|info|verify|bench|atomic|delta|sync|compaction|mapping|transfer|multisource|single\n"
        "Graph source (verify/atomic/delta/mapping/transfer/multisource/single):\n"
        "  --graph grid|rmat|dimacs|snap|bin   (default grid)\n"
        "  --n N              number of vertices for grid/rmat (e.g. 1e6)\n"
        "  --ef E             R-MAT edge factor (default 8)\n"
        "  --gr FILE --co FILE  DIMACS road network (.gr + optional .co coordinates)\n"
        "  --file FILE        SNAP edge list or binary cache (.bin)\n"
        "  --directed         SNAP: do not symmetrize\n"
        "  --save-bin FILE    save the loaded graph as binary cache\n"
        "Benchmark:\n"
        "  --family grid|rmat|both   --sizes 1e4,1e5,1e6   --queries Q\n"
        "GPU options:\n"
        "  --mapping thread|warp|hybrid   --atomic min|cas|nocheck   --delta D   --block B\n"
        "  --sync-every S     launch S relaxation kernels per host synchronization (default 8, 1 = classic)\n"
        "  --compaction warp|scan   next-frontier construction: warp-aggregated atomics or CUB prefix sum\n"
        "Single query:\n"
        "  --algo naive|frontier|delta  --astar  --src S --dst T  --print-path\n"
        "Other:\n"
        "  --batch K  --threads T  --seed S  --out DIR  --no-color\n");
}

static u64 parseNum(const char* s) { return (u64)std::llround(std::strtod(s, nullptr)); }

static bool parseArgs(int argc, char** argv, Config& c) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(); exit(0); }
        else if (a == "--mode") c.mode = next();
        else if (a == "--graph") c.graph = next();
        else if (a == "--n") c.n = parseNum(next());
        else if (a == "--ef") c.ef = (u32)parseNum(next());
        else if (a == "--gr") { c.gr = next(); c.graph = "dimacs"; }
        else if (a == "--co") c.co = next();
        else if (a == "--file") c.file = next();
        else if (a == "--directed") c.symmetrize = false;
        else if (a == "--save-bin") c.saveBin = next();
        else if (a == "--family") c.family = next();
        else if (a == "--sizes") {
            c.sizes.clear();
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                c.sizes.push_back(parseNum(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
        else if (a == "--queries") c.queries = std::max<u32>(1, (u32)parseNum(next()));
        else if (a == "--seed") c.seed = parseNum(next());
        else if (a == "--src") c.src = (u32)parseNum(next());
        else if (a == "--dst") c.dst = (u32)parseNum(next());
        else if (a == "--algo") c.algo = next();
        else if (a == "--astar") c.astar = true;
        else if (a == "--mapping") {
            std::string m = next();
            c.gpu.mapping = m == "thread" ? Mapping::Thread : m == "warp" ? Mapping::Warp : Mapping::Hybrid;
        }
        else if (a == "--atomic") {
            std::string m = next();
            c.gpu.atomic = m == "cas" ? AtomicMode::CasRBA : m == "nocheck" ? AtomicMode::MinNoCheck : AtomicMode::MinRBA;
        }
        else if (a == "--delta") c.gpu.delta = (u32)parseNum(next());
        else if (a == "--block") c.gpu.blockSize = (int)parseNum(next());
        else if (a == "--compaction") {
            std::string m = next();
            c.gpu.compaction = (m == "scan" || m == "prefix") ? Compaction::PrefixScan : Compaction::WarpAtomic;
        }
        else if (a == "--sync-every") c.gpu.syncEvery = std::max<u32>(1, (u32)parseNum(next()));
        else if (a == "--batch") c.batch = std::max<u32>(1, (u32)parseNum(next()));
        else if (a == "--threads") c.threads = (int)parseNum(next());
        else if (a == "--out") c.outDir = next();
        else if (a == "--no-color") c.color = false;
        else if (a == "--print-path") c.printPath = true;
        else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); usage(); return false; }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Tổng hợp kiểm thử toàn chương trình
// -----------------------------------------------------------------------------
static VerifyReport g_allChecks;
static u64 g_runsVerified = 0;

static Cell verdictCell(const VerifyReport& r) {
    if (r.pass()) return Cell("PASS " + fmt("%.0f%%", r.passRate()), term::green());
    return Cell("FAIL " + fmt("%.4f%%", r.passRate()), term::red());
}
static void account(const VerifyReport& r) {
    g_allChecks.merge(r);
    g_runsVerified++;
    if (!r.pass()) printf("    %s[verify] %s%s\n", term::red(), r.firstError.c_str(), term::reset());
}

// -----------------------------------------------------------------------------
// Đồ thị
// -----------------------------------------------------------------------------
static CSRGraph makeSynthetic(const std::string& family, u64 n, const Config& c) {
    HostTimer t; t.start();
    CSRGraph g;
    if (family == "rmat") {
        u32 scale = (u32)std::max<long long>(4, std::llround(std::log2((double)std::max<u64>(n, 16))));
        g = buildCSR(genRMAT(scale, c.ef, 1000, c.seed));
    } else {
        u32 side = (u32)std::max<long long>(2, std::llround(std::sqrt((double)n)));
        g = buildCSR(genGridRoad(side, side, c.seed));
    }
    printf("    %s(generated + CSR built in %.1f ms)%s\n", term::dim(), t.elapsedMs(), term::reset());
    return g;
}

static CSRGraph loadGraph(const Config& c, u64 defaultN) {
    CSRGraph g;
    HostTimer t; t.start();
    if (c.graph == "dimacs") {
        EdgeList el;
        if (!loadDIMACS(c.gr, c.co, el)) throw std::runtime_error("failed to load DIMACS graph");
        g = buildCSR(std::move(el));
        printf("    %s(DIMACS loaded + CSR built in %.1f ms)%s\n", term::dim(), t.elapsedMs(), term::reset());
    } else if (c.graph == "snap") {
        EdgeList el;
        if (!loadSNAP(c.file, c.symmetrize, 1000, c.seed, el)) throw std::runtime_error("failed to load SNAP graph");
        g = buildCSR(std::move(el));
        printf("    %s(SNAP loaded + CSR built in %.1f ms)%s\n", term::dim(), t.elapsedMs(), term::reset());
    } else if (c.graph == "bin") {
        if (!loadBinary(c.file, g)) throw std::runtime_error("failed to load binary graph " + c.file);
        printf("    %s(binary cache loaded in %.1f ms)%s\n", term::dim(), t.elapsedMs(), term::reset());
    } else {
        g = makeSynthetic(c.graph, c.n ? c.n : defaultN, c);
    }
    if (!c.saveBin.empty()) {
        if (saveBinary(g, c.saveBin)) printf("    saved binary cache -> %s\n", c.saveBin.c_str());
    }
    return g;
}

static u64 reachableArcs(const CSRGraph& g, const std::vector<u32>& dist) {
    u64 s = 0;
    for (u32 v = 0; v < g.n; ++v) if (dist[v] != INF_DIST) s += g.degree(v);
    return s;
}

static u32 pickReachableTarget(const std::vector<u32>& dist, Rng& rng, u32 src) {
    // ưu tiên đích xa (nửa trên theo khoảng cách) để truy vấn A* có ý nghĩa
    u32 best = src, bestD = 0;
    for (int t = 0; t < 64; ++t) {
        u32 v = rng.below((u32)dist.size());
        if (dist[v] != INF_DIST && dist[v] >= bestD) { best = v; bestD = dist[v]; }
    }
    return best;
}

static GpuOptions opts(GpuAlgo a, Mapping m, AtomicMode am, const GpuOptions& base) {
    GpuOptions o = base;
    o.algo = a; o.mapping = m; o.atomic = am;
    return o;
}

// -----------------------------------------------------------------------------
// Thông tin hệ thống (mục 3.1.1 / 3.1.2 của báo cáo)
// -----------------------------------------------------------------------------
static void printSystemInfo(const GpuDeviceInfo& d) {
    printSection("Hardware / software environment");
    Table t("");
    t.setHeader({"Item", "Value"}, "ll");
    if (d.ok) {
        t.addRow({Cell("GPU"), Cell(d.name, term::green())});
        t.addRow({Cell("Compute capability"), Cell(fmt("%.0f", d.ccMajor) + "." + fmt("%.0f", d.ccMinor))});
        t.addRow({Cell("SMs / max threads per SM"), Cell(fmt("%.0f", d.smCount) + " / " + fmt("%.0f", d.maxThreadsPerSM))});
        t.addRow({Cell("Core clock"), Cell(fmt("%.0f MHz", d.clockMHz))});
        t.addRow({Cell("Global memory (VRAM)"), Cell(fmt("%.2f GB", d.globalMem / 1073741824.0))});
        t.addRow({Cell("L2 cache / shared per block"), Cell(fmt("%.1f MB", d.l2Bytes / 1048576.0) + " / " + fmt("%.0f KB", d.sharedPerBlock / 1024.0))});
        t.addRow({Cell("Memory bus / peak bandwidth"), Cell(fmt("%.0f-bit", d.busWidth) + " / " + fmt("%.1f GB/s", d.peakBandwidthGBs))});
        t.addRow({Cell("CUDA driver / runtime"), Cell(fmt("%.0f", d.driverVersion / 1000) + "." + fmt("%.0f", (d.driverVersion % 1000) / 10) + " / " +
                                                    fmt("%.0f", d.runtimeVersion / 1000) + "." + fmt("%.0f", (d.runtimeVersion % 1000) / 10))});
    } else {
        t.addRow({Cell("GPU"), Cell("NOT FOUND", term::red())});
    }
    t.addRow({Cell("CPU hardware threads"), Cell(fmt("%.0f", (double)std::thread::hardware_concurrency()))});
    t.addRow({Cell("OpenMP threads"), Cell(fmt("%.0f", cpuMaxThreads()))});
#if defined(_MSC_VER)
    t.addRow({Cell("Host compiler"), Cell("MSVC " + fmt("%.0f", _MSC_VER))});
#elif defined(__clang__)
    t.addRow({Cell("Host compiler"), Cell(std::string("Clang ") + __clang_version__)});
#elif defined(__GNUC__)
    t.addRow({Cell("Host compiler"), Cell(std::string("GCC ") + __VERSION__)});
#endif
    t.addRow({Cell("Timing"), Cell("GPU: cudaEvent_t | CPU: std::chrono::steady_clock")});
    t.print();
}

// =============================================================================
// 1) KIỂM THỬ ĐÚNG ĐẮN: mọi biến thể GPU so với CPU Dijkstra
// =============================================================================
struct Variant {
    std::string name;
    GpuOptions o;
};

static void runVerify(const CSRGraph& g, const Config& c, GpuSolver& solver) {
    printSection("Correctness verification: " + g.name);
    std::vector<Variant> vs;
    const GpuOptions& b = c.gpu;
    vs.push_back({"K1 Naive-Relax", opts(GpuAlgo::NaiveRelax, Mapping::Thread, AtomicMode::MinRBA, b)});
    {
        GpuOptions ps = opts(GpuAlgo::Frontier, Mapping::Hybrid, AtomicMode::MinRBA, b);
        ps.compaction = Compaction::PrefixScan;
        vs.push_back({"K2 Frontier (prefix-sum)", ps});
    }
    for (Mapping m : {Mapping::Thread, Mapping::Warp, Mapping::Hybrid})
        vs.push_back({"K2 Frontier", opts(GpuAlgo::Frontier, m, AtomicMode::MinRBA, b)});
    for (Mapping m : {Mapping::Thread, Mapping::Warp, Mapping::Hybrid})
        vs.push_back({"K2 Delta-Step", opts(GpuAlgo::DeltaStep, m, AtomicMode::MinRBA, b)});
    for (AtomicMode am : {AtomicMode::CasRBA, AtomicMode::MinNoCheck}) {
        vs.push_back({"K1 Naive-Relax", opts(GpuAlgo::NaiveRelax, Mapping::Thread, am, b)});
        vs.push_back({"K2 Frontier", opts(GpuAlgo::Frontier, Mapping::Hybrid, am, b)});
        vs.push_back({"K2 Delta-Step", opts(GpuAlgo::DeltaStep, Mapping::Hybrid, am, b)});
    }
    std::vector<Variant> va;   // A*
    va.push_back({"A* Naive-Relax", opts(GpuAlgo::NaiveRelax, Mapping::Thread, AtomicMode::MinRBA, b)});
    va.push_back({"A* Frontier", opts(GpuAlgo::Frontier, Mapping::Hybrid, AtomicMode::MinRBA, b)});
    va.push_back({"A* Delta-Step", opts(GpuAlgo::DeltaStep, Mapping::Hybrid, AtomicMode::MinRBA, b)});
    va.push_back({"A* Delta-Step", opts(GpuAlgo::DeltaStep, Mapping::Thread, AtomicMode::CasRBA, b)});

    std::vector<VerifyReport> rs(vs.size()), ra(va.size());
    VerifyReport cpuAStarRep;
    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 7);
    Rng rng(c.seed + 99);
    for (u32 q = 0; q < srcs.size(); ++q) {
        const u32 s = srcs[q];
        SSSPResult ref = cpuDijkstra(g, s);
        for (size_t i = 0; i < vs.size(); ++i) {
            GpuRunResult r = solver.sssp(s, vs[i].o, true);
            VerifyReport v = verifySSSP(g, s, ref.dist, r.dist, r.parent, c.samplePaths, c.seed + q);
            rs[i].merge(v);
        }
        const u32 t = pickReachableTarget(ref.dist, rng, s);
        P2PResult ca = cpuAStar(g, s, t, true);
        cpuAStarRep.merge(verifyP2P(g, s, t, ref.dist[t], ca.dist, ca.path));
        for (size_t i = 0; i < va.size(); ++i) {
            GpuRunResult r = solver.astar(s, t, va[i].o);
            ra[i].merge(verifyP2P(g, s, t, ref.dist[t], r.targetDist, r.path));
        }
    }

    // Đa nguồn: K nguồn trong 1 lần chạy GPU, so với K lần Dijkstra CPU
    const u32 KB = 4;
    std::vector<u32> msrc = pickSources(g, KB, c.seed + 31);
    std::vector<std::vector<u32>> mref;
    for (u32 s0 : msrc) mref.push_back(cpuDijkstra(g, s0).dist);
    struct MsVariant { std::string name; GpuOptions o; VerifyReport v; };
    std::vector<MsVariant> ms;
    {
        GpuOptions f = opts(GpuAlgo::Frontier, Mapping::Hybrid, AtomicMode::MinRBA, b);
        GpuOptions fs = f; fs.compaction = Compaction::PrefixScan;
        GpuOptions d = opts(GpuAlgo::DeltaStep, Mapping::Hybrid, AtomicMode::MinRBA, b);
        ms = {{"MSSP K=4 Frontier batch", f, {}}, {"MSSP K=4 Frontier batch (prefix-sum)", fs, {}}, {"MSSP K=4 Delta-Step batch", d, {}}};
        GpuSolver bsolver(g, KB);
        for (auto& m : ms) {
            std::vector<std::vector<u32>> gd;
            bsolver.multiSource(msrc, m.o, &gd);
            for (u32 i = 0; i < KB; ++i)
                for (u32 v = 0; v < g.n; ++v) {
                    m.v.distChecked++;
                    if (gd[i][v] != mref[i][v] && m.v.distMismatch++ == 0)
                        m.v.firstError = m.name + ": dist mismatch, source " + std::to_string(msrc[i]) + " vertex " + std::to_string(v);
                }
        }
    }

    Table t("Verification matrix (GPU vs CPU std::priority_queue Dijkstra), " + fmtCount(srcs.size()) + " sources");
    t.setHeader({"Test", "Mapping", "Atomic", "Dist checks", "Tree checks", "Paths", "Errors", "Result"}, "lllrrrrl");
    auto addRow = [&](const std::string& name, const char* mp, const char* am, const VerifyReport& v) {
        t.addRow({Cell(name), Cell(mp), Cell(am), Cell(fmtCount(v.distChecked)), Cell(fmtCount(v.treeChecked)),
                  Cell(fmtCount(v.pathsChecked)), Cell(fmtCount(v.totalErrors()), v.totalErrors() ? term::red() : ""),
                  verdictCell(v)});
        account(v);
    };
    for (size_t i = 0; i < vs.size(); ++i) addRow("SSSP " + vs[i].name, toString(vs[i].o.mapping), toString(vs[i].o.atomic), rs[i]);
    t.addSeparator();
    addRow("CPU A* (reference check)", "-", "-", cpuAStarRep);
    for (size_t i = 0; i < va.size(); ++i) addRow(va[i].name, toString(va[i].o.mapping), toString(va[i].o.atomic), ra[i]);
    t.addSeparator();
    for (auto& m : ms) addRow(m.name, toString(m.o.mapping), toString(m.o.atomic), m.v);
    t.print();
}

// =============================================================================
// 2) BENCHMARK: SSSP (Dijkstra) + điểm-điểm (A*) trên 1 đồ thị
// =============================================================================
struct SummaryRow {
    std::string graph;
    u32 n, m;
    double cpuMs, naiveMs, frontierMs, deltaMs, cpuAStarMs, gpuAStarMs;
    double bestTeps;
    bool pass;
};

static void benchGraph(const CSRGraph& g, const Config& c, GpuSolver& solver, CsvWriter& csvS, CsvWriter& csvA,
                       std::vector<SummaryRow>& summary) {
    printSection("Benchmark: " + g.name + "  (n = " + fmtCount(g.n) + ", m = " + fmtCount(g.m) + ")");
    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 1);
    const GpuOptions& b = c.gpu;
    std::vector<Variant> vs = {
        {"GPU K1 Naive-Relax", opts(GpuAlgo::NaiveRelax, Mapping::Thread, b.atomic, b)},
        {"GPU K2 Frontier", opts(GpuAlgo::Frontier, b.mapping, b.atomic, b)},
        {"GPU K2 Delta-Stepping", opts(GpuAlgo::DeltaStep, b.mapping, b.atomic, b)},
    };
    // Warm-up: nạp module kernel (lazy loading) + làm nóng cache trước khi đo
    for (auto& v : vs) { solver.sssp(srcs[0], v.o, false); solver.astar(srcs[0], srcs[0], v.o); }

    const size_t NV = vs.size();
    double cpuMs = 0, cpuEdges = 0, mReached = 0;
    std::vector<double> ms(NV, 0), edges(NV, 0), iters(NV, 0), buckets(NV, 0), syncs(NV, 0);
    std::vector<VerifyReport> vr(NV);
    // A*
    double cpuAms = 0, cpuAedges = 0, cpuDijP2P = 0;
    std::vector<double> ams(NV, 0), aedges(NV, 0);
    std::vector<VerifyReport> avr(NV);
    Rng rng(c.seed + 5);

    for (u32 q = 0; q < srcs.size(); ++q) {
        const u32 s = srcs[q];
        SSSPResult ref = cpuDijkstra(g, s);
        cpuMs += ref.ms; cpuEdges += ref.edgesScanned;
        mReached += (double)reachableArcs(g, ref.dist);
        for (size_t i = 0; i < NV; ++i) {
            GpuRunResult r = solver.sssp(s, vs[i].o, true);
            ms[i] += r.ms; edges[i] += r.stats.edgesScanned; iters[i] += r.stats.iterations; buckets[i] += r.stats.buckets;
            syncs[i] += r.stats.hostSyncs;
            vr[i].merge(verifySSSP(g, s, ref.dist, r.dist, r.parent, c.samplePaths, c.seed + q));
        }
        // A* điểm - điểm
        const u32 t = pickReachableTarget(ref.dist, rng, s);
        P2PResult ca = cpuAStar(g, s, t, true);
        P2PResult cd = cpuAStar(g, s, t, false);    // Dijkstra dừng sớm (tham chiếu)
        cpuAms += ca.ms; cpuAedges += ca.edgesScanned; cpuDijP2P += cd.ms;
        for (size_t i = 0; i < NV; ++i) {
            GpuRunResult r = solver.astar(s, t, vs[i].o);
            ams[i] += r.ms; aedges[i] += r.stats.edgesScanned;
            avr[i].merge(verifyP2P(g, s, t, ref.dist[t], r.targetDist, r.path));
        }
    }
    const double Q = (double)srcs.size();
    cpuMs /= Q; cpuEdges /= Q; mReached /= Q; cpuAms /= Q; cpuAedges /= Q; cpuDijP2P /= Q;
    for (size_t i = 0; i < NV; ++i) { ms[i] /= Q; edges[i] /= Q; iters[i] /= Q; buckets[i] /= Q; syncs[i] /= Q; ams[i] /= Q; aedges[i] /= Q; }

    // ---------------- Bảng SSSP ----------------
    Table t("SSSP (single-source shortest path), average of " + fmt("%.0f", Q) + " sources | TEPS = reachable arcs / time");
    t.setHeader({"Method", "Time (ms)", "Iter", "Syncs", "Edges scanned", "Work/m", "MTEPS", "Speedup", "Verify"}, "lrrrrrrrl");
    auto teps = [&](double t_ms) { return t_ms > 0 ? mReached / (t_ms * 1e-3) : 0.0; };
    t.addRow({Cell("CPU Dijkstra (1 thread)"), Cell(fmtMs(cpuMs)), Cell("-"), Cell("-"), Cell(fmtCount((u64)cpuEdges)),
              Cell(fmt("%.2f", cpuEdges / std::max(1.0, mReached))), Cell(fmtTeps(teps(cpuMs))), Cell("1.00x"), Cell("reference")});
    csvS.row({g.name, std::to_string(g.n), std::to_string(g.m), "CPU-Dijkstra", fmt("%.4f", cpuMs), "0",
              fmt("%.0f", cpuEdges), fmt("%.4f", teps(cpuMs) / 1e6), "1", "REF", "0"});
    double best = 1e300, bestTeps = 0;
    bool pass = true;
    for (size_t i = 0; i < NV; ++i) {
        const double sp = cpuMs / std::max(1e-9, ms[i]);
        std::string iterStr = fmt("%.0f", iters[i]);
        if (vs[i].o.algo == GpuAlgo::DeltaStep) iterStr += " (" + fmt("%.0f", buckets[i]) + "b)";
        t.addRow({Cell(vs[i].name), Cell(fmtMs(ms[i])), Cell(iterStr), Cell(fmt("%.0f", syncs[i])), Cell(fmtCount((u64)edges[i])),
                  Cell(fmt("%.2f", edges[i] / std::max(1.0, mReached))), Cell(fmtTeps(teps(ms[i]))),
                  Cell(fmtSpeedup(sp), speedupColor(sp)), verdictCell(vr[i])});
        csvS.row({g.name, std::to_string(g.n), std::to_string(g.m), toString(vs[i].o.algo), fmt("%.4f", ms[i]),
                  fmt("%.1f", iters[i]), fmt("%.0f", edges[i]), fmt("%.4f", teps(ms[i]) / 1e6), fmt("%.4f", sp),
                  vr[i].pass() ? "PASS" : "FAIL", fmt("%.1f", syncs[i])});
        account(vr[i]);
        pass = pass && vr[i].pass();
        if (ms[i] < best) { best = ms[i]; bestTeps = teps(ms[i]); }
    }
    t.print();

    // ---------------- Bảng A* ----------------
    Table ta("Point-to-point A* (admissible Euclidean heuristic), average of " + fmt("%.0f", Q) + " queries");
    ta.setHeader({"Method", "Time (ms)", "Edges scanned", "Work-MTEPS", "Speedup vs CPU A*", "Verify"}, "lrrrrl");
    ta.addRow({Cell("CPU Dijkstra early-exit"), Cell(fmtMs(cpuDijP2P)), Cell("-"), Cell("-"),
               Cell(fmtSpeedup(cpuAms / std::max(1e-9, cpuDijP2P))), Cell("reference")});
    ta.addRow({Cell("CPU A* (1 thread)"), Cell(fmtMs(cpuAms)), Cell(fmtCount((u64)cpuAedges)),
               Cell(fmtTeps(cpuAedges / std::max(1e-12, cpuAms * 1e-3))), Cell("1.00x"), Cell("reference")});
    csvA.row({g.name, std::to_string(g.n), std::to_string(g.m), "CPU-Dijkstra-early-exit", fmt("%.4f", cpuDijP2P), "-", "-", "REF"});
    csvA.row({g.name, std::to_string(g.n), std::to_string(g.m), "CPU-AStar", fmt("%.4f", cpuAms), fmt("%.0f", cpuAedges), "1", "REF"});
    double bestA = 1e300;
    for (size_t i = 0; i < NV; ++i) {
        const double sp = cpuAms / std::max(1e-9, ams[i]);
        std::string nm = vs[i].name;
        nm.replace(0, 3, "GPU A*");
        ta.addRow({Cell(nm), Cell(fmtMs(ams[i])), Cell(fmtCount((u64)aedges[i])),
                   Cell(fmtTeps(aedges[i] / std::max(1e-12, ams[i] * 1e-3))), Cell(fmtSpeedup(sp), speedupColor(sp)),
                   verdictCell(avr[i])});
        csvA.row({g.name, std::to_string(g.n), std::to_string(g.m), std::string("AStar-") + toString(vs[i].o.algo),
                  fmt("%.4f", ams[i]), fmt("%.0f", aedges[i]), fmt("%.4f", sp), avr[i].pass() ? "PASS" : "FAIL"});
        account(avr[i]);
        pass = pass && avr[i].pass();
        bestA = std::min(bestA, ams[i]);
    }
    ta.print();
    summary.push_back({g.name, g.n, g.m, cpuMs, ms[0], ms[1], ms[2], cpuAms, bestA, bestTeps, pass});
}

static void printSummary(const std::vector<SummaryRow>& rows, const Config& c) {
    if (rows.empty()) return;
    Table t("SUMMARY - speedup of GPU over single-thread CPU vs graph size");
    t.setHeader({"Graph", "V", "E", "CPU Dijkstra", "GPU Naive", "GPU Frontier", "GPU Delta", "Best speedup", "Best GTEPS",
                 "CPU A*", "GPU A* best", "A* speedup", "Verify"}, "lrrrrrrrrrrrl");
    CsvWriter csv(c.outDir + "/summary_speedup.csv",
                  {"graph", "n", "m", "cpu_ms", "naive_ms", "frontier_ms", "delta_ms", "best_speedup", "best_gteps",
                   "cpu_astar_ms", "gpu_astar_ms", "astar_speedup"});
    for (auto& r : rows) {
        double bestMs = std::min({r.naiveMs, r.frontierMs, r.deltaMs});
        double sp = r.cpuMs / std::max(1e-9, bestMs);
        double spa = r.cpuAStarMs / std::max(1e-9, r.gpuAStarMs);
        t.addRow({Cell(r.graph), Cell(fmtCount(r.n)), Cell(fmtCount(r.m)), Cell(fmtMs(r.cpuMs)), Cell(fmtMs(r.naiveMs)),
                  Cell(fmtMs(r.frontierMs)), Cell(fmtMs(r.deltaMs)), Cell(fmtSpeedup(sp), speedupColor(sp)),
                  Cell(fmt("%.3f", r.bestTeps / 1e9)), Cell(fmtMs(r.cpuAStarMs)), Cell(fmtMs(r.gpuAStarMs)),
                  Cell(fmtSpeedup(spa), speedupColor(spa)),
                  r.pass ? Cell("PASS", term::green()) : Cell("FAIL", term::red())});
        csv.row({r.graph, std::to_string(r.n), std::to_string(r.m), fmt("%.4f", r.cpuMs), fmt("%.4f", r.naiveMs),
                 fmt("%.4f", r.frontierMs), fmt("%.4f", r.deltaMs), fmt("%.4f", sp), fmt("%.5f", r.bestTeps / 1e9),
                 fmt("%.4f", r.cpuAStarMs), fmt("%.4f", r.gpuAStarMs), fmt("%.4f", spa)});
    }
    t.print();
}

static void runBenchSizes(const Config& c, const std::vector<std::string>& families, const std::vector<u64>& sizes,
                          std::vector<SummaryRow>& summary) {
    CsvWriter csvS(c.outDir + "/bench_sssp.csv",
                   {"graph", "n", "m", "method", "time_ms", "iterations", "edges_scanned", "mteps", "speedup", "verify", "host_syncs"});
    CsvWriter csvA(c.outDir + "/bench_astar.csv",
                   {"graph", "n", "m", "method", "time_ms", "edges_scanned", "speedup_vs_cpu_astar", "verify"});
    for (const auto& fam : families)
        for (u64 n : sizes) {
            printSection("Building graph: " + fam + ", ~" + fmtCount(n) + " vertices");
            CSRGraph g = makeSynthetic(fam, n, c);
            printGraphInfo(g);
            GpuSolver solver(g, 1);
            printf("    VRAM allocated: %.1f MB, graph upload (pinned + 3 streams): %.2f ms\n",
                   solver.deviceBytes() / 1048576.0, solver.uploadMs());
            benchGraph(g, c, solver, csvS, csvA, summary);
        }
}

// =============================================================================
// 3) THÍ NGHIỆM ATOMIC (3.3.2): đếm số lệnh atomic thực sự phát ra
// =============================================================================
static void runAtomicStudy(const CSRGraph& g, const Config& c, GpuSolver& solver) {
    printSection("Atomic contention study: " + g.name);
    CsvWriter csv(c.outDir + "/study_atomic.csv",
                  {"graph", "algorithm", "atomic_mode", "time_ms", "atomics", "updates", "atomics_per_update", "edges_scanned", "verify"});
    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 3);
    std::vector<SSSPResult> refs;
    for (u32 s : srcs) refs.push_back(cpuDijkstra(g, s));
    Table t("Atomic strategies (Read-Before-Atomic vs blind atomicMin vs CAS loop), average of " + fmtCount(srcs.size()) + " sources");
    t.setHeader({"Algorithm", "Atomic mode", "Time (ms)", "Atomics issued", "Real updates", "Atomics/update",
                 "Atomics saved", "Verify"}, "llrrrrrl");
    for (GpuAlgo a : {GpuAlgo::NaiveRelax, GpuAlgo::Frontier, GpuAlgo::DeltaStep}) {
        double noCheckAtomics = 0;
        for (AtomicMode am : {AtomicMode::MinNoCheck, AtomicMode::MinRBA, AtomicMode::CasRBA}) {
            GpuOptions o = opts(a, a == GpuAlgo::NaiveRelax ? Mapping::Thread : c.gpu.mapping, am, c.gpu);
            solver.sssp(srcs[0], o, false);   // warm-up
            double ms = 0, at = 0, up = 0, ed = 0;
            VerifyReport v;
            for (size_t q = 0; q < srcs.size(); ++q) {
                GpuRunResult r = solver.sssp(srcs[q], o, true);
                ms += r.ms; at += r.stats.atomicsIssued; up += r.stats.updates; ed += r.stats.edgesScanned;
                v.merge(verifySSSP(g, srcs[q], refs[q].dist, r.dist, r.parent, 16, q));
            }
            const double Q = (double)srcs.size();
            ms /= Q; at /= Q; up /= Q; ed /= Q;
            if (am == AtomicMode::MinNoCheck) noCheckAtomics = at;
            const double saved = noCheckAtomics > 0 ? 100.0 * (1.0 - at / noCheckAtomics) : 0.0;
            t.addRow({Cell(toString(a)), Cell(toString(am)), Cell(fmtMs(ms)), Cell(fmtCount((u64)at)), Cell(fmtCount((u64)up)),
                      Cell(fmt("%.2f", at / std::max(1.0, up))),
                      Cell(am == AtomicMode::MinNoCheck ? "baseline" : fmt("%.1f%%", saved), am == AtomicMode::MinNoCheck ? "" : term::green()),
                      verdictCell(v)});
            csv.row({g.name, toString(a), toString(am), fmt("%.4f", ms), fmt("%.0f", at), fmt("%.0f", up),
                     fmt("%.4f", at / std::max(1.0, up)), fmt("%.0f", ed), v.pass() ? "PASS" : "FAIL"});
            account(v);
        }
        t.addSeparator();
    }
    t.print();
}

// =============================================================================
// 4) QUÉT THAM SỐ DELTA (3.3.3)
// =============================================================================
static void runDeltaStudy(const CSRGraph& g, const Config& c, GpuSolver& solver, bool quick) {
    printSection("Delta sweep: " + g.name + "  (avg weight " + fmt("%.1f", g.avgW) + ", avg degree " + fmt("%.2f", g.avgDeg) + ")");
    CsvWriter csv(c.outDir + "/study_delta_" + g.name + ".csv",
                  {"graph", "delta", "delta_over_avgw", "time_ms", "buckets", "iterations", "edges_scanned", "work_ratio", "verify"});
    std::vector<double> mult = quick ? std::vector<double>{0.5, 2, 8, 32, 128, 1024}
                                     : std::vector<double>{0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 512, 4096};
    std::vector<u32> deltas;
    for (double m : mult) deltas.push_back((u32)std::max(1.0, std::round(m * g.avgW)));
    deltas.push_back(solver.autoDelta());
    std::sort(deltas.begin(), deltas.end());
    deltas.erase(std::unique(deltas.begin(), deltas.end()), deltas.end());

    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 11);
    std::vector<SSSPResult> refs;
    double mReached = 0;
    for (u32 s : srcs) { refs.push_back(cpuDijkstra(g, s)); mReached += (double)reachableArcs(g, refs.back().dist); }
    mReached /= srcs.size();

    struct Row { u32 d; double ms, b, it, ed; VerifyReport v; };
    std::vector<Row> rows;
    for (u32 d : deltas) {
        GpuOptions o = opts(GpuAlgo::DeltaStep, c.gpu.mapping, c.gpu.atomic, c.gpu);
        o.delta = d;
        solver.sssp(srcs[0], o, false);  // warm-up + tính lightEnd
        Row r{d, 0, 0, 0, 0, {}};
        for (size_t q = 0; q < srcs.size(); ++q) {
            GpuRunResult x = solver.sssp(srcs[q], o, true);
            r.ms += x.ms; r.b += x.stats.buckets; r.it += x.stats.iterations; r.ed += x.stats.edgesScanned;
            r.v.merge(verifySSSP(g, srcs[q], refs[q].dist, x.dist, x.parent, 16, q));
        }
        const double Q = (double)srcs.size();
        r.ms /= Q; r.b /= Q; r.it /= Q; r.ed /= Q;
        rows.push_back(r);
    }
    double bestMs = 1e300;
    for (auto& r : rows) bestMs = std::min(bestMs, r.ms);
    Table t("Delta-Stepping parameter sweep (small Delta -> Dijkstra-like, huge Delta -> Bellman-Ford-like)");
    t.setHeader({"Delta", "Delta/avgW", "Time (ms)", "Buckets", "Kernel iters", "Edges scanned", "Work/m", "Verify"}, "rrrrrrrl");
    for (auto& r : rows) {
        std::string dl = fmtCount(r.d) + (r.d == solver.autoDelta() ? " (auto)" : "");
        t.addRow({Cell(dl), Cell(fmt("%.2f", r.d / std::max(1.0, g.avgW))),
                  Cell(fmtMs(r.ms), r.ms == bestMs ? term::green() : ""), Cell(fmt("%.0f", r.b)), Cell(fmt("%.0f", r.it)),
                  Cell(fmtCount((u64)r.ed)), Cell(fmt("%.2f", r.ed / std::max(1.0, mReached))), verdictCell(r.v)});
        csv.row({g.name, std::to_string(r.d), fmt("%.4f", r.d / std::max(1.0, g.avgW)), fmt("%.4f", r.ms), fmt("%.1f", r.b),
                 fmt("%.1f", r.it), fmt("%.0f", r.ed), fmt("%.4f", r.ed / std::max(1.0, mReached)), r.v.pass() ? "PASS" : "FAIL"});
        account(r.v);
    }
    t.print();
}

// =============================================================================
// 4b) GIẢM ĐỒNG BỘ HOST-DEVICE: phát S kernel liên tiếp, kernel tự đọc kích thước biên
// =============================================================================
static void runSyncStudy(const CSRGraph& g, const Config& c, GpuSolver& solver) {
    printSection("Host-device synchronization study: " + g.name);
    CsvWriter csv(c.outDir + "/study_sync_" + g.name + ".csv",
                  {"graph", "algorithm", "query", "sync_every", "time_ms", "iterations", "launches", "host_syncs",
                   "speedup_vs_s1", "verify"});
    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 17);
    std::vector<SSSPResult> refs;
    std::vector<u32> tgts;
    Rng rng(c.seed + 19);
    for (u32 s : srcs) {
        refs.push_back(cpuDijkstra(g, s));
        tgts.push_back(pickReachableTarget(refs.back().dist, rng, s));
    }
    Table t("Kernels launched per host synchronization (S = 1: classic loop, sync after every iteration)");
    t.setHeader({"Algorithm", "Query", "S", "Time (ms)", "Iterations", "Launches", "Host syncs", "vs S=1", "Verify"}, "llrrrrrrl");
    struct Cfg { GpuAlgo a; bool astar; };
    for (Cfg cf : {Cfg{GpuAlgo::Frontier, false}, Cfg{GpuAlgo::DeltaStep, false}, Cfg{GpuAlgo::DeltaStep, true}}) {
        double base = 0;
        for (u32 S : {1u, 2u, 4u, 8u, 16u, 32u}) {
            GpuOptions o = opts(cf.a, c.gpu.mapping, c.gpu.atomic, c.gpu);
            o.syncEvery = S;
            if (cf.astar) solver.astar(srcs[0], tgts[0], o); else solver.sssp(srcs[0], o, false);   // warm-up
            double ms = 0, it = 0, la = 0, sy = 0;
            VerifyReport v;
            for (size_t q = 0; q < srcs.size(); ++q) {
                GpuRunResult r = cf.astar ? solver.astar(srcs[q], tgts[q], o) : solver.sssp(srcs[q], o, true);
                ms += r.ms; it += r.stats.iterations; la += r.stats.launches; sy += r.stats.hostSyncs;
                if (cf.astar) v.merge(verifyP2P(g, srcs[q], tgts[q], refs[q].dist[tgts[q]], r.targetDist, r.path));
                else v.merge(verifySSSP(g, srcs[q], refs[q].dist, r.dist, r.parent, 16, q));
            }
            const double Q = (double)srcs.size();
            ms /= Q; it /= Q; la /= Q; sy /= Q;
            if (S == 1) base = ms;
            const double sp = base / std::max(1e-9, ms);
            t.addRow({Cell(toString(cf.a)), Cell(cf.astar ? "A* p2p" : "SSSP"), Cell(fmtCount(S)), Cell(fmtMs(ms)),
                      Cell(fmt("%.0f", it)), Cell(fmt("%.0f", la)), Cell(fmt("%.0f", sy)),
                      Cell(fmtSpeedup(sp), speedupColor(sp)), verdictCell(v)});
            csv.row({g.name, toString(cf.a), cf.astar ? "astar" : "sssp", std::to_string(S), fmt("%.4f", ms),
                     fmt("%.1f", it), fmt("%.1f", la), fmt("%.1f", sy), fmt("%.4f", sp), v.pass() ? "PASS" : "FAIL"});
            account(v);
        }
        t.addSeparator();
    }
    t.print();
}

// =============================================================================
// 4c) NÉN DANH SÁCH BIÊN (2.3.3): atomic gộp theo warp vs Prefix Sum (CUB ExclusiveSum)
// =============================================================================
static void runCompactionStudy(const CSRGraph& g, const Config& c, GpuSolver& solver) {
    printSection("Frontier compaction study: " + g.name);
    CsvWriter csv(c.outDir + "/study_compaction_" + g.name + ".csv",
                  {"graph", "compaction", "time_ms", "iterations", "launches", "pushes", "edges_scanned", "speedup_vs_warp", "verify"});
    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 23);
    std::vector<SSSPResult> refs;
    for (u32 s0 : srcs) refs.push_back(cpuDijkstra(g, s0));
    Table t("Building F_next: bitmask + warp-aggregated atomic queue vs flags + Prefix Sum + scatter (Frontier, SSSP)");
    t.setHeader({"Compaction", "Time (ms)", "Iterations", "Kernel launches", "Frontier writes", "Edges scanned", "vs warp", "Verify"}, "lrrrrrrl");
    double base = 0;
    for (Compaction cm : {Compaction::WarpAtomic, Compaction::PrefixScan}) {
        GpuOptions o = opts(GpuAlgo::Frontier, c.gpu.mapping, c.gpu.atomic, c.gpu);
        o.compaction = cm;
        solver.sssp(srcs[0], o, false);   // warm-up
        double ms = 0, it = 0, la = 0, pu = 0, ed = 0;
        VerifyReport v;
        for (size_t q = 0; q < srcs.size(); ++q) {
            GpuRunResult r = solver.sssp(srcs[q], o, true);
            ms += r.ms; it += r.stats.iterations; la += r.stats.launches; pu += r.stats.pushes; ed += r.stats.edgesScanned;
            v.merge(verifySSSP(g, srcs[q], refs[q].dist, r.dist, r.parent, 16, q));
        }
        const double Q = (double)srcs.size();
        ms /= Q; it /= Q; la /= Q; pu /= Q; ed /= Q;
        if (cm == Compaction::WarpAtomic) base = ms;
        const double sp = base / std::max(1e-9, ms);
        t.addRow({Cell(toString(cm)), Cell(fmtMs(ms)), Cell(fmt("%.0f", it)), Cell(fmt("%.0f", la)), Cell(fmtCount((u64)pu)),
                  Cell(fmtCount((u64)ed)), Cell(fmtSpeedup(sp), speedupColor(sp)), verdictCell(v)});
        csv.row({g.name, toString(cm), fmt("%.4f", ms), fmt("%.1f", it), fmt("%.1f", la), fmt("%.0f", pu), fmt("%.0f", ed),
                 fmt("%.4f", sp), v.pass() ? "PASS" : "FAIL"});
        account(v);
    }
    t.print();
}

// =============================================================================
// 5) ÁNH XẠ LUỒNG / CÂN BẰNG TẢI (2.2.3, 2.4.3)
// =============================================================================
static void runMappingStudy(const CSRGraph& g, const Config& c, GpuSolver& solver, CsvWriter& csv) {
    printSection("Thread mapping / load balancing: " + g.name + "  (max degree " + fmtCount(g.maxDeg) + ")");
    std::vector<u32> srcs = pickSources(g, c.queries, c.seed + 13);
    std::vector<SSSPResult> refs;
    for (u32 s : srcs) refs.push_back(cpuDijkstra(g, s));
    Table t("Thread-per-vertex vs Warp-per-vertex vs Hybrid (CTA/Warp/Thread)");
    t.setHeader({"Algorithm", "Mapping", "Time (ms)", "vs thread", "Verify"}, "llrrl");
    for (GpuAlgo a : {GpuAlgo::Frontier, GpuAlgo::DeltaStep}) {
        double base = 0;
        for (Mapping m : {Mapping::Thread, Mapping::Warp, Mapping::Hybrid}) {
            GpuOptions o = opts(a, m, c.gpu.atomic, c.gpu);
            solver.sssp(srcs[0], o, false);
            double ms = 0;
            VerifyReport v;
            for (size_t q = 0; q < srcs.size(); ++q) {
                GpuRunResult r = solver.sssp(srcs[q], o, true);
                ms += r.ms;
                v.merge(verifySSSP(g, srcs[q], refs[q].dist, r.dist, r.parent, 16, q));
            }
            ms /= srcs.size();
            if (m == Mapping::Thread) base = ms;
            const double sp = base / std::max(1e-9, ms);
            t.addRow({Cell(toString(a)), Cell(toString(m)), Cell(fmtMs(ms)), Cell(fmtSpeedup(sp), speedupColor(sp)), verdictCell(v)});
            csv.row({g.name, toString(a), toString(m), fmt("%.4f", ms), fmt("%.4f", sp), v.pass() ? "PASS" : "FAIL"});
            account(v);
        }
        t.addSeparator();
    }
    t.print();
}

// =============================================================================
// 6) TRUYỀN DỮ LIỆU HOST -> DEVICE (2.4.4)
// =============================================================================
static void runTransfer(const CSRGraph& g, const Config& c) {
    printSection("Host -> Device transfer of CSR arrays: " + g.name);
    TransferStudy r = gpuTransferStudy(g, 5);
    CsvWriter csv(c.outDir + "/study_transfer.csv", {"graph", "bytes", "method", "time_ms", "gbps"});
    Table t("CSR upload (" + fmt("%.1f MB", r.bytes / 1048576.0) + "), average of 5 runs");
    t.setHeader({"Method", "Time (ms)", "Bandwidth (GB/s)", "vs pageable"}, "lrrr");
    auto add = [&](const char* nm, double ms) {
        double bw = r.bytes / (ms * 1e-3) / 1e9;
        double sp = r.pageableMs / std::max(1e-9, ms);
        t.addRow({Cell(nm), Cell(fmtMs(ms)), Cell(fmt("%.2f", bw)), Cell(fmtSpeedup(sp), speedupColor(sp))});
        csv.row({g.name, fmt("%.0f", r.bytes), nm, fmt("%.4f", ms), fmt("%.4f", bw)});
    };
    add("cudaMemcpy (pageable)", r.pageableMs);
    add("cudaMemcpyAsync (pinned)", r.pinnedMs);
    add("pinned + 3 CUDA streams (8MB chunks)", r.pinnedStreamsMs);
    t.print();
}

// =============================================================================
// 7) NHIỀU NGUỒN (3.4.2): GPU batch vs CPU 1 luồng vs CPU OpenMP
// =============================================================================
static void runMultiSource(const CSRGraph& g, const Config& c) {
    printSection("Multi-source pathfinding: " + g.name);
    u32 Kmax = c.batch;
    std::unique_ptr<GpuSolver> solver;
    while (!solver) {
        try { solver.reset(new GpuSolver(g, Kmax)); }
        catch (const std::exception& e) {
            if (Kmax <= 1) throw;
            printf("    %s(%s -> batch %u -> %u)%s\n", term::yellow(), e.what(), Kmax, Kmax / 2, term::reset());
            Kmax /= 2;
        }
    }
    const int T = c.threads > 0 ? c.threads : cpuMaxThreads();
    CsvWriter csv(c.outDir + "/study_multisource_" + g.name + ".csv",
                  {"graph", "K", "cpu_1t_ms", "cpu_omp_ms", "omp_threads", "gpu_batch_ms", "gpu_batch_delta_ms",
                   "gpu_seq_delta_ms", "speedup_vs_1t", "speedup_vs_omp", "verify"});
    Table t("K sources: CPU Dijkstra x K (1 thread / OpenMP " + fmt("%.0f", T) + " threads) vs GPU (batch = K sources in one run)");
    t.setHeader({"K", "CPU 1T (ms)", "CPU OMP (ms)", "GPU batch Frontier", "GPU batch Delta", "GPU K x Delta",
                 "vs CPU 1T", "vs CPU OMP", "Queries/s (GPU)", "Verify"}, "rrrrrrrrrl");
    GpuOptions ob = opts(GpuAlgo::Frontier, c.gpu.mapping, c.gpu.atomic, c.gpu);
    GpuOptions od = opts(GpuAlgo::DeltaStep, c.gpu.mapping, c.gpu.atomic, c.gpu);
    {   // warm-up
        std::vector<u32> w = pickSources(g, 2, 1);
        solver->multiSource(w, ob, nullptr);
        solver->multiSource(w, od, nullptr);
        solver->sssp(w[0], od, false);
    }
    auto checkAll = [&](const std::vector<std::vector<u32>>& gd, const MultiSourceCpuResult& ref,
                        const std::vector<u32>& srcs, const char* what, VerifyReport& v) {
        for (u32 i = 0; i < srcs.size(); ++i)
            for (u32 x = 0; x < g.n; ++x) {
                v.distChecked++;
                if (gd[i][x] != ref.dist[i][x] && v.distMismatch++ == 0)
                    v.firstError = std::string(what) + ": dist mismatch at source " + std::to_string(srcs[i]);
            }
    };
    for (u32 K = 1; K <= Kmax; K *= 2) {
        std::vector<u32> srcs = pickSources(g, K, c.seed + 100 + K);
        MultiSourceCpuResult c1 = cpuMultiSource(g, srcs, 1, false);
        MultiSourceCpuResult cm = cpuMultiSource(g, srcs, T, true);
        VerifyReport v;
        std::vector<std::vector<u32>> gd;
        GpuRunResult gb = solver->multiSource(srcs, ob, &gd);
        checkAll(gd, cm, srcs, "batch Frontier", v);
        GpuRunResult gdb = solver->multiSource(srcs, od, &gd);
        checkAll(gd, cm, srcs, "batch Delta", v);
        double seq = 0;
        for (u32 s : srcs) seq += solver->sssp(s, od, false).ms;
        const double bestGpu = std::min({gb.ms, gdb.ms, seq});
        const double s1 = c1.ms / bestGpu, sm = cm.ms / bestGpu;
        auto best = [&](double x) { return x == bestGpu ? term::green() : ""; };
        t.addRow({Cell(fmtCount(K)), Cell(fmtMs(c1.ms)), Cell(fmtMs(cm.ms)), Cell(fmtMs(gb.ms), best(gb.ms)),
                  Cell(fmtMs(gdb.ms), best(gdb.ms)), Cell(fmtMs(seq), best(seq)), Cell(fmtSpeedup(s1), speedupColor(s1)),
                  Cell(fmtSpeedup(sm), speedupColor(sm)), Cell(fmt("%.1f", K / (bestGpu * 1e-3))), verdictCell(v)});
        csv.row({g.name, std::to_string(K), fmt("%.4f", c1.ms), fmt("%.4f", cm.ms), std::to_string(T), fmt("%.4f", gb.ms),
                 fmt("%.4f", gdb.ms), fmt("%.4f", seq), fmt("%.4f", s1), fmt("%.4f", sm), v.pass() ? "PASS" : "FAIL"});
        account(v);
    }
    t.print();
}

// =============================================================================
// 8) MỘT TRUY VẤN (demo truy vết đường đi)
// =============================================================================
static void runSingle(const CSRGraph& g, const Config& c, GpuSolver& solver) {
    GpuOptions o = c.gpu;
    o.algo = c.algo == "naive" ? GpuAlgo::NaiveRelax : c.algo == "frontier" ? GpuAlgo::Frontier : GpuAlgo::DeltaStep;
    std::vector<u32> rs = pickSources(g, 2, c.seed + 21);
    const u32 s = c.src < g.n ? c.src : rs[0];
    printSection(std::string("Single query on ") + g.name + " with GPU " + toString(o.algo) + (c.astar ? " (A*)" : " (SSSP)"));
    SSSPResult ref = cpuDijkstra(g, s);
    Rng rng(c.seed);
    const u32 t = c.dst < g.n ? c.dst : pickReachableTarget(ref.dist, rng, s);
    solver.sssp(s, o, false);  // warm-up
    std::vector<u32> path;
    u32 dist;
    double gms;
    if (c.astar) {
        GpuRunResult r = solver.astar(s, t, o);
        path = r.path; dist = r.targetDist; gms = r.ms;
        P2PResult ca = cpuAStar(g, s, t, true);
        printf("  CPU A*  : dist = %u, %.3f ms, %s edges scanned\n", ca.dist, ca.ms, fmtCount(ca.edgesScanned).c_str());
        account(verifyP2P(g, s, t, ref.dist[t], dist, path));
    } else {
        GpuRunResult r = solver.sssp(s, o, true);
        dist = r.dist[t]; gms = r.ms;
        path = reconstructPath(r.parent, s, t);
        printf("  CPU Dijkstra: %.3f ms | GPU iterations %u, buckets %u, delta %u\n", ref.ms, r.stats.iterations, r.stats.buckets, r.stats.delta);
        VerifyReport v = verifySSSP(g, s, ref.dist, r.dist, r.parent, c.samplePaths, 1);
        account(v);
        printf("  Full SSSP verification: %s%s%s\n", v.pass() ? term::green() : term::red(), v.pass() ? "PASS 100%" : v.firstError.c_str(), term::reset());
    }
    printf("  Query   : %u -> %u\n", s, t);
    printf("  GPU     : dist = %u (CPU Dijkstra = %u) -> %s%s%s, %.3f ms\n", dist, ref.dist[t],
           dist == ref.dist[t] ? term::green() : term::red(), dist == ref.dist[t] ? "MATCH" : "MISMATCH", term::reset(), gms);
    printf("  Path    : %zu vertices", path.size());
    if (!path.empty()) {
        printf("  [");
        const size_t L = path.size();
        for (size_t i = 0; i < L; ++i) {
            if (!c.printPath && L > 12 && i == 6) { printf(" ... "); i = L - 6; }
            printf("%s%u", i ? " -> " : "", path[i]);
        }
        printf("]");
    }
    printf("\n");
}

// =============================================================================
static void finalVerdict() {
    printBanner("FINAL VERIFICATION SUMMARY");
    const bool ok = g_allChecks.pass();
    printf("  GPU runs verified       : %s\n", fmtCount(g_runsVerified).c_str());
    printf("  Distance comparisons    : %s\n", fmtCount(g_allChecks.distChecked).c_str());
    printf("  Parent-tree checks      : %s\n", fmtCount(g_allChecks.treeChecked).c_str());
    printf("  Reconstructed paths     : %s\n", fmtCount(g_allChecks.pathsChecked).c_str());
    printf("  Errors                  : %s%s%s\n", ok ? term::green() : term::red(), fmtCount(g_allChecks.totalErrors()).c_str(), term::reset());
    printf("  %s%s>>> %s %s <<<%s\n\n", term::bold(), ok ? term::green() : term::red(), ok ? "PASS" : "FAIL",
           fmt("%.2f%%", g_allChecks.passRate()).c_str(), term::reset());
}

int main(int argc, char** argv) {
    Config c;
    if (!parseArgs(argc, argv, c)) return 1;
    term::init(c.color);
    printBanner("GPU-Accelerated Parallel Dijkstra / A*  |  CSR + Frontier + Delta-Stepping + Atomics");

    try {
        GpuDeviceInfo dev = gpuQueryDevice(0);
        if (!dev.ok) {
            printf("%sNo CUDA-capable GPU detected. Check the NVIDIA driver.%s\n", term::red(), term::reset());
            return 2;
        }
        gpuWarmup();
        const std::string& m = c.mode;
        if (m == "info") {
            printSystemInfo(dev);
            return 0;
        }
        if (m == "demo" || m == "all") {
            printSystemInfo(dev);
            const bool all = (m == "all");
            // (1) Kiểm thử đúng đắn trên 2 họ đồ thị
            {
                CSRGraph g1 = makeSynthetic("grid", 100000, c);
                printGraphInfo(g1);
                GpuSolver s1(g1, 1);
                runVerify(g1, c, s1);
                CSRGraph g2 = makeSynthetic("rmat", all ? (1u << 17) : (1u << 15), c);
                printGraphInfo(g2);
                GpuSolver s2(g2, 1);
                runVerify(g2, c, s2);
            }
            // (2) Benchmark theo kích thước
            std::vector<SummaryRow> summary;
            std::vector<u64> sizes = !c.sizes.empty() ? c.sizes
                                   : (all ? std::vector<u64>{10000, 100000, 1000000, 4000000} : std::vector<u64>{10000, 100000, 1000000});
            std::vector<std::string> fams = all ? std::vector<std::string>{"grid", "rmat"} : std::vector<std::string>{"grid"};
            runBenchSizes(c, fams, sizes, summary);
            printSummary(summary, c);
            // (3) Atomic + Delta trên đồ thị 1 triệu đỉnh
            {
                CSRGraph g = makeSynthetic("grid", 1000000, c);
                GpuSolver s(g, 1);
                runAtomicStudy(g, c, s);
                runSyncStudy(g, c, s);
                runCompactionStudy(g, c, s);
                runDeltaStudy(g, c, s, !all);
                if (all) {
                    CsvWriter csv(c.outDir + "/study_mapping.csv", {"graph", "algorithm", "mapping", "time_ms", "speedup_vs_thread", "verify"});
                    runMappingStudy(g, c, s, csv);
                    CSRGraph r = makeSynthetic("rmat", 1u << 20, c);
                    printGraphInfo(r);
                    GpuSolver sr(r, 1);
                    runMappingStudy(r, c, sr, csv);
                    runDeltaStudy(r, c, sr, false);
                    runTransfer(g, c);
                    runTransfer(r, c);
                }
            }
            // (4) Nhiều nguồn
            {
                CSRGraph g = makeSynthetic("grid", all ? 1000000 : 250000, c);
                printGraphInfo(g);
                runMultiSource(g, c);
            }
            finalVerdict();
            printf("  CSV results written to: %s/\n\n", c.outDir.c_str());
            return g_allChecks.pass() ? 0 : 3;
        }
        if (m == "bench") {
            std::vector<SummaryRow> summary;
            if (c.graph == "dimacs" || c.graph == "snap" || c.graph == "bin") {
                CSRGraph g = loadGraph(c, 0);
                printGraphInfo(g);
                GpuSolver solver(g, 1);
                CsvWriter csvS(c.outDir + "/bench_sssp_" + g.name + ".csv",
                               {"graph", "n", "m", "method", "time_ms", "iterations", "edges_scanned", "mteps", "speedup", "verify", "host_syncs"});
                CsvWriter csvA(c.outDir + "/bench_astar_" + g.name + ".csv",
                               {"graph", "n", "m", "method", "time_ms", "edges_scanned", "speedup_vs_cpu_astar", "verify"});
                benchGraph(g, c, solver, csvS, csvA, summary);
            } else {
                std::vector<u64> sizes = c.sizes.empty() ? std::vector<u64>{10000, 100000, 1000000} : c.sizes;
                std::vector<std::string> fams = c.family == "both" ? std::vector<std::string>{"grid", "rmat"}
                                                                   : std::vector<std::string>{c.family};
                runBenchSizes(c, fams, sizes, summary);
            }
            printSummary(summary, c);
            finalVerdict();
            return g_allChecks.pass() ? 0 : 3;
        }

        // Các chế độ làm việc trên 1 đồ thị
        CSRGraph g = loadGraph(c, m == "multisource" ? 250000 : 1000000);
        printGraphInfo(g);
        if (m == "multisource") {
            runMultiSource(g, c);
        } else if (m == "transfer") {
            runTransfer(g, c);
        } else {
            GpuSolver solver(g, 1);
            printf("    VRAM allocated: %.1f MB, graph upload: %.2f ms, auto Delta = %u\n",
                   solver.deviceBytes() / 1048576.0, solver.uploadMs(), solver.autoDelta());
            if (m == "verify") runVerify(g, c, solver);
            else if (m == "atomic") runAtomicStudy(g, c, solver);
            else if (m == "delta") runDeltaStudy(g, c, solver, false);
            else if (m == "sync") runSyncStudy(g, c, solver);
            else if (m == "compaction") runCompactionStudy(g, c, solver);
            else if (m == "mapping") {
                CsvWriter csv(c.outDir + "/study_mapping_" + g.name + ".csv", {"graph", "algorithm", "mapping", "time_ms", "speedup_vs_thread", "verify"});
                runMappingStudy(g, c, solver, csv);
            } else if (m == "single") runSingle(g, c, solver);
            else { fprintf(stderr, "Unknown mode '%s'\n", m.c_str()); usage(); return 1; }
        }
        finalVerdict();
        return g_allChecks.pass() ? 0 : 3;
    } catch (const std::exception& e) {
        fprintf(stderr, "%sERROR: %s%s\n", term::red(), e.what(), term::reset());
        return 1;
    }
}
