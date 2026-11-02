// =============================================================================
//  gpu_kernels.cuh - Các CUDA Kernel song song (Step 2, Step 3, Step 4)
// =============================================================================
//  Kernel 1 : naiveRelaxKernel     - đồng bộ hóa mảng khoảng cách (Bellman-Ford, thread/đỉnh)
//  Kernel 2a: relaxFrontierKernel<SEL_ALL>   - duyệt biên sóng F_curr -> F_next
//  Kernel 2b: relaxFrontierKernel<SEL_LIGHT/SEL_HEAVY> + farMin/farPartition - Delta-Stepping
//  Mọi kernel có biến thể A* (ASTAR=true): cắt tỉa theo f = g + h >= dist[target].
//
//  KỸ THUẬT ATOMIC (Step 4):
//   * Trạng thái mỗi đỉnh là 1 từ 64-bit:  state = (dist << 32) | parent
//     => MỘT lệnh atomicMin 64-bit cập nhật ĐỒNG THỜI khoảng cách và đỉnh cha,
//        không bao giờ xảy ra trạng thái "dist của luồng A nhưng parent của luồng B".
//        (Khi bằng khoảng cách, đỉnh cha có chỉ số nhỏ nhất thắng -> kết quả tất định.)
//   * Read-Before-Atomic: đọc state hiện tại trước; nếu không cải thiện thì bỏ qua
//     lệnh atomic -> loại phần lớn lệnh atomic thừa, giảm tranh chấp tại L2.
//   * Warp-aggregated atomics: khi đẩy đỉnh vào hàng đợi, cả nhóm luồng đang hoạt động
//     của warp chỉ phát 1 atomicAdd lên bộ đếm đuôi hàng đợi.
//   * Bitmask khử trùng lặp: mỗi đỉnh 1 bit, atomicOr; đỉnh chỉ vào hàng đợi 1 lần/vòng.
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

#include "common.h"
#include "heuristic.h"
#include "gpu_sssp.h"
#include "cuda_check.cuh"

namespace cg = cooperative_groups;

namespace pf {

constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
constexpr u64 STATE_INIT = ((u64)INF_DIST << 32) | (u64)NO_PARENT;
constexpr u32 NONE32 = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Tiện ích trạng thái (dist, parent) đóng gói 64-bit
// ---------------------------------------------------------------------------
__device__ __forceinline__ u64 packState(u32 d, u32 p) { return ((u64)d << 32) | (u64)p; }
__device__ __forceinline__ u32 stDist(u64 s) { return (u32)(s >> 32); }
// Đọc "volatile": luôn đọc giá trị mới nhất từ L2 (bỏ qua L1 không nhất quán giữa các SM)
__device__ __forceinline__ u64 ldState(const u64* p) { return *((const volatile u64*)p); }
__device__ __forceinline__ u32 ldU32(const u32* p) { return *((const volatile u32*)p); }

// ---------------------------------------------------------------------------
// Đồ thị CSR trên thiết bị
// ---------------------------------------------------------------------------
struct DevGraph {
    const u32* row;       // row_offsets [n+1]
    const u32* col;       // col_indices [m]
    const u32* w;         // weights     [m]  (mỗi hàng đã sắp theo w tăng dần)
    const u32* lightEnd;  // [n]: cạnh nhẹ = [row[u], lightEnd[u]), cạnh nặng = [lightEnd[u], row[u+1])
    u32 n, m;
};

// Hàng đợi biên: mảng dữ liệu + bộ đếm đuôi + bitmask khử trùng lặp
struct DevQueue {
    u32* data;
    u32* count;
    u32* mask;
};

enum EdgeSel { SEL_ALL = 0, SEL_LIGHT = 1, SEL_HEAVY = 2 };

struct LocalStats {
    u32 edges = 0, atomics = 0, updates = 0, pushes = 0;
};

// Ngữ cảnh truyền cho kernel nới lỏng theo biên
struct RelaxCtx {
    DevGraph g;
    u64* state;            // [K*n] (dist<<32 | parent)
    const u32* h;          // heuristic [n] (A*) hoặc nullptr
    u32 target;            // đích (A*)
    const u32* in;         // F_curr
    u32 inSize;            // kích thước F_curr do Host truyền (khi inCount == nullptr)
    const u32* inCount;    // nếu khác nullptr: kernel tự đọc kích thước F_curr trên GPU
                           // => Host phát nhiều kernel liên tiếp mà không cần đồng bộ
    u32* inMask;           // bitmask của F_curr (bị xóa khi xử lý) hoặc nullptr
    DevQueue nearQ;        // F_next (key < bound)
    DevQueue farQ;         // "far pile" (key >= bound) - Delta-Stepping
    DevQueue rQ;           // tập R: các đỉnh đã xử lý trong bucket (để nới cạnh nặng)
    u64 bound;             // cận trên bucket hiện tại; ~0 => mọi đỉnh vào nearQ
    unsigned long long* stats;  // [5] edges, atomics, updates, pushes, số vòng có việc (hoặc nullptr)
};

// ---------------------------------------------------------------------------
// STEP 4: Cập nhật khoảng cách + cha an toàn luồng
// Trả về true nếu khoảng cách GIẢM thực sự (=> cần đưa v vào biên).
// ---------------------------------------------------------------------------
template <AtomicMode AM>
__device__ __forceinline__ bool relaxAtomic(u64* addr, u32 nd, u32 parent, LocalStats& ls) {
    const u64 nv = packState(nd, parent);
    if constexpr (AM == AtomicMode::MinNoCheck) {
        ls.atomics++;
        const u64 old = atomicMin((unsigned long long*)addr, (unsigned long long)nv);
        return stDist(old) > nd;
    } else {
        u64 cur = ldState(addr);
        if (cur <= nv) return false;                 // Read-Before-Atomic: không cải thiện -> 0 atomic
        if constexpr (AM == AtomicMode::MinRBA) {
            ls.atomics++;
            const u64 old = atomicMin((unsigned long long*)addr, (unsigned long long)nv);
            return stDist(old) > nd;
        } else {                                     // vòng lặp Compare-And-Swap
            while (true) {
                ls.atomics++;
                const u64 old = atomicCAS((unsigned long long*)addr, (unsigned long long)cur,
                                          (unsigned long long)nv);
                if (old == cur) return stDist(old) > nd;   // CAS thành công
                cur = old;                                 // luồng khác vừa ghi -> thử lại
                if (cur <= nv) return false;               // giá trị mới đã tốt hơn -> dừng
            }
        }
    }
}

// Đặt bit i trong bitmask; trả về true nếu bit trước đó = 0 (lần đầu được đưa vào hàng đợi)
__device__ __forceinline__ bool testAndSetBit(u32* mask, u32 i) {
    u32* wp = mask + (i >> 5);
    const u32 bit = 1u << (i & 31u);
    if (ldU32(wp) & bit) return false;               // đọc trước: đã có -> không cần atomic
    return (atomicOr(wp, bit) & bit) == 0u;
}
__device__ __forceinline__ void clearBit(u32* mask, u32 i) {
    atomicAnd(mask + (i >> 5), ~(1u << (i & 31u)));
}

// Warp-aggregated atomic increment: 1 atomicAdd cho cả nhóm luồng đang hoạt động
__device__ __forceinline__ u32 warpAggregatedInc(u32* ctr) {
    cg::coalesced_group active = cg::coalesced_threads();
    u32 base = 0;
    if (active.thread_rank() == 0) base = atomicAdd(ctr, (u32)active.size());
    base = active.shfl(base, 0);
    return base + active.thread_rank();
}
__device__ __forceinline__ void pushQueue(const DevQueue& q, u32 item) {
    q.data[warpAggregatedInc(q.count)] = item;
}

// Cộng dồn thống kê theo warp (shuffle reduction) rồi 1 atomicAdd / warp
__device__ __forceinline__ void flushStats(unsigned long long* stats, const LocalStats& ls) {
    unsigned long long a = ls.edges, b = ls.atomics, c = ls.updates, d = ls.pushes;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(FULL_MASK, a, off);
        b += __shfl_down_sync(FULL_MASK, b, off);
        c += __shfl_down_sync(FULL_MASK, c, off);
        d += __shfl_down_sync(FULL_MASK, d, off);
    }
    if ((threadIdx.x & 31u) == 0) {
        if (a) atomicAdd(&stats[0], a);
        if (b) atomicAdd(&stats[1], b);
        if (c) atomicAdd(&stats[2], c);
        if (d) atomicAdd(&stats[3], d);
    }
}

// ---------------------------------------------------------------------------
// Nới lỏng các cạnh e = beg, beg+stride, ... < end của đỉnh u (dist du)
//   base : offset của truy vấn trong batch (k*n), ub: cận trên A* (dist[target])
// ---------------------------------------------------------------------------
template <bool ASTAR, AtomicMode AM>
__device__ __forceinline__ void relaxRange(const RelaxCtx& c, u32 u, u32 du, u32 base, u32 ub,
                                           u32 beg, u32 end, u32 stride, LocalStats& ls) {
    for (u32 e = beg; e < end; e += stride) {
        const u32 v = c.g.col[e];
        const u64 nd = (u64)du + c.g.w[e];
        ls.edges++;
        if (nd >= INF_DIST) continue;
        u64 key = nd;
        if constexpr (ASTAR) {
            key += c.h[v];
            if (key >= ub) continue;                 // cắt tỉa A*: f(v) >= cận trên -> vô ích
        }
        const u32 gv = base + v;
        if (relaxAtomic<AM>(&c.state[gv], (u32)nd, u, ls)) {
            ls.updates++;
            if (key < c.bound) {                     // vào F_next (bucket hiện tại)
                if (testAndSetBit(c.nearQ.mask, gv)) { pushQueue(c.nearQ, gv); ls.pushes++; }
            } else {                                 // vào far pile (bucket sau)
                if (testAndSetBit(c.farQ.mask, gv)) { pushQueue(c.farQ, gv); ls.pushes++; }
            }
        }
    }
}

// ===========================================================================
// KERNEL 2 (Frontier / Delta-Stepping): mỗi phần tử của F_curr được xử lý bởi
//   - MAP=Thread : 1 luồng
//   - MAP=Warp   : cả warp (32 luồng chia nhau các cạnh)
//   - MAP=Hybrid : cân bằng tải động 3 tầng theo bậc (Merrill et al. 2012)
//        bậc >= blockDim : cả CTA (block) xử lý, phân phối qua Shared Memory
//        bậc >= 32       : cả warp xử lý (ballot + shuffle)
//        còn lại         : luồng tự xử lý
//     => tránh phân kỳ nhánh (warp divergence) khi bậc đỉnh chênh lệch lớn.
// SEL: tập cạnh (ALL / LIGHT / HEAVY). MARK_R: ghi nhận đỉnh vào tập R.
// LƯU Ý: không có lệnh return sớm -> mọi luồng tham gia __syncthreads/__shfl_sync.
// ===========================================================================
template <int SEL, bool ASTAR, AtomicMode AM, Mapping MAP, bool MARK_R>
__global__ void relaxFrontierKernel(RelaxCtx c) {
    LocalStats ls;
    // Kích thước biên: do Host truyền, hoặc đọc trực tiếp từ bộ đếm trên GPU (không cần đồng bộ Host)
    const u32 size = c.inCount ? ldU32(c.inCount) : c.inSize;
    if (c.stats && size > 0 && blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&c.stats[4], 1ull);

    // Vòng grid-stride: điều kiện lặp đồng nhất trong block (base giống nhau cho mọi luồng)
    for (u32 blockBase = blockIdx.x * blockDim.x; blockBase < size; blockBase += gridDim.x * blockDim.x) {
    const u32 idx = blockBase + threadIdx.x;
    const bool valid = idx < size;

    u32 gi = 0, u = 0, base = 0, du = INF_DIST, ub = INF_DIST, beg = 0, end = 0;
    if (valid) {
        gi = c.in[idx];
        if (c.inMask) clearBit(c.inMask, gi);    // cho phép đỉnh vào lại biên ở vòng sau
        const u32 k = gi / c.g.n;                // chỉ số truy vấn trong batch
        base = k * c.g.n;
        u = gi - base;
        du = stDist(ldState(&c.state[gi]));
        bool active = (du != INF_DIST);
        if constexpr (ASTAR) {
            ub = stDist(ldState(&c.state[c.target]));
            if (active && (u64)du + c.h[u] >= ub) active = false;
        }
        if (active) {
            if constexpr (MARK_R) {
                if (testAndSetBit(c.rQ.mask, gi)) pushQueue(c.rQ, gi);
            }
            const u32 r0 = c.g.row[u], r1 = c.g.row[u + 1];
            if constexpr (SEL == SEL_ALL)        { beg = r0;              end = r1; }
            else if constexpr (SEL == SEL_LIGHT) { beg = r0;              end = c.g.lightEnd[u]; }
            else                                 { beg = c.g.lightEnd[u]; end = r1; }
        }
    }
    bool done = (end <= beg);

    // ---------------- Tầng 1: CTA (block) xử lý đỉnh bậc rất lớn ----------------
    if constexpr (MAP == Mapping::Hybrid) {
        __shared__ u32 s_owner, s_u, s_du, s_base, s_ub, s_beg, s_end;
        while (true) {
            if (threadIdx.x == 0) s_owner = NONE32;
            __syncthreads();
            if (!done && end - beg >= blockDim.x) s_owner = threadIdx.x;   // tranh chấp lành tính: 1 luồng thắng
            __syncthreads();
            const u32 owner = s_owner;
            if (owner == NONE32) break;                                     // đồng nhất toàn block
            if (owner == threadIdx.x) {
                s_u = u; s_du = du; s_base = base; s_ub = ub; s_beg = beg; s_end = end;
                done = true;
            }
            __syncthreads();
            relaxRange<ASTAR, AM>(c, s_u, s_du, s_base, s_ub, s_beg + threadIdx.x, s_end, blockDim.x, ls);
            __syncthreads();
        }
    }

    // ---------------- Tầng 2: Warp xử lý đỉnh bậc trung bình ----------------
    if constexpr (MAP == Mapping::Hybrid || MAP == Mapping::Warp) {
        const u32 lane = threadIdx.x & 31u;
        const bool want = !done && (MAP == Mapping::Warp || (end - beg) >= 32u);
        u32 ballot = __ballot_sync(FULL_MASK, want);
        while (ballot) {
            const int leader = __ffs(ballot) - 1;
            ballot &= ballot - 1;
            const u32 wu = __shfl_sync(FULL_MASK, u, leader);
            const u32 wdu = __shfl_sync(FULL_MASK, du, leader);
            const u32 wbase = __shfl_sync(FULL_MASK, base, leader);
            const u32 wub = __shfl_sync(FULL_MASK, ub, leader);
            const u32 wbeg = __shfl_sync(FULL_MASK, beg, leader);
            const u32 wend = __shfl_sync(FULL_MASK, end, leader);
            relaxRange<ASTAR, AM>(c, wu, wdu, wbase, wub, wbeg + lane, wend, 32u, ls);
        }
        if (want) done = true;
    }

    // ---------------- Tầng 3: mỗi luồng tự xử lý đỉnh bậc nhỏ ----------------
    if (!done) relaxRange<ASTAR, AM>(c, u, du, base, ub, beg, end, 1u, ls);
    }  // hết vòng grid-stride

    if (c.stats) flushStats(c.stats, ls);
}

// ===========================================================================
// KERNEL 1: Đồng bộ hóa mảng khoảng cách (Harish & Narayanan 2007)
//   Mỗi vòng lặp: 1 luồng / đỉnh. Đỉnh u "active" (dist vừa giảm ở vòng trước)
//   nới lỏng mọi cạnh ra; đỉnh đích được cải thiện đánh dấu activeOut[v] = 1 và
//   bật cờ hội tụ toàn cục d_changed. Host lặp tới khi d_changed == 0.
//   Duyệt TOÀN BỘ n đỉnh mỗi vòng -> nhiều luồng nhàn rỗi (động lực cho Kernel 2).
// ===========================================================================
template <bool ASTAR, AtomicMode AM>
__global__ void naiveRelaxKernel(RelaxCtx c, u8* activeIn, u8* activeOut, u32* changed) {
    LocalStats ls;
    const u32 u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < c.g.n && activeIn[u]) {
        activeIn[u] = 0;
        const u32 du = stDist(ldState(&c.state[u]));
        u32 ub = INF_DIST;
        bool active = (du != INF_DIST);
        if constexpr (ASTAR) {
            ub = stDist(ldState(&c.state[c.target]));
            if (active && (u64)du + c.h[u] >= ub) active = false;
        }
        if (active) {
            const u32 r1 = c.g.row[u + 1];
            for (u32 e = c.g.row[u]; e < r1; ++e) {
                const u32 v = c.g.col[e];
                const u64 nd = (u64)du + c.g.w[e];
                ls.edges++;
                if (nd >= INF_DIST) continue;
                if constexpr (ASTAR) {
                    if (nd + c.h[v] >= ub) continue;
                }
                if (relaxAtomic<AM>(&c.state[v], (u32)nd, u, ls)) {
                    ls.updates++;
                    activeOut[v] = 1;
                    if (ldU32(changed) == 0u) *changed = 1u;   // đọc trước khi ghi: tránh ghi trùng
                }
            }
        }
    }
    if (c.stats) flushStats(c.stats, ls);
}

// ===========================================================================
// Kernel phụ trợ
// ===========================================================================
__global__ void initStateKernel(u64* state, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
        state[i] = STATE_INIT;
}

// Gieo K nguồn: state[k*n + s_k] = (0, NONE), đưa vào hàng đợi / cờ active
__global__ void seedKernel(u64* state, const u32* srcs, u32 k, u32 n, DevQueue q, u8* active) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        const u32 gi = i * n + srcs[i];
        state[gi] = packState(0u, NO_PARENT);
        if (q.data) {
            q.data[i] = gi;
            atomicOr(q.mask + (gi >> 5), 1u << (gi & 31u));
        }
        if (active) active[gi] = 1;
    }
    if (i == 0 && q.count) *q.count = k;
}

// h(v) cho A*: tính 1 lần / truy vấn, dùng chung công thức với CPU (heuristic.h)
__global__ void heuristicKernel(const double* x, const double* y, u32 n, double xt, double yt, double scale, u32* h) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) h[v] = heuristicValue(x[v], y[v], xt, yt, scale);
}

// lightEnd[u] = vị trí cạnh đầu tiên có w > Delta trong hàng u (hàng đã sắp xếp theo w)
__global__ void lightEndKernel(const u32* row, const u32* w, u32 n, u32 delta, u32* lightEnd) {
    const u32 u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) {
        u32 lo = row[u], hi = row[u + 1];
        while (lo < hi) {                       // upper_bound nhị phân
            const u32 mid = lo + ((hi - lo) >> 1);
            if (w[mid] <= delta) lo = mid + 1; else hi = mid;
        }
        lightEnd[u] = lo;
    }
}

// Delta-Stepping: tìm khóa nhỏ nhất còn hiệu lực trong far pile (warp-reduce rồi 1 atomicMin/warp)
template <bool ASTAR>
__global__ void farMinKernel(const u32* far, u32 size, const u64* state, const u32* h, u32 target,
                             u64 oldBound, unsigned long long* minKey) {
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long key = ~0ull;
    if (idx < size) {
        const u32 gi = far[idx];
        const u32 d = stDist(ldState(&state[gi]));
        if (d != INF_DIST) {
            u64 kk = d;
            bool ok = true;
            if constexpr (ASTAR) {
                kk += h[gi];
                ok = kk < (u64)stDist(ldState(&state[target]));
            }
            if (ok && kk >= oldBound) key = kk;  // kk < oldBound: đỉnh đã được xử lý trong bucket cũ
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_down_sync(FULL_MASK, key, off);
        key = o < key ? o : key;
    }
    if ((threadIdx.x & 31u) == 0 && key != ~0ull) atomicMin(minKey, key);
}

// Delta-Stepping: phân hoạch far pile -> bucket mới (nearQ) / far pile mới; loại phần tử cũ
template <bool ASTAR>
__global__ void farPartitionKernel(const u32* far, u32 size, const u64* state, const u32* h, u32 target,
                                   u64 oldBound, u64 newBound, DevQueue nearQ, DevQueue farOut) {
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const u32 gi = far[idx];
        const u32 d = stDist(ldState(&state[gi]));
        u64 key = d;
        bool ok = (d != INF_DIST);
        if constexpr (ASTAR) {
            key += h[gi];
            ok = ok && key < (u64)stDist(ldState(&state[target]));
        }
        ok = ok && key >= oldBound;
        if (!ok) {
            clearBit(farOut.mask, gi);                         // phần tử cũ / bị cắt tỉa
        } else if (key < newBound) {
            clearBit(farOut.mask, gi);                         // rời far pile -> bucket hiện tại
            if (testAndSetBit(nearQ.mask, gi)) pushQueue(nearQ, gi);
        } else {
            pushQueue(farOut, gi);                             // vẫn ở far pile (bit giữ nguyên)
        }
    }
}

}  // namespace pf
