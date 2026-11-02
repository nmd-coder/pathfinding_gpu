// =============================================================================
//  pfshim_core.h - Mô phỏng (emulation) một tập con CUDA trên CPU để kiểm thử
//  LOGIC thuật toán khi máy không có GPU NVIDIA. KHÔNG dùng để đo hiệu năng.
//   * Mỗi CUDA thread của một block = 1 std::thread thật (chạy đồng thời)
//   * __syncthreads / __ballot_sync / __shfl_sync dùng std::barrier (C++20)
//   * Các block chạy tuần tự; atomic* dùng __atomic builtins của GCC/Clang
//   * coalesced_threads() mô phỏng nhóm 1 luồng (không gộp atomic, vẫn đúng ngữ nghĩa)
// =============================================================================
#pragma once
#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __launch_bounds__(...)

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
struct pf_uint3 { unsigned x, y, z; };

namespace pfshim {
struct Block {
    explicit Block(unsigned T) : bar((std::ptrdiff_t)T), nWarps((T + 31) / 32), slots(nWarps) {
        for (unsigned w = 0; w < nWarps; ++w) warpBars.emplace_back(new std::barrier<>(32));
    }
    std::barrier<> bar;
    unsigned nWarps;
    std::vector<std::unique_ptr<std::barrier<>>> warpBars;
    std::vector<std::array<unsigned long long, 32>> slots;
};
struct ThreadCtx {
    pf_uint3 tid{0, 0, 0}, bid{0, 0, 0};
    dim3 bdim, gdim;
    unsigned lane = 0, warp = 0;
    Block* blk = nullptr;
};
inline thread_local ThreadCtx tc;

struct Cfg {
    dim3 g, b;
    Cfg(dim3 grid, dim3 block, size_t = 0, void* = nullptr) : g(grid), b(block) {}
};
inline void launch(const Cfg& cfg, const std::function<void()>& fn) {
    const unsigned T = cfg.b.x;
    if (T == 0 || cfg.g.x == 0) return;
    if (T % 32 != 0) { fprintf(stderr, "[shim] blockDim must be a multiple of 32\n"); abort(); }
    Block blk(T);
    std::vector<std::thread> ws;
    ws.reserve(T);
    for (unsigned t = 0; t < T; ++t)
        ws.emplace_back([&, t] {
            tc.tid = {t, 0, 0};
            tc.bdim = cfg.b;
            tc.gdim = cfg.g;
            tc.lane = t & 31u;
            tc.warp = t >> 5;
            tc.blk = &blk;
            for (unsigned b = 0; b < cfg.g.x; ++b) {
                tc.bid = {b, 0, 0};
                fn();
                blk.bar.arrive_and_wait();
            }
        });
    for (auto& w : ws) w.join();
}
template <class T>
inline unsigned long long toBits(T v) { unsigned long long b = 0; memcpy(&b, &v, sizeof(T)); return b; }
template <class T>
inline T fromBits(unsigned long long b) { T v; memcpy(&v, &b, sizeof(T)); return v; }
inline void warpSync() { tc.blk->warpBars[tc.warp]->arrive_and_wait(); }
}  // namespace pfshim

#define threadIdx (pfshim::tc.tid)
#define blockIdx (pfshim::tc.bid)
#define blockDim (pfshim::tc.bdim)
#define gridDim (pfshim::tc.gdim)

inline void __syncthreads() { pfshim::tc.blk->bar.arrive_and_wait(); }

inline unsigned __ballot_sync(unsigned, bool pred) {
    auto& s = pfshim::tc.blk->slots[pfshim::tc.warp];
    s[pfshim::tc.lane] = pred ? 1 : 0;
    pfshim::warpSync();
    unsigned r = 0;
    for (unsigned i = 0; i < 32; ++i) if (s[i]) r |= 1u << i;
    pfshim::warpSync();
    return r;
}
template <class T>
inline T __shfl_sync(unsigned, T v, int src) {
    auto& s = pfshim::tc.blk->slots[pfshim::tc.warp];
    s[pfshim::tc.lane] = pfshim::toBits(v);
    pfshim::warpSync();
    T r = pfshim::fromBits<T>(s[(unsigned)src & 31u]);
    pfshim::warpSync();
    return r;
}
template <class T>
inline T __shfl_down_sync(unsigned, T v, unsigned d) {
    auto& s = pfshim::tc.blk->slots[pfshim::tc.warp];
    s[pfshim::tc.lane] = pfshim::toBits(v);
    pfshim::warpSync();
    unsigned src = pfshim::tc.lane + d;
    T r = src < 32 ? pfshim::fromBits<T>(s[src]) : v;
    pfshim::warpSync();
    return r;
}
inline int __ffs(unsigned x) { return __builtin_ffs((int)x); }

// ---- atomics ----
inline unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
inline unsigned atomicOr(unsigned* p, unsigned v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
inline unsigned atomicAnd(unsigned* p, unsigned v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicCAS(unsigned long long* p, unsigned long long cmp, unsigned long long val) {
    __atomic_compare_exchange_n(p, &cmp, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return cmp;   // giá trị cũ
}
inline unsigned long long atomicMin(unsigned long long* p, unsigned long long v) {
    unsigned long long old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v < old && !__atomic_compare_exchange_n(p, &old, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return old;
}
