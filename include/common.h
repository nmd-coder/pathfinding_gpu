// =============================================================================
//  common.h - Kiểu dữ liệu, hằng số và tiện ích dùng chung (Host + Device)
//  Đề tài: GPU-Accelerated Parallel Dijkstra / A* trên đồ thị quy mô lớn
// =============================================================================
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;

// Khoảng cách "vô cực" và đỉnh cha "không tồn tại".
// Trọng số cạnh là số nguyên không âm 32-bit -> phép so sánh/atomicMin chính xác tuyệt đối,
// kết quả GPU và CPU có thể đối chiếu bằng phép so sánh bằng (==), không có sai số dấu phẩy động.
constexpr u32 INF_DIST  = 0xFFFFFFFFu;
constexpr u32 NO_PARENT = 0xFFFFFFFFu;

#if defined(__CUDACC__)
#define PF_HD __host__ __device__ __forceinline__
#else
#define PF_HD inline
#endif

// Đồng hồ đo thời gian phía CPU (độ phân giải nano giây).
class HostTimer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double elapsedMs() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
};

// Bộ sinh số ngẫu nhiên SplitMix64: kết quả giống hệt nhau trên MSVC/GCC/Clang
// (std::uniform_*_distribution phụ thuộc thư viện chuẩn -> đồ thị sinh ra có thể khác nhau).
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    u64 next() {
        u64 z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }   // [0,1)
    double uniform(double a, double b) { return a + (b - a) * uniform(); }
    u32 below(u32 n) { return (u32)(next() % n); }                              // [0,n)
};

inline u64 hash64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33; return x;
}

// ----------------------------------------------------------------------------
// Màu ANSI cho Terminal (Windows 10+ cần bật Virtual Terminal Processing).
// ----------------------------------------------------------------------------
namespace term {
void init(bool enableColor);   // gọi 1 lần đầu chương trình
bool colorEnabled();
const char* reset();
const char* bold();
const char* green();
const char* red();
const char* yellow();
const char* cyan();
const char* magenta();
const char* dim();
}
