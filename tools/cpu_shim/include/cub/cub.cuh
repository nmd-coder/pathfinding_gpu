// Mô phỏng tối thiểu CUB cho chế độ kiểm thử trên CPU: DeviceScan::ExclusiveSum (tuần tự)
#pragma once
#include "../cuda_runtime.h"
namespace cub {
struct DeviceScan {
    template <class InT, class OutT>
    static cudaError_t ExclusiveSum(void* temp, size_t& bytes, InT in, OutT out, int n, cudaStream_t = nullptr) {
        if (!temp) { bytes = 16; return cudaSuccess; }
        unsigned long long acc = 0;
        for (int i = 0; i < n; ++i) { auto x = in[i]; out[i] = (decltype(+out[i]))acc; acc += x; }
        return cudaSuccess;
    }
};
}
