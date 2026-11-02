// =============================================================================
//  cuda_check.cuh - Macro kiểm tra lỗi CUDA
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error '%s' at %s:%d -> %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                            \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// Kiểm tra lỗi cấu hình khi khởi chạy kernel (lỗi thực thi sẽ hiện ở lần đồng bộ kế tiếp)
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
