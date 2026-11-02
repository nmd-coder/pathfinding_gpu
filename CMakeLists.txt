cmake_minimum_required(VERSION 3.18)
project(PathfindingGPU LANGUAGES CXX CUDA)

# ---- Chuẩn C++/CUDA ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Kiến trúc GPU: mặc định "native" (GPU đang cắm trên máy), có thể ghi đè:
#      cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 61 75 86)
  endif()
endif()

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(pathfinder
  src/main.cpp
  src/graph.cpp
  src/cpu_sssp.cpp
  src/verify.cpp
  src/report.cpp
  src/term.cpp
  src/gpu_device.cu
  src/gpu_sssp.cu)

target_include_directories(pathfinder PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pathfinder PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)

if(MSVC)
  target_compile_options(pathfinder PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /utf-8 /EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -Xcompiler=/O2>)
  target_compile_definitions(pathfinder PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(pathfinder PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>)
endif()
target_compile_options(pathfinder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
