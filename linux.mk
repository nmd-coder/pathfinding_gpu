# =============================================================================
#  Linux build: make -f linux.mk           (mặc định ARCH=-arch=native)
#               make -f linux.mk ARCH=-arch=sm_86
#  CPU-shim build (không cần GPU, chỉ để kiểm thử logic): make -f linux.mk shim
# =============================================================================
NVCC     ?= nvcc
ARCH     ?= -arch=native
NVFLAGS  := -O3 -std=c++17 $(ARCH) -lineinfo -Iinclude -Isrc -Xcompiler "-O3 -fopenmp"
SRCS_CPP := src/main.cpp src/graph.cpp src/cpu_sssp.cpp src/verify.cpp src/report.cpp src/term.cpp
SRCS_CU  := src/gpu_device.cu src/gpu_sssp.cu

pathfinder: $(SRCS_CPP) $(SRCS_CU) $(wildcard include/*.h) $(wildcard src/*.cuh)
	$(NVCC) $(NVFLAGS) $(SRCS_CPP) $(SRCS_CU) -o $@ -lgomp

shim:
	bash tools/cpu_shim/build_shim.sh

demo: pathfinder
	./pathfinder --mode demo

clean:
	rm -rf pathfinder build build_shim

.PHONY: shim demo clean
