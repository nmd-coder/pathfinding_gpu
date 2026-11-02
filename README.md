# GPU-Accelerated Parallel Dijkstra / A* trên đồ thị quy mô lớn

Mã nguồn CUDA C++ cho đề tài **"Thuật toán Pathfinding song song quy mô lớn trên bản đồ đồ thị"**.
Gồm: biểu diễn CSR, CPU baseline (`std::priority_queue`), 3 họ kernel GPU (Naive Relaxation,
Frontier-based, Delta-Stepping) cho cả Dijkstra và A*, tối ưu Atomic (Read-Before-Atomic,
atomicMin/atomicCAS 64-bit), cân bằng tải, đa nguồn, kiểm thử tự động và benchmark xuất bảng + CSV + biểu đồ.

---

## 1. Cấu trúc thư mục

```
pathfinding_gpu/
├── include/
│   ├── common.h          kiểu dữ liệu, INF, bộ sinh ngẫu nhiên tái lập được, màu Terminal
│   ├── heuristic.h       h(v) cho A* (dùng chung CPU & GPU, chứng minh admissible + consistent)
│   ├── graph.h           STEP 1: cấu trúc CSR, bộ sinh / bộ đọc đồ thị
│   ├── cpu_sssp.h        CPU baseline: Dijkstra, A*, đa nguồn OpenMP
│   ├── verify.h          kiểm thử đúng đắn GPU vs CPU
│   ├── gpu_sssp.h        giao diện Host của bộ giải GPU (không lộ kiểu CUDA)
│   └── report.h          bảng ASCII + CSV
├── src/
│   ├── graph.cpp         dựng CSR (counting sort O(n+m)), lưới đường bộ, R-MAT, DIMACS, SNAP, cache .bin
│   ├── cpu_sssp.cpp      Dijkstra / A* tuần tự chuẩn
│   ├── verify.cpp        so khoảng cách, kiểm cây cha, truy vết đường đi
│   ├── gpu_kernels.cuh   ★ TẤT CẢ CUDA KERNEL (Step 2, 3, 4)
│   ├── gpu_sssp.cu       ★ điều phối Host: cấp phát VRAM, vòng lặp Naive/Frontier/Delta, cudaEvent
│   ├── gpu_device.cu     thông tin GPU, thí nghiệm pageable / pinned / streams
│   ├── cuda_check.cuh    macro CUDA_CHECK
│   ├── report.cpp, term.cpp
│   └── main.cpp          CLI + mọi kịch bản thực nghiệm
├── scripts/
│   ├── run_demo.bat                 kịch bản quay video demo
│   ├── run_all_experiments.bat/.sh  toàn bộ số liệu Chương 3
│   ├── plot_results.py              vẽ biểu đồ PNG 300dpi từ CSV
│   └── download_dimacs.ps1/.sh      tải bản đồ thật USA-road-d (DIMACS)
├── tools/cpu_shim/       mô phỏng CUDA trên CPU để kiểm thử logic khi không có GPU
├── build.bat  CMakeLists.txt  linux.mk
```

## 2. Đối chiếu yêu cầu đề tài ↔ mã nguồn

| Yêu cầu | Vị trí trong mã | Ghi chú |
|---|---|---|
| **Step 1** – CSR trên GPU | `graph.h/.cpp: buildCSR`, `gpu_sssp.cu: Impl::Impl` | 3 mảng `row_offsets`, `col_indices`, `weights`; nạp lên VRAM bằng pinned memory + 3 CUDA streams |
| Sinh/đọc đồ thị 10⁴–10⁶ đỉnh có tọa độ | `genGridRoad`, `genRMAT`, `loadDIMACS`, `loadSNAP` | Lưới đường bộ có tọa độ (cho A*), R-MAT bậc lũy thừa, bản đồ thật DIMACS |
| CPU baseline | `cpu_sssp.cpp: cpuDijkstra, cpuAStar` | `std::priority_queue` (min-heap), lazy deletion |
| Đối chiếu tự động, PASS 100% | `verify.cpp`, `main.cpp: runVerify, finalVerdict` | So **mọi** `dist[v]`, kiểm **mọi** cạnh cha, dựng lại 64 đường đi/truy vấn |
| **Kernel 1** – đồng bộ mảng khoảng cách | `gpu_kernels.cuh: naiveRelaxKernel` | thread/đỉnh, cờ active, cờ hội tụ `d_changed`; có biến thể A* |
| **Kernel 2** – Frontier `F_curr`/`F_next` | `relaxFrontierKernel<SEL_ALL>` + `runFrontier` | chỉ xử lý đỉnh vừa thay đổi; bitmask khử trùng lặp |
| **Kernel 2** – Delta-Stepping light/heavy | `relaxFrontierKernel<SEL_LIGHT/SEL_HEAVY>`, `farMinKernel`, `farPartitionKernel`, `runDelta` | hàng CSR sắp theo trọng số ⇒ tách light/heavy bằng 1 tìm kiếm nhị phân (`lightEndKernel`) |
| **Step 4** – atomicMin / atomicCAS + Read-Before-Atomic | `relaxAtomic<AtomicMode>` | `dist` và `parent` đóng gói 64-bit ⇒ cập nhật **đồng thời, nguyên tử** |
| Giảm tranh chấp | `testAndSetBit`, `warpAggregatedInc`, `flushStats`, `farMinKernel` | đọc trước, gộp atomic theo warp, reduce theo warp |
| Thread/Warp-per-vertex, cân bằng tải | `relaxFrontierKernel<…, Mapping, …>` | Hybrid 3 tầng CTA (Shared Memory) / Warp / Thread |
| Đa nguồn (Multi-Source Batching) | `GpuSolver::multiSource` | K truy vấn chung 1 biên, phần tử = `k·n + v` |
| Benchmark: cudaEvent, TEPS, Speedup | `gpu_sssp.cu: run`, `main.cpp: benchGraph, printSummary` | bảng Terminal + CSV + biểu đồ |

## 3. Ý tưởng kỹ thuật chính

### 3.1 CSR (Step 1)
```
Đồ thị: 0→1(5), 0→2(3), 1→2(1), 2→0(7)
row_offsets = [0, 2, 3, 4]      // cạnh của u nằm trong [row_offsets[u], row_offsets[u+1])
col_indices = [2, 1, 2, 0]      // mỗi hàng đã sắp theo trọng số tăng dần
weights     = [3, 5, 1, 7]
```
Bộ nhớ `(n+1)·4 + m·8` byte, đọc liên tiếp (coalesced). Sắp xếp mỗi hàng theo trọng số không đổi
ngữ nghĩa đồ thị nhưng cho phép Delta-Stepping lấy cạnh nhẹ = `[row[u], lightEnd[u])`,
cạnh nặng = `[lightEnd[u], row[u+1])` mà không cần sao chép đồ thị khi đổi Delta.

### 3.2 Trạng thái đóng gói 64-bit + atomic (Step 4)
```
state[v] = (dist << 32) | parent          // 1 từ 64-bit
atomicMin(&state[v], (nd << 32) | u)      // cập nhật dist + parent trong 1 lệnh
```
* Không bao giờ có trạng thái "dist của luồng A nhưng parent của luồng B" ⇒ cây đường đi luôn hợp lệ.
* Khi bằng khoảng cách, đỉnh cha có chỉ số nhỏ hơn thắng ⇒ kết quả **tất định**.
* **Read-Before-Atomic**: đọc `state[v]` trước, chỉ phát lệnh atomic khi thực sự cải thiện.
  Chế độ `--mode atomic` đếm chính xác số lệnh atomic để so sánh 3 chiến lược
  (`atomicMin` không kiểm tra / `atomicMin`+RBA / vòng `atomicCAS`+RBA).
* Đẩy đỉnh vào hàng đợi: bitmask `atomicOr` (mỗi đỉnh 1 lần/vòng) + **warp-aggregated atomicAdd**
  trên bộ đếm đuôi (cả warp chỉ 1 atomic).

### 3.3 Ba họ thuật toán GPU (Step 2 & 3)
| Thuật toán | Mỗi vòng lặp xử lý | Ưu | Nhược |
|---|---|---|---|
| K1 Naive-Relax | toàn bộ n đỉnh (đỉnh active mới nới cạnh) | đơn giản, không hàng đợi | phí luồng nhàn rỗi, số vòng = độ sâu cây |
| K2 Frontier | chỉ đỉnh trong `F_curr` | không duyệt đỉnh thừa | làm lại nhiều (Bellman-Ford: Work/m > 1) |
| K2 Delta-Stepping | bucket khóa `[iΔ,(i+1)Δ)`, cạnh nhẹ lặp, cạnh nặng 1 lần | Work/m ≈ 1 như Dijkstra, vẫn song song trong bucket | cần chọn Δ |

Δ mặc định = `32 · avg_w / avg_deg` (Davidson et al. 2014); `--mode delta` quét Δ để vẽ đường cong
đánh đổi (Δ nhỏ → nhiều bucket/ít song song; Δ lớn → suy biến về Bellman-Ford, nhiều việc thừa).

### 3.4 A* song song
* `h(v) = floor(scale · ‖p(v) − p(t)‖ · (1 − 1e-9))`, `scale = min w(u,v)/‖p(u) − p(v)‖` tính từ dữ liệu
  ⇒ chứng minh được **admissible và consistent** (xem `heuristic.h`), nên A* luôn tối ưu.
* GPU A*: cắt tỉa mọi cạnh có `g + h ≥ dist[target]` (cận trên hiện tại); Delta-Stepping dùng khóa
  `f = g + h` (tương đương Dijkstra trên trọng số rút gọn) và dừng khi bucket nhỏ nhất ≥ `dist[target]`.

### 3.5 Cân bằng tải (Hybrid)
Theo bậc đỉnh trong biên: bậc ≥ blockDim → cả block xử lý (thông tin phát qua **Shared Memory**);
bậc ≥ 32 → cả warp (`__ballot_sync` + `__shfl_sync`); còn lại → từng luồng. Tránh warp divergence
nặng trên đồ thị bậc lũy thừa (R-MAT, mạng xã hội).

## 4. Biên dịch

### Windows (khuyến nghị cho máy của bạn)
Yêu cầu: **Visual Studio 2019/2022** (workload *Desktop development with C++*) + **CUDA Toolkit 11.6+** (khuyến nghị 12.x).

```bat
:: Mở "x64 Native Tools Command Prompt for VS 2022", rồi:
cd D:\TTHNC\pathfinding_gpu
build.bat
:: Nếu nvcc không hỗ trợ -arch=native, chỉ định kiến trúc GPU:
set ARCH=-arch=sm_86 & build.bat        :: RTX 30xx (sm_75: RTX 20xx, sm_89: RTX 40xx, sm_120: RTX 50xx)
```
Hoặc bằng CMake (≥ 3.18):
```bat
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Linux
```bash
make -f linux.mk                    # hoặc: make -f linux.mk ARCH=-arch=sm_86
```

## 5. Chạy

```bat
pathfinder.exe --mode demo                  :: ★ kịch bản demo đầy đủ (~1-3 phút)
scripts\run_demo.bat                        :: demo + vẽ biểu đồ
scripts\run_all_experiments.bat             :: toàn bộ số liệu Chương 3
```
Các chế độ riêng lẻ:
```bat
pathfinder.exe --mode info
pathfinder.exe --mode verify --graph grid --n 1e6 --queries 5
pathfinder.exe --mode verify --graph rmat --n 262144
pathfinder.exe --mode bench  --family grid --sizes 1e4,1e5,1e6,4e6 --queries 5
pathfinder.exe --mode bench  --family both
pathfinder.exe --mode atomic --graph grid --n 1e6
pathfinder.exe --mode delta  --graph grid --n 1e6
pathfinder.exe --mode mapping --graph rmat --n 1048576
pathfinder.exe --mode transfer --graph grid --n 4e6
pathfinder.exe --mode multisource --graph grid --n 1e6 --batch 32
pathfinder.exe --mode single --graph grid --n 1e6 --astar --algo delta --print-path
:: Bản đồ thật DIMACS (tải bằng scripts\download_dimacs.ps1 NY)
pathfinder.exe --mode bench --gr data\USA-road-d.NY.gr --co data\USA-road-d.NY.co --queries 5 --save-bin data\ny.bin
pathfinder.exe --mode verify --graph bin --file data\ny.bin
```
Tùy chọn GPU: `--mapping thread|warp|hybrid`, `--atomic min|cas|nocheck`, `--delta D`, `--block 256`.
Tắt màu khi ghi log ra file: `--no-color`.

### Ý nghĩa các cột trong bảng
| Cột | Ý nghĩa |
|---|---|
| Time (ms) | GPU: `cudaEvent` bao trọn khởi tạo + mọi vòng lặp + đồng bộ Host (không tính nạp đồ thị). CPU: `steady_clock` |
| Iter | số lần khởi chạy kernel nới lỏng (bước đồng bộ toàn cục); `(Nb)` = số bucket Delta |
| Edges scanned / Work/m | tổng cạnh đã xét / số cạnh của phần tới được — đo **hiệu quả công việc** (Dijkstra = 1.00) |
| MTEPS | (số cạnh của thành phần tới được) / thời gian — chuẩn Graph500 |
| Speedup | thời gian CPU 1 luồng / thời gian GPU |
| Verify | `PASS 100%` = toàn bộ khoảng cách, cây cha và đường đi truy vết đều khớp CPU |

CSV ở `results/`: `bench_sssp.csv`, `bench_astar.csv`, `summary_speedup.csv`, `study_atomic.csv`,
`study_delta_*.csv`, `study_mapping*.csv`, `study_transfer.csv`, `study_multisource.csv`.
Biểu đồ: `python scripts\plot_results.py results` → `results\figures\*.png`.

## 6. Gợi ý ánh xạ sang mục lục báo cáo
| Mục báo cáo | Lấy từ |
|---|---|
| 2.1 CSR, 2.1.3 Multi-Source | `graph.cpp`, `GpuSolver::Impl`, `multiSource` |
| 2.2.1 d_dist / d_parent / d_changed | `state` 64-bit (dist‖parent), `d_ctr[C_CHANGED]` |
| 2.2.2–2.2.3 Kernel nới lỏng, Thread/Warp-per-vertex | `naiveRelaxKernel`, `relaxFrontierKernel` |
| 2.3.1 F_curr/F_next | `runFrontier`, `d_q[0..1]`, `d_qMask[0..1]` |
| 2.3.2 Delta-Stepping light/heavy | `runDelta`, `lightEndKernel`, `farMinKernel`, `farPartitionKernel` |
| 2.3.3 Nén biên + Bitmask | `testAndSetBit` (bitmask), `warpAggregatedInc` (nén biên bằng prefix trong warp) |
| 2.4.1–2.4.2 atomicMin/CAS, RBA, Warp aggregation | `relaxAtomic`, `--mode atomic` |
| 2.4.3 Cân bằng tải Shared Memory | tầng CTA trong `relaxFrontierKernel`, `--mode mapping` |
| 2.4.4 Pinned + Streams | `gpuTransferStudy`, `--mode transfer` |
| 3.1 Môi trường | `--mode info` |
| 3.2 Correctness | `--mode verify` |
| 3.3.1, 3.3.3 | `--mode bench`, `--mode delta` |
| 3.3.2 Atomic contention (+ Nsight Compute) | `--mode atomic`; chạy thêm `ncu --set full pathfinder.exe --mode single ...` |
| 3.4.1, 3.4.3 TEPS, Speedup theo kích thước | `summary_speedup.csv`, `speedup_vs_size_*.png` |
| 3.4.2 Multi-source vs OpenMP | `--mode multisource` |

**Lưu ý khi phân tích kết quả (trung thực khoa học):** mạng đường bộ có đường kính rất lớn (hàng nghìn
bước) và bậc thấp (~4) nên mỗi bước đồng bộ chỉ có vài nghìn đỉnh hoạt động; tăng tốc GPU cho SSSP trên
đồ thị đường bộ thường ở mức vài lần, còn trên R-MAT (đường kính nhỏ, nhiều cạnh) cao hơn nhiều. Với A*
điểm–điểm, CPU A* chỉ duyệt một phần nhỏ đồ thị nên có thể nhanh hơn GPU với truy vấn ngắn; GPU có lợi
thế rõ khi truy vấn dài, đồ thị lớn hoặc xử lý **nhiều truy vấn đồng thời** (chế độ multisource).
Đây là những điểm nên đưa vào phần thảo luận.

## 7. Kiểm thử không cần GPU (CPU shim)
`tools/cpu_shim` mô phỏng CUDA trên CPU (mỗi CUDA thread = 1 `std::thread`, `__syncthreads`/warp
shuffle bằng `std::barrier`). Dùng để kiểm thử **logic** trên máy không có GPU NVIDIA; thời gian
đo được **không** phản ánh hiệu năng GPU.
```bash
bash tools/cpu_shim/build_shim.sh      # cần g++ >= 11 (C++20), python3
./build_shim/pathfinder_shim --mode verify --graph rmat --n 2048 --block 64
```

## 8. Xử lý sự cố
| Hiện tượng | Cách khắc phục |
|---|---|
| `nvcc fatal: Unsupported gpu architecture 'native'` | `set ARCH=-arch=sm_XX` (CUDA < 11.6) |
| `cl.exe not found` | mở *x64 Native Tools Command Prompt for VS* |
| `Not enough GPU memory` | giảm `--n`/`--sizes`, giảm `--batch` |
| Màn hình đen / driver reset khi chạy rất lâu | Windows TDR (~2 s/kernel) — các kernel ở đây ngắn, nhưng nếu dùng đồ thị cực lớn hãy tăng `TdrDelay` |
| Ký tự lạ trong Terminal | thêm `--no-color` hoặc dùng Windows Terminal |

## 9. Tài liệu tham khảo
1. P. Harish, P. J. Narayanan, *Accelerating large graph algorithms on the GPU using CUDA*, HiPC 2007.
2. U. Meyer, P. Sanders, *Δ-stepping: a parallelizable shortest path algorithm*, J. Algorithms 49(1), 2003.
3. A. Davidson, S. Baxter, M. Garland, J. D. Owens, *Work-efficient parallel GPU methods for single-source shortest paths*, IPDPS 2014.
4. D. Merrill, M. Garland, A. Grimshaw, *Scalable GPU graph traversal*, PPoPP 2012.
5. Y. Wang et al., *Gunrock: A high-performance graph processing library on the GPU*, PPoPP 2016.
6. 9th DIMACS Implementation Challenge – Shortest Paths, http://www.diag.uniroma1.it/challenge9/
7. D. Chakrabarti, Y. Zhan, C. Faloutsos, *R-MAT: A recursive model for graph mining*, SDM 2004.
8. NVIDIA, *CUDA C++ Programming Guide* (Atomic Functions, Cooperative Groups, Warp Shuffle).
