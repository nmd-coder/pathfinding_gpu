# Kết quả thực nghiệm trên GPU thật — Google Colab, NVIDIA Tesla T4

Ngày chạy: 03/11/2026. Mã nguồn: `pathfinding_gpu` (bản có `--sync-every`). Tất cả số liệu là **trung bình 3–5 truy vấn ngẫu nhiên**, thời gian GPU đo bằng `cudaEvent_t` (gồm khởi tạo + mọi vòng lặp + đồng bộ Host, không gồm nạp đồ thị), CPU đo bằng `std::chrono::steady_clock`.

## 1. Môi trường (mục 3.1.1 – 3.1.2)

| Thành phần | Cấu hình |
|---|---|
| GPU | NVIDIA Tesla T4 — Turing, CC 7.5, 40 SM, 1590 MHz, 15 GB GDDR6, 256-bit, 320 GB/s (lý thuyết), L2 4 MB |
| CPU | Intel Xeon (Colab), 2 luồng phần cứng (OpenMP 2 luồng) |
| Phần mềm | CUDA 13.0 (driver 580.82), nvcc 13.0.88, GCC 13.3, `-O3 -arch=sm_75` |
| Biên dịch | Lần đầu tiên, không lỗi, không cảnh báo |

## 2. Kiểm thử đúng đắn (mục 3.2)

| Hạng mục | Số lượng |
|---|---|
| Lần chạy GPU được kiểm chứng | **308** |
| Phép so khoảng cách GPU ↔ CPU `std::priority_queue` | **705.479.974** |
| Phép kiểm cạnh cha trên cây đường đi ngắn nhất | **571.250.332** |
| Đường đi được dựng lại và kiểm tổng trọng số | **34.452** |
| Lỗi | **0 → PASS 100%** |

Bao phủ: 3 thuật toán (Naive / Frontier / Delta-Stepping) × 3 ánh xạ luồng × 3 chiến lược atomic × SSSP + A*, trên lưới đường bộ 10⁴–4·10⁶ đỉnh, R-MAT tới 16 triệu cạnh, mạng đường bộ thật SNAP roadNet-PA / roadNet-CA.

## 3. Hệ số tăng tốc theo kích thước (mục 3.4.1 – 3.4.3)

**Lưới đường bộ có tọa độ** (CPU = Dijkstra 1 luồng):

| Đồ thị | V | E | CPU (ms) | GPU Naive | GPU Frontier | GPU Delta | **Tăng tốc** | GTEPS | CPU A* | GPU A* | **A* tăng tốc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100×100 | 10.000 | 39.822 | 0.91 | 3.08 | 2.39 | 3.50 | 0.38× | 0.017 | 1.08 | 2.27 | 0.48× |
| 173×173 | 29.929 | 119.498 | 3.18 | 6.47 | 4.89 | 6.64 | 0.65× | 0.024 | 3.73 | 4.82 | 0.77× |
| 316×316 | 99.856 | 399.762 | 11.90 | 14.53 | 10.53 | 14.42 | 1.13× | 0.038 | 17.86 | 11.34 | 1.57× |
| 548×548 | 300.304 | 1.203.734 | 44.39 | 29.80 | 21.80 | 23.98 | 2.04× | 0.055 | 44.36 | 21.93 | 2.02× |
| 1000×1000 | 1.000.000 | 4.008.120 | 187.1 | 122.8 | 107.9 | **50.20** | **3.73×** | 0.080 | 219.1 | 54.91 | **3.99×** |
| 1414×1414 | 1.999.396 | 8.017.360 | 423.9 | 355.8 | 455.1 | **76.86** | **5.52×** | 0.104 | 602.5 | 93.62 | **6.44×** |
| 2000×2000 | 4.000.000 | 16.041.488 | 920.8 | 862.9 | 1598.0 | **105.2** | **8.76×** | 0.153 | 1269.0 | 129.3 | **9.82×** |

**Mạng đường bộ thật (SNAP, trọng số sinh ngẫu nhiên 1–1000, A* với h = 0 vì không có tọa độ):**

| Đồ thị | V | E | CPU Dijkstra | GPU Naive | GPU Frontier | GPU Delta | **Tăng tốc** | CPU A* | GPU A* Delta | **A* tăng tốc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 1.090.920 | 3.083.796 | 178.7 | 73.4 | 64.1 | **28.9** | **6.18×** | 211.9 | 31.1 | **6.81×** |
| roadNet-CA | 1.971.281 | 5.533.214 | 361.8 | 131.3 | 137.5 | **38.1** | **9.50×** | 482.0 | 39.1 | **12.3×** |

**R-MAT (bậc lũy thừa, đường kính nhỏ):**

| Đồ thị | V | E | CPU (ms) | GPU Naive | GPU Frontier | GPU Delta | **Tăng tốc** | GTEPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s14 | 16.384 | 227.784 | 3.37 | 6.60 | 1.44 | 1.60 | 2.34× | 0.158 |
| s17 | 131.072 | 1.942.362 | 36.03 | 23.37 | 4.25 | 4.70 | 8.48× | 0.457 |
| s20 | 1.048.576 | 16.084.984 | 457.2 | 125.8 | **32.07** | 35.16 | **14.3×** | 0.502 |

**Nhận xét:**
* Đồ thị nhỏ (≤ 10⁵ đỉnh): GPU chậm hơn CPU do chi phí khởi chạy kernel và đồng bộ chiếm ưu thế. Điểm hòa vốn khoảng 10⁵ đỉnh.
* Tăng tốc của **Delta-Stepping tăng đều theo kích thước** (0.4× → 8.8×) vì khối lượng công việc gần bằng Dijkstra (Work/m ≈ 1.2–1.6) trong khi mỗi bucket có hàng nghìn đỉnh xử lý song song.
* Naive và Frontier (bản chất Bellman-Ford) **làm thừa 30–90 lần số cạnh** trên mạng đường bộ (đường kính lớn), nên tăng tốc giảm dần khi đồ thị lớn hơn. Đây là minh chứng định lượng cho việc cần Delta-Stepping.
* Trên R-MAT (đường kính nhỏ, nhiều cạnh), cả Frontier lẫn Delta đều đạt khoảng 14×.

## 4. Atomic Operations (mục 2.4.1–2.4.2, 3.3.2) — lưới 10⁶ đỉnh

| Thuật toán | Chiến lược | Thời gian (ms) | Lệnh atomic | Cập nhật thật | Atomic/cập nhật | Tiết kiệm |
|---|---|---:|---:|---:|---:|---:|
| Naive | atomicMin không kiểm tra | 121.5 | 347.231.972 | 107.127.763 | 3.24 | — |
| Naive | atomicMin + Read-Before-Atomic | 126.5 | 110.070.204 | 107.158.156 | 1.03 | **68.3%** |
| Naive | atomicCAS + RBA | 126.2 | 113.175.096 | 107.564.039 | 1.05 | 67.4% |
| Frontier | atomicMin không kiểm tra | 114.7 | 269.804.790 | 80.315.053 | 3.36 | — |
| Frontier | atomicMin + RBA | 113.9 | 82.515.658 | 81.286.453 | 1.02 | **69.4%** |
| Delta | atomicMin không kiểm tra | 45.8 | 4.909.336 | 1.591.974 | 3.08 | — |
| Delta | atomicMin + RBA | 46.6 | 1.648.900 | 1.590.561 | 1.04 | **66.4%** |

Read-Before-Atomic loại khoảng **2/3 số lệnh atomic** (từ ~3.2 xuống ~1.03 atomic cho mỗi lần cập nhật). Thời gian gần như không đổi trên T4, vì đồ thị đường bộ thưa (bậc 4) nên mức tranh chấp trên cùng địa chỉ thấp, và atomic tại L2 của Turing có thông lượng cao. Lợi ích về băng thông/tranh chấp sẽ rõ hơn trên đồ thị bậc cao; nên đo thêm bằng Nsight Compute nếu có GPU riêng. `atomicCAS` chậm hơn `atomicMin` 64-bit khoảng 3% do vòng thử lại.

## 5. Giảm đồng bộ Host–Device (`--sync-every`)

Lưới 10⁶ đỉnh (file `study_sync_grid_1000x1000.csv`): Delta-Stepping 55.9 → 45.5 ms (S = 16, **1.23×**, số lần đồng bộ 1.830 → 388), A* 74.7 → 53.4 ms (S = 8, **1.40×**), Frontier 127.3 → 111.4 ms (S = 32, 1.14×, đồng bộ 1.482 → 47). S quá lớn (32) với Delta lại chậm hơn vì phát thừa nhiều kernel rỗng. Trên R-MAT (chỉ ~20 vòng) không có tác dụng. Phần chi phí còn lại là độ trễ khởi chạy kernel cùng chuỗi phụ thuộc khoảng 1.700 vòng, vốn là giới hạn nội tại của mạng đường bộ.

## 6. Tham số Delta (mục 3.3.3)

| Đồ thị | Δ tự động (32·w̄/d̄) | Thời gian | Δ tốt nhất | Thời gian | Ghi chú |
|---|---:|---:|---:|---:|---|
| Lưới 10⁶ | 1.245 | 45.5 ms | 9.978 | **35.1 ms** | Δ nhỏ: 3.946 bucket (503 ms); Δ rất lớn: Work/m = 66 (129 ms) |
| R-MAT s20 | 1.044 | 44.1 ms | 125 | **22.6 ms** | Với Δ tối ưu, tăng tốc R-MAT đạt khoảng 20× |

Đường cong hình chữ U khớp lý thuyết: Δ nhỏ thì nhiều bucket và ít song song; Δ lớn thì suy biến về Bellman-Ford và làm thừa việc.

## 7. Ánh xạ luồng / cân bằng tải (mục 2.2.3, 2.4.3)

| Đồ thị | Thuật toán | Thread | Warp | Hybrid (CTA/Warp/Thread) |
|---|---|---:|---:|---:|
| Lưới 10⁶ (bậc ≤ 8) | Frontier | 125.6 ms | 200.9 ms (0.63×) | 126.7 ms (0.99×) |
| Lưới 10⁶ | Delta | 51.3 ms | 120.3 ms (0.43×) | 51.7 ms (0.99×) |
| R-MAT s20 (bậc lớn nhất 39.665) | Frontier | 229.4 ms | 45.6 ms (5.0×) | **35.7 ms (6.4×)** |
| R-MAT s20 | Delta | 226.7 ms | 46.6 ms (4.9×) | **37.4 ms (6.1×)** |

Hybrid gần như không tốn thêm chi phí trên đồ thị bậc thấp, và nhanh hơn **6×** trên đồ thị bậc lũy thừa, nhờ loại bỏ phân kỳ warp khi bậc chênh lệch lớn.

## 8. Truyền dữ liệu Host → Device (mục 2.4.4) — CSR 144 MB (4 triệu đỉnh)

| Phương pháp | Thời gian | Băng thông |
|---|---:|---:|
| cudaMemcpy (pageable) | 31.4 ms | 4.6 GB/s |
| cudaMemcpyAsync (pinned) | 11.7 ms | **12.3 GB/s (2.7×)** |
| pinned + 3 CUDA streams | 11.8 ms | 12.3 GB/s |

Pinned memory nhanh hơn 2.7×. Thêm stream không có lợi vì T4 chỉ có một copy engine theo chiều H2D và băng thông đã bão hòa PCIe.

## 9. Nhiều nguồn (mục 3.4.2) — lưới 10⁶ đỉnh, CPU OpenMP 2 luồng

| K | CPU 1 luồng | CPU OpenMP | GPU batch Frontier | GPU K × Delta | Tăng tốc (so CPU 1T / OMP) |
|---:|---:|---:|---:|---:|---:|
| 1 | 193 ms | 193 ms | 101 ms | **38 ms** | 5.1× / 5.1× |
| 8 | 1.835 ms | 1.401 ms | 1.793 ms | **406 ms** | 4.5× / 3.5× |
| 32 | 5.757 ms | 4.199 ms | 6.605 ms | **1.539 ms** | 3.7× / 2.7× |

Gộp K nguồn vào chung một biên Frontier không hiệu quả trên mạng đường bộ, vì Frontier làm thừa việc ×75. Chạy K lần Delta-Stepping nhanh hơn 3.7–5×. Hướng phát triển: Delta-Stepping dạng batch.

## 10. Hạn chế đã ghi nhận
* Máy chủ DIMACS (diag.uniroma1.it) không truy cập được từ Colab, nên dữ liệu đường bộ thật lấy từ SNAP (không có trọng số và tọa độ gốc).
* CPU baseline của Colab chỉ có 2 luồng; trên máy nhiều nhân, CPU OpenMP sẽ mạnh hơn.
* Chưa đo bằng Nsight Compute (Colab không cấp quyền profiler counters cho người dùng thường).
