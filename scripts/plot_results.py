#!/usr/bin/env python3
"""
plot_results.py - Vẽ biểu đồ cho Chương 3 từ các file CSV do pathfinder xuất ra.
Dùng:  python scripts/plot_results.py results
Cần:   pip install matplotlib
Ảnh được lưu vào <results>/figures/*.png (300 dpi, chèn thẳng vào báo cáo Word).
"""
import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed: pip install matplotlib")
    sys.exit(0)

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"CPU": "#6b7280", "Naive": "#f59e0b", "Frontier": "#3b82f6", "Delta": "#10b981", "AStar": "#8b5cf6"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print("  saved", p)


def family(graph):
    return graph.split("_")[0]


# ---------------------------------------------------------------- 3.4.3 Speedup theo kích thước
def plot_summary(rows, suffix=""):
    if not rows:
        return
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[family(r["graph"])].append(r)
    for fam, rs in by_fam.items():
        rs.sort(key=lambda r: int(r["n"]))
        n = [int(r["n"]) for r in rs]
        # Thời gian thực thi (log-log)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, label, col in [("cpu_ms", "CPU Dijkstra (1 thread)", COLORS["CPU"]),
                                ("naive_ms", "GPU K1 Naive-Relax", COLORS["Naive"]),
                                ("frontier_ms", "GPU K2 Frontier", COLORS["Frontier"]),
                                ("delta_ms", "GPU K2 Delta-Stepping", COLORS["Delta"])]:
            ax.plot(n, [float(r[key]) for r in rs], marker="o", label=label, color=col)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Number of vertices |V|"); ax.set_ylabel("Execution time (ms, log scale)")
        ax.set_title(f"SSSP execution time vs graph size ({fam})")
        ax.legend()
        save(fig, f"time_vs_size_{fam}{suffix}.png")
        # Hệ số tăng tốc
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, label, col in [("naive_ms", "Naive-Relax", COLORS["Naive"]),
                                ("frontier_ms", "Frontier", COLORS["Frontier"]),
                                ("delta_ms", "Delta-Stepping", COLORS["Delta"])]:
            ax.plot(n, [float(r["cpu_ms"]) / max(1e-9, float(r[key])) for r in rs], marker="o", label=label, color=col)
        ax.plot(n, [float(r["astar_speedup"]) for r in rs], marker="s", ls="--", label="A* (best GPU vs CPU A*)", color=COLORS["AStar"])
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Number of vertices |V|"); ax.set_ylabel("Speedup over single-thread CPU (x)")
        ax.set_title(f"GPU speedup factor vs graph size ({fam})")
        ax.legend()
        save(fig, f"speedup_vs_size_{fam}{suffix}.png")
        # TEPS
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar([f"{x:,}" for x in n], [float(r["best_gteps"]) for r in rs], color=COLORS["Delta"])
        ax.set_xlabel("Number of vertices |V|"); ax.set_ylabel("Best GPU GTEPS")
        ax.set_title(f"Traversed edges per second ({fam})")
        save(fig, f"teps_{fam}{suffix}.png")


# ---------------------------------------------------------------- 3.3.2 Atomic
def plot_atomic(rows):
    if not rows:
        return
    algos = list(dict.fromkeys(r["algorithm"] for r in rows))
    modes = list(dict.fromkeys(r["atomic_mode"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        xs = [a + (i - (len(modes) - 1) / 2) * w for a in range(len(algos))]
        at = [next(float(r["atomics"]) for r in rows if r["algorithm"] == al and r["atomic_mode"] == m) for al in algos]
        tm = [next(float(r["time_ms"]) for r in rows if r["algorithm"] == al and r["atomic_mode"] == m) for al in algos]
        axes[0].bar(xs, at, w, label=m)
        axes[1].bar(xs, tm, w, label=m)
    for ax, t, yl in [(axes[0], "Atomic operations issued", "count"), (axes[1], "Execution time", "ms")]:
        ax.set_xticks(range(len(algos))); ax.set_xticklabels(algos)
        ax.set_title(t); ax.set_ylabel(yl); ax.legend(fontsize=8)
    save(fig, "atomic_study.png")


# ---------------------------------------------------------------- 3.3.3 Delta
def plot_delta():
    files = [f for f in os.listdir(RES) if f.startswith("study_delta_") and f.endswith(".csv")]
    for f in files:
        rows = read(os.path.join(RES, f))
        if not rows:
            continue
        d = [float(r["delta"]) for r in rows]
        fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
        ax1.plot(d, [float(r["time_ms"]) for r in rows], marker="o", color=COLORS["Delta"], label="time (ms)")
        ax1.set_xscale("log"); ax1.set_xlabel("Delta (log scale)"); ax1.set_ylabel("Time (ms)", color=COLORS["Delta"])
        ax2 = ax1.twinx()
        ax2.plot(d, [float(r["work_ratio"]) for r in rows], marker="s", ls="--", color=COLORS["Naive"], label="work / m")
        ax2.set_ylabel("Edges scanned / reachable edges", color=COLORS["Naive"])
        ax2.grid(False)
        ax1.set_title(f"Delta-Stepping sweep: {rows[0]['graph']}")
        save(fig, f.replace("study_", "").replace(".csv", ".png"))


# ---------------------------------------------------------------- 3.4.2 Multi-source
def plot_multisource(rows):
    if not rows:
        return
    K = [int(r["K"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(K, [float(r["cpu_1t_ms"]) for r in rows], marker="o", color=COLORS["CPU"], label="CPU 1 thread")
    ax.plot(K, [float(r["cpu_omp_ms"]) for r in rows], marker="o", color="#111827", label=f"CPU OpenMP ({rows[0]['omp_threads']} threads)")
    ax.plot(K, [float(r["gpu_batch_ms"]) for r in rows], marker="o", color=COLORS["Frontier"], label="GPU batched Frontier")
    ax.plot(K, [float(r["gpu_seq_delta_ms"]) for r in rows], marker="o", color=COLORS["Delta"], label="GPU K x Delta-Stepping")
    if "gpu_batch_delta_ms" in rows[0]:
        ax.plot(K, [float(r["gpu_batch_delta_ms"]) for r in rows], marker="s", linewidth=2.4, color="#7c3aed", label="GPU batched Delta-Stepping")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Number of sources K"); ax.set_ylabel("Total time (ms)")
    ax.set_title(f"Multi-source pathfinding: {rows[0]['graph']}")
    ax.legend()
    save(fig, f"multisource_{rows[0]['graph']}.png")


def plot_compaction():
    files = sorted(f for f in os.listdir(RES) if f.startswith("study_compaction_") and f.endswith(".csv"))
    rows = [r for f in files for r in read(os.path.join(RES, f))]
    if not rows:
        return
    graphs = list(dict.fromkeys(r["graph"] for r in rows))
    modes = ["warp-atomic", "prefix-sum"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.38
    for i, m in enumerate(modes):
        vals = [next((float(r["time_ms"]) for r in rows if r["graph"] == g and r["compaction"] == m), 0) for g in graphs]
        bars = ax.bar([x + (i - 0.5) * w for x in range(len(graphs))], vals, w, label=m)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(range(len(graphs))); ax.set_xticklabels(graphs, fontsize=8)
    ax.set_ylabel("Time (ms)"); ax.set_title("Frontier compaction: warp-aggregated atomic vs Prefix Sum (CUB)")
    ax.legend()
    save(fig, "compaction.png")


def plot_mapping():
    files = [f for f in os.listdir(RES) if f.startswith("study_mapping") and f.endswith(".csv")]
    rows = [r for f in files for r in read(os.path.join(RES, f))]
    if not rows:
        return
    groups = list(dict.fromkeys((r["graph"], r["algorithm"]) for r in rows))
    maps = ["thread", "warp", "hybrid"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.25
    for i, m in enumerate(maps):
        vals = [next((float(r["time_ms"]) for r in rows if (r["graph"], r["algorithm"]) == g and r["mapping"] == m), 0) for g in groups]
        ax.bar([x + (i - 1) * w for x in range(len(groups))], vals, w, label=m)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{g}\n{a}" for g, a in groups], fontsize=8)
    ax.set_ylabel("Time (ms)"); ax.set_title("Thread mapping / load balancing"); ax.legend()
    save(fig, "mapping.png")


def plot_sync():
    files = [f for f in os.listdir(RES) if f.startswith("study_sync_") and f.endswith(".csv")]
    for f in files:
        rows = read(os.path.join(RES, f))
        if not rows:
            continue
        groups = list(dict.fromkeys((r["algorithm"], r["query"]) for r in rows))
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for (a, q) in groups:
            rs = [r for r in rows if (r["algorithm"], r["query"]) == (a, q)]
            ax.plot([int(r["sync_every"]) for r in rs], [float(r["time_ms"]) for r in rs], marker="o", label=f"{a} ({q})")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Kernels launched per host synchronization (S)"); ax.set_ylabel("Time (ms)")
        ax.set_title(f"Reducing host-device synchronization: {rows[0]['graph']}")
        ax.legend()
        save(fig, f.replace("study_", "").replace(".csv", ".png"))


if __name__ == "__main__":
    print(f"Reading CSV from {RES}/")
    plot_summary(read(os.path.join(RES, "summary_speedup.csv")))
    plot_summary(read(os.path.join(RES, "rmat", "summary_speedup.csv")))
    plot_atomic(read(os.path.join(RES, "study_atomic.csv")))
    plot_delta()
    plot_sync()
    for f in sorted(os.listdir(RES)):
        if f.startswith("study_multisource") and f.endswith(".csv"):
            plot_multisource(read(os.path.join(RES, f)))
    plot_compaction()
    plot_mapping()
    print("Done.")
