#!/usr/bin/env bash
# Toàn bộ thực nghiệm cho Chương 3 (Linux). Chạy từ thư mục gốc: bash scripts/run_all_experiments.sh
set -e
cd "$(dirname "$0")/.."
[ -x ./pathfinder ] || make -f linux.mk
mkdir -p results
./pathfinder --mode info --no-color | tee results/log_info.txt
./pathfinder --mode verify --graph grid --n 1e6 --queries 5
./pathfinder --mode verify --graph rmat --n 262144 --queries 5
./pathfinder --mode bench --family grid --sizes 1e4,3e4,1e5,3e5,1e6,2e6,4e6 --queries 5
./pathfinder --mode bench --family rmat --sizes 16384,131072,1048576 --queries 5 --out results/rmat
./pathfinder --mode atomic --graph grid --n 1e6 --queries 5
./pathfinder --mode sync --graph grid --n 1e6 --queries 3
./pathfinder --mode sync --graph rmat --n 1048576 --queries 3
./pathfinder --mode delta --graph grid --n 1e6 --queries 3
./pathfinder --mode delta --graph rmat --n 1048576 --queries 3
./pathfinder --mode mapping --graph grid --n 1e6 --queries 3
./pathfinder --mode mapping --graph rmat --n 1048576 --queries 3
./pathfinder --mode transfer --graph grid --n 4e6
./pathfinder --mode multisource --graph grid --n 1e6 --batch 32
python3 scripts/plot_results.py results || true
