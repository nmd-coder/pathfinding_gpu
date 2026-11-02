@echo off
REM ===========================================================================
REM  Toan bo thuc nghiem cho Chuong 3 (mat vai phut tuy GPU/CPU)
REM  Log terminal duoc luu vao results\log_*.txt de dua vao bao cao
REM ===========================================================================
cd /d "%~dp0\.."
if not exist pathfinder.exe (echo Build first: build.bat & exit /b 1)
if not exist results mkdir results

echo [1/8] Hardware info
pathfinder.exe --mode info --no-color > results\log_info.txt
type results\log_info.txt

echo [2/8] Correctness verification (grid 1e6 + R-MAT 2^18)
pathfinder.exe --mode verify --graph grid --n 1e6 --queries 5
pathfinder.exe --mode verify --graph rmat --n 262144 --queries 5

echo [3/8] Benchmark grid road networks 1e4 .. 4e6
pathfinder.exe --mode bench --family grid --sizes 1e4,3e4,1e5,3e5,1e6,2e6,4e6 --queries 5

echo [4/8] Benchmark R-MAT (power-law)
pathfinder.exe --mode bench --family rmat --sizes 16384,131072,1048576 --queries 5 --out results\rmat

echo [5/8] Atomic contention study
pathfinder.exe --mode atomic --graph grid --n 1e6 --queries 5

echo [6/8] Delta sweep (grid + rmat)
pathfinder.exe --mode delta --graph grid --n 1e6 --queries 3
pathfinder.exe --mode delta --graph rmat --n 1048576 --queries 3

echo [7/8] Mapping / load balancing + transfer
pathfinder.exe --mode mapping --graph grid --n 1e6 --queries 3
pathfinder.exe --mode mapping --graph rmat --n 1048576 --queries 3
pathfinder.exe --mode transfer --graph grid --n 4e6

echo [8/8] Multi-source batching
pathfinder.exe --mode multisource --graph grid --n 1e6 --batch 32

where python >nul 2>nul && python scripts\plot_results.py results
echo Done. See results\ and results\figures\
