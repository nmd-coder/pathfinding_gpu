#!/usr/bin/env bash
# Biên dịch toàn bộ dự án ở chế độ mô phỏng CPU (không cần GPU / CUDA Toolkit).
# Dùng để kiểm thử LOGIC thuật toán; thời gian đo được KHÔNG phản ánh hiệu năng GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build_shim"
mkdir -p "$OUT/gen"
for f in "$ROOT"/src/*.cu; do
  python3 "$ROOT/tools/cpu_shim/transform_launch.py" "$f" "$OUT/gen/$(basename "${f%.cu}").cpp"
done
cp "$ROOT"/src/*.cuh "$OUT/gen/"
CXX=${CXX:-g++}
$CXX -std=c++20 -O2 -fopenmp -pthread -Wall -Wno-unknown-pragmas \
  -I"$ROOT/tools/cpu_shim/include" -I"$ROOT/include" -I"$OUT/gen" \
  "$ROOT"/src/*.cpp "$OUT"/gen/*.cpp -o "$OUT/pathfinder_shim"
echo "Built $OUT/pathfinder_shim"
