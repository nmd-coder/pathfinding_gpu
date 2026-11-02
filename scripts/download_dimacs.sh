#!/usr/bin/env bash
# Tải bản đồ đường bộ thật. Dùng: bash scripts/download_dimacs.sh NY
#  1) DIMACS USA-road-d (có trọng số + tọa độ) - máy chủ đôi khi không truy cập được
#  2) Dự phòng: SNAP roadNet-PA/CA/TX (Stanford) - không có trọng số/tọa độ (trọng số sinh ngẫu nhiên)
set -u
REGION=${1:-NY}
BASE="http://www.diag.uniroma1.it/challenge9/data/USA-road-d"
mkdir -p "$(dirname "$0")/../data" && cd "$(dirname "$0")/../data"
ok=1
for ext in gr co; do
  f="USA-road-d.$REGION.$ext"
  [ -s "$f" ] && { echo "Exists: $f"; continue; }
  if curl -fsSL --connect-timeout 20 -m 600 -o "$f.gz" "$BASE/$f.gz" && gunzip -f "$f.gz"; then echo "OK: $f"; else rm -f "$f.gz"; ok=0; fi
done
if [ $ok = 1 ]; then
  echo "Run: ./pathfinder --mode bench --gr data/USA-road-d.$REGION.gr --co data/USA-road-d.$REGION.co --queries 5"
else
  echo "DIMACS server unreachable -> downloading SNAP roadNet-PA instead"
  [ -s roadNet-PA.txt ] || (curl -fsSL --connect-timeout 20 -m 600 -o roadNet-PA.txt.gz https://snap.stanford.edu/data/roadNet-PA.txt.gz && gunzip -f roadNet-PA.txt.gz)
  echo "Run: ./pathfinder --mode bench --graph snap --file data/roadNet-PA.txt --directed --queries 5"
fi
