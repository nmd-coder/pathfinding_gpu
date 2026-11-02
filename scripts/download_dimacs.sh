#!/usr/bin/env bash
# Tải bản đồ DIMACS USA-road-d. Dùng: bash scripts/download_dimacs.sh NY
set -e
REGION=${1:-NY}
BASE="http://www.diag.uniroma1.it/challenge9/data/USA-road-d"
mkdir -p "$(dirname "$0")/../data" && cd "$(dirname "$0")/../data"
for ext in gr co; do
  f="USA-road-d.$REGION.$ext"
  [ -f "$f" ] && { echo "Exists: $f"; continue; }
  curl -L -o "$f.gz" "$BASE/$f.gz" && gunzip -f "$f.gz"
done
echo "Run: ./pathfinder --mode bench --gr data/USA-road-d.$REGION.gr --co data/USA-road-d.$REGION.co --queries 5"
