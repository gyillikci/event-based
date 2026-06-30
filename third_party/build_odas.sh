#!/usr/bin/env bash
set -e
cd ~
if [ -d odas ]; then
  echo "odas already cloned"
else
  git clone --depth 1 --branch master https://github.com/introlab/odas.git
fi
mkdir -p odas/build
cd odas/build
echo "=== cmake ==="
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -n 10
echo "=== make ==="
make -j"$(nproc)" 2>&1 | tail -n 25
echo "=== binaries ==="
ls -la ~/odas/build/bin/ 2>/dev/null || echo "no bin dir"
