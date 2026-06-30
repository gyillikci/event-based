#!/usr/bin/env bash
echo "kernel: $(uname -r)"
echo "--- tools ---"
for t in git cmake gcc g++ make usbip; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "$t: $(command -v "$t")"
  else
    echo "$t: MISSING"
  fi
done
echo "--- libs (dpkg) ---"
for p in libfftw3-dev libconfig-dev libasound2-dev linux-tools-virtual; do
  if dpkg -s "$p" >/dev/null 2>&1; then
    echo "$p: installed"
  else
    echo "$p: MISSING"
  fi
done
echo "--- sound cards in WSL ---"
cat /proc/asound/cards 2>/dev/null || echo "no /proc/asound/cards"
