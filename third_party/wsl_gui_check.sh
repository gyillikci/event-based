#!/usr/bin/env bash
echo "DISPLAY=$DISPLAY"
if [ -d /mnt/wslg ]; then echo "WSLg present"; else echo "no WSLg"; fi
echo "--- node ---"
which node && node -v || echo "no node"
which npm && npm -v || echo "no npm"
echo "--- linux electron in odas_web ---"
ls -l /mnt/c/Users/z003n5uc/Desktop/event-based/third_party/odas_web/node_modules/.bin/electron 2>/dev/null || echo "no linux electron"
echo "--- gui libs ---"
ldconfig -p | grep -E 'libgtk-3|libnss3|libgbm' | head || echo "no gui libs"
