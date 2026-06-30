#!/usr/bin/env bash
echo "kernel: $(uname -r)"
echo "--- /lib/modules contents ---"
ls -la /lib/modules/ 2>&1 || echo "no /lib/modules"
echo "--- find target modules ---"
find /lib/modules -name 'vhci-hcd*' -o -name 'snd-usb-audio*' 2>/dev/null | head -20
echo "--- modules.dep ---"
ls -la "/lib/modules/$(uname -r)/modules.dep" 2>&1 || echo "no modules.dep for running kernel"
echo "--- any usbip on host C: drive? ---"
ls -la /mnt/c/Windows/System32/lxss/tools/ 2>/dev/null | head || echo "no lxss tools dir"
