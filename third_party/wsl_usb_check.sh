#!/usr/bin/env bash
echo "--- lsusb ---"
lsusb 2>&1 || echo "lsusb failed"
echo "--- vhci_hcd ---"
if [ -e /sys/devices/platform/vhci_hcd.0 ]; then echo "vhci present"; else echo "vhci MISSING"; fi
echo "--- usbip tool ---"
command -v usbip || echo "usbip MISSING"
echo "--- sound cards ---"
cat /proc/asound/cards 2>/dev/null || echo "no /proc/asound/cards"
echo "--- arecord -l ---"
arecord -l 2>&1 || echo "arecord missing"
