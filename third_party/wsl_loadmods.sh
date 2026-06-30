#!/usr/bin/env bash
depmod -a 2>&1
modprobe vhci-hcd 2>&1 && echo "vhci-hcd: LOADED" || echo "vhci-hcd: FAILED"
modprobe snd-usb-audio 2>&1 && echo "snd-usb-audio: LOADED" || echo "snd-usb-audio: FAILED"
echo "--- lsmod ---"
lsmod | grep -E 'vhci|snd_usb' || echo "(not in lsmod)"
echo "--- vhci sysfs ---"
if [ -e /sys/devices/platform/vhci_hcd.0 ]; then echo "vhci_hcd.0 present"; else echo "vhci_hcd.0 MISSING"; fi
