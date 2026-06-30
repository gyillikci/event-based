#!/usr/bin/env bash
# Load USB/audio modules (idempotent) and run odaslive with the socket config.
modprobe vhci-hcd 2>/dev/null
modprobe snd-usb-audio 2>/dev/null
CFG="/mnt/c/Users/z003n5uc/Desktop/event-based/third_party/odas_respeaker_socket.cfg"
echo "config: $CFG"
exec ~/odas/build/bin/odaslive -c "$CFG"
