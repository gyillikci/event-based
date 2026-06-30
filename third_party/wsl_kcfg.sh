#!/usr/bin/env bash
echo "--- kernel config ---"
if [ -f /proc/config.gz ]; then
  zcat /proc/config.gz | grep -E 'CONFIG_USBIP_VHCI_HCD|CONFIG_SND_USB_AUDIO|CONFIG_USB_SUPPORT|CONFIG_SND=' || echo "(none of those symbols found)"
else
  echo "no /proc/config.gz"
fi
echo "--- modprobe vhci-hcd ---"
if modprobe vhci-hcd 2>/dev/null; then echo "vhci loaded OK"; else echo "vhci modprobe failed"; fi
echo "--- modprobe snd-usb-audio ---"
if modprobe snd-usb-audio 2>/dev/null; then echo "snd-usb-audio loaded OK"; else echo "snd-usb-audio modprobe failed"; fi
