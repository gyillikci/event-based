#!/usr/bin/env python3
"""
ReSpeaker USB Mic Array v2.0 — onboard DSP / LED ring controller.

The round ReSpeaker carries an XMOS XVF-3000 DSP. Its firmware runs its OWN
direction-of-arrival (DOA) estimator and voice-activity detector, and drives the
12-LED APA102 ring to point at whatever it hears. That behaviour is the factory
"trace" mode and is completely independent of any host-side processing.

Over the device's vendor-specific USB interface we can:
  * READ the firmware's DOA angle and voice-activity flag (Tuning)
  * TAKE OVER the LED ring and drive it ourselves (PixelRing)

This lets you either watch what the card thinks, or override the ring so it
points at the azimuth your own pipeline (e.g. detect_drone_fused.py) computed.

Requires:  pip install pyusb libusb-package
On Windows no Zadig driver swap is needed when libusb_package supplies the
backend; on Linux add a udev rule or run with sudo.

Usage examples:
  python respeaker_led_control.py --mode trace          # firmware auto-points at sound
  python respeaker_led_control.py --mode read           # print firmware DOA + VAD
  python respeaker_led_control.py --mode follow         # we light the LED at firmware DOA
  python respeaker_led_control.py --mode point --angle 90
  python respeaker_led_control.py --mode align         # find --led0-offset / direction
  python respeaker_led_control.py --mode off
"""

import argparse
import json
import os
import struct
import sys
import time

DEFAULT_CAMERA_REF = "camera_led_ref.json"

try:
    import usb.core
    import usb.util
    import libusb_package
except ImportError:
    sys.exit("Missing deps. Run:  pip install pyusb libusb-package")

VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018
NUM_LEDS = 12
TIMEOUT = 8000  # ms


# ----------------------------------------------------------------------------
# DSP parameter read/write (subset of the XVF-3000 tuning table)
# ----------------------------------------------------------------------------
# Each entry: (id, offset, type, max, min, access)
PARAMETERS = {
    "DOAANGLE":      (21, 0, "int", 359, 0, "ro"),   # estimated direction, degrees
    "VOICEACTIVITY": (19, 32, "int", 1, 0, "ro"),    # 1 = speech/sound present
    "SPEECHDETECTED":(19, 22, "int", 1, 0, "ro"),
    "AGCGAIN":       (19, 5, "float", 1000, 0, "ro"),
}


class Tuning:
    """Reads the firmware's DSP parameters (DOA angle, VAD, ...)."""

    def __init__(self, dev):
        self.dev = dev

    def read(self, name):
        data = PARAMETERS[name]
        param_id = data[0]
        cmd = 0x80 | data[1]
        if data[2] == "int":
            cmd |= 0x40
        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, param_id, 8, TIMEOUT)
        response = struct.unpack(b"ii", response.tobytes())
        if data[2] == "int":
            return response[0]
        return response[0] * (2.0 ** response[1])

    @property
    def direction(self):
        return self.read("DOAANGLE")

    @property
    def voice_activity(self):
        return self.read("VOICEACTIVITY")


# ----------------------------------------------------------------------------
# LED ring control (APA102 x12 driven by the XMOS chip)
# ----------------------------------------------------------------------------
class PixelRing:
    """Drives the 12-LED ring over the vendor control endpoint."""

    def __init__(self, dev):
        self.dev = dev

    def _write(self, cmd, data=(0,)):
        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, 0x1C, data, TIMEOUT)

    def trace(self):
        """Hand the ring back to firmware DOA mode (auto-point at sound)."""
        self._write(0)

    def mono(self, r, g, b):
        self._write(1, [r, g, b, 0])

    def off(self):
        self.mono(0, 0, 0)

    def spin(self):
        self._write(5)

    def set_brightness(self, value):
        self._write(0x20, [int(max(0, min(31, value)))])

    def set_center_led(self, on):
        self._write(0x22, [1 if on else 0])

    def customize(self, colors):
        """colors: list of NUM_LEDS (r,g,b) tuples."""
        data = []
        for (r, g, b) in colors[:NUM_LEDS]:
            data += [int(r), int(g), int(b), 0]
        self._write(6, data)

    def point_at(self, angle_deg, rgb=(0, 255, 255), led0_offset_deg=0.0,
                 clockwise=True, beam_width=1):
        """Light the LED(s) nearest `angle_deg` (host override of the ring).

        angle_deg       : bearing to indicate (0-360, 0 = LED0 direction)
        led0_offset_deg : physical mounting offset of LED index 0
        clockwise       : LED index increase direction around the ring
        beam_width      : extra LEDs lit on each side of the main one
        """
        rel = (angle_deg - led0_offset_deg) % 360.0
        if not clockwise:
            rel = (360.0 - rel) % 360.0
        center = int(round(rel / (360.0 / NUM_LEDS))) % NUM_LEDS
        colors = [(0, 0, 0)] * NUM_LEDS
        for d in range(-beam_width, beam_width + 1):
            idx = (center + d) % NUM_LEDS
            scale = 1.0 if d == 0 else 0.25
            colors[idx] = (int(rgb[0] * scale), int(rgb[1] * scale),
                           int(rgb[2] * scale))
        self.customize(colors)
        return center


# ----------------------------------------------------------------------------
def find_device():
    dev = libusb_package.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        sys.exit("ReSpeaker USB Mic Array v2.0 (2886:0018) not found.")
    return dev


def load_camera_ref(path):
    """Load the camera-forward reference and return (led0_offset_deg, forward_led).

    The ref marks which LED points along the camera's optical (+z) axis.
    We derive the led0_offset that makes a host bearing of 0 deg light that
    forward LED, so all `point_at` bearings are relative to the camera axis.
    Returns (None, None) if the file is missing or invalid.
    """
    if not path or not os.path.isfile(path):
        return None, None
    try:
        with open(path) as f:
            ref = json.load(f)
    except (OSError, ValueError):
        return None, None
    num = int(ref.get("num_leds", NUM_LEDS))
    fwd = ref.get("forward_led", ref.get("led_index"))
    if fwd is None:
        return None, None
    deg_per_led = 360.0 / num
    forward_angle = float(fwd) * deg_per_led
    led0_offset = (-forward_angle) % 360.0
    return led0_offset, int(fwd)


def run_alignment(tuning, ring, clockwise, args):
    """Interactive alignment to find --led0-offset and ring direction.

    Two stages:
      1. SWEEP   — light each LED in turn so you can see which physical LED is
                   index 0 and which way the index increases.
      2. CAPTURE — play a steady sound from a known bearing; the helper reads
                   the firmware DOA and computes the --led0-offset that makes a
                   host-driven dot land on the same direction the firmware picks.

    The recommended offset and direction are printed at the end so you can pass
    them straight to detect_drone_fused.py.
    """
    print("=" * 70)
    print("  ReSpeaker LED ring alignment helper")
    print("=" * 70)

    # --- Stage 1: visual sweep so the user can see ring geometry ---
    print("\n[Stage 1] Sweeping LEDs 0..11 (white). Watch the ring.")
    print("          Note where LED #0 sits and which way the dot travels.")
    for i in range(NUM_LEDS):
        colors = [(0, 0, 0)] * NUM_LEDS
        colors[i] = (80, 80, 80)
        ring.customize(colors)
        print(f"\r          LED #{i:2d} lit "
              f"(nominal bearing {i * 360.0 / NUM_LEDS:5.1f} deg)",
              end="", flush=True)
        time.sleep(0.35)
    ring.off()
    print("\n          Sweep done.")

    # --- Stage 2: capture firmware DOA for a known physical bearing ---
    print("\n[Stage 2] Point a steady sound source (e.g. the running fan) at a")
    print("          KNOWN bearing relative to the array, then keep it there.")
    try:
        known = float(input("          Enter that true bearing in degrees "
                            "(0-359), or blank to skip: ").strip() or "nan")
    except ValueError:
        known = float("nan")

    if known != known:  # NaN -> user skipped
        print("\n          Skipped capture. Use Stage-1 observations to set")
        print("          --led0-offset and --led-counter-clockwise manually.")
        return

    import math as _m

    print("          Reading firmware DOA for ~10 s (keep the tone steady; "
          "Ctrl+C to stop early)...")
    print("          NOTE: a loud steady tone is read continuously; the card's")
    print("          VAD flag is ignored because it is tuned for speech.")
    samples = []

    def _circ_mean(vals):
        cx = sum(_m.cos(_m.radians(v)) for v in vals)
        cy = sum(_m.sin(_m.radians(v)) for v in vals)
        return _m.degrees(_m.atan2(cy, cx)) % 360.0

    def _circ_dist(a, b):
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)

    try:
        t_end = time.time() + 10.0
        while time.time() < t_end:
            doa_now = tuning.direction
            samples.append(doa_now)
            # mirror onto ring so the user sees it tracking
            ring.point_at(doa_now, rgb=(0, 255, 255),
                          led0_offset_deg=0.0, clockwise=clockwise)
            print(f"\r          samples={len(samples):3d}  "
                  f"last DOA={doa_now:3d} deg", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    ring.off()

    if not samples:
        print("\n          No DOA samples captured. Check the USB control "
              "channel and rerun --mode align.")
        return

    # Robust central bearing: circular mean, then reject outliers >40 deg and
    # recompute so a few wild readings cannot drag the estimate.
    rough = _circ_mean(samples)
    inliers = [s for s in samples if _circ_dist(s, rough) <= 40.0]
    if len(inliers) >= max(5, len(samples) // 4):
        fw_doa = _circ_mean(inliers)
    else:
        inliers = samples
        fw_doa = rough
    spread = max((_circ_dist(s, fw_doa) for s in inliers), default=0.0)
    print(f"\n          inliers {len(inliers)}/{len(samples)}  "
          f"spread +/-{spread:.0f} deg")
    if spread > 30.0:
        print("          WARNING: DOA is unstable (>30 deg spread). Use a")
        print("          louder/closer tone or a more reflective-free spot,")
        print("          then rerun --mode align for a trustworthy offset.")

    # The offset that aligns OUR host-driven dot is the gap between the
    # firmware's reported angle and the physical bearing the user supplied.
    recommended_offset = (fw_doa - known) % 360.0
    if recommended_offset > 180.0:
        recommended_offset -= 360.0

    print("\n" + "-" * 70)
    print(f"  Samples used         : {len(samples)}")
    print(f"  Firmware DOA (mean)  : {fw_doa:6.1f} deg")
    print(f"  True bearing (you)   : {known:6.1f} deg")
    print(f"  => --led0-offset     : {recommended_offset:6.1f}")
    print(f"  => direction         : "
          f"{'(default CW)' if clockwise else '--led-counter-clockwise'}")
    print("-" * 70)
    print("  Verify with:")
    print(f"    python respeaker_led_control.py --mode point --angle {known:.0f} "
          f"--led0-offset {recommended_offset:.0f}"
          f"{'' if clockwise else ' --counter-clockwise'}")
    print("  The lit LED should face your sound source. If it sits on the")
    print("  opposite side, add/remove --counter-clockwise and rerun align.")


def main():
    p = argparse.ArgumentParser(description="ReSpeaker v2.0 DSP/LED controller")
    p.add_argument("--mode",
                   choices=["trace", "read", "follow", "point", "spin", "off", "align"],
                   default="read")
    p.add_argument("--angle", type=float, default=0.0,
                   help="Bearing for --mode point (degrees).")
    p.add_argument("--brightness", type=int, default=12,
                   help="LED brightness 0-31.")
    p.add_argument("--led0-offset", type=float, default=None,
                   help="Physical mounting offset of LED index 0 (deg). "
                        "Overrides the camera-ref derived offset.")
    p.add_argument("--camera-ref", default=DEFAULT_CAMERA_REF,
                   help="Path to camera_led_ref.json (LED aligned with camera "
                        "+z axis). Used as the 0-deg bearing reference.")
    p.add_argument("--no-camera-ref", action="store_true",
                   help="Ignore camera_led_ref.json and use --led0-offset only.")
    p.add_argument("--counter-clockwise", action="store_true",
                   help="Reverse the ring direction if the dot moves the wrong way.")
    p.add_argument("--hz", type=float, default=10.0,
                   help="Poll rate for read/follow modes.")
    p.add_argument("--ignore-vad", action="store_true",
                   help="In --mode follow, light the DOA LED continuously even "
                        "when the firmware VAD flag is off (use for steady "
                        "tones the speech VAD ignores).")
    args = p.parse_args()

    # Resolve led0_offset: explicit flag wins, else camera-ref, else legacy 60.
    ref_offset, ref_fwd = (None, None)
    if not args.no_camera_ref:
        ref_offset, ref_fwd = load_camera_ref(args.camera_ref)
    if args.led0_offset is None:
        if ref_offset is not None:
            args.led0_offset = ref_offset
            print(f"Camera-forward reference: LED {ref_fwd} (offset "
                  f"{ref_offset:.1f} deg) from {args.camera_ref}.")
        else:
            args.led0_offset = 60.0

    dev = find_device()
    tuning = Tuning(dev)
    ring = PixelRing(dev)
    ring.set_brightness(args.brightness)
    cw = not args.counter_clockwise

    try:
        if args.mode == "trace":
            ring.trace()
            print("Ring handed to firmware DOA mode. It now auto-points at sound.")
            print("Ctrl+C to exit (mode persists on the board).")
            while True:
                time.sleep(1)

        elif args.mode == "spin":
            ring.spin()
            print("Spinning. Ctrl+C to exit.")
            while True:
                time.sleep(1)

        elif args.mode == "off":
            ring.off()
            print("Ring off.")

        elif args.mode == "point":
            idx = ring.point_at(args.angle, led0_offset_deg=args.led0_offset,
                                clockwise=cw)
            print(f"Lit LED #{idx} for bearing {args.angle:.0f} deg.")

        elif args.mode == "read":
            period = 1.0 / max(args.hz, 0.5)
            print("Reading firmware DOA + VAD. Ctrl+C to exit.")
            while True:
                doa = tuning.direction
                vad = tuning.voice_activity
                bar = "SOUND" if vad else "  -  "
                print(f"\rDOA = {doa:3d} deg   VAD [{bar}]", end="", flush=True)
                time.sleep(period)

        elif args.mode == "follow":
            period = 1.0 / max(args.hz, 0.5)
            print("Following firmware DOA on the ring (cyan). Ctrl+C to exit.")
            while True:
                doa = tuning.direction
                vad = tuning.voice_activity
                if vad or args.ignore_vad:
                    ring.point_at(doa, rgb=(0, 255, 255),
                                  led0_offset_deg=args.led0_offset, clockwise=cw)
                else:
                    ring.off()
                print(f"\rDOA = {doa:3d} deg   VAD [{'SOUND' if vad else '  -  '}]",
                      end="", flush=True)
                time.sleep(period)

        elif args.mode == "align":
            run_alignment(tuning, ring, cw, args)

    except KeyboardInterrupt:
        print("\nExiting; returning ring to firmware trace mode.")
        try:
            ring.trace()
        except Exception:
            pass


if __name__ == "__main__":
    main()
