#!/usr/bin/env python3
"""
Align the GREEN reference LED with the camera's optical (z) axis.

The green LED marks the direction the camera is looking (centre of view).
Rotate it around the ring with the keyboard until it physically points at the
camera, then save the index/angle as the camera-forward reference.

Keys:
  a / LEFT   move green LED one step counter-clockwise
  d / RIGHT  move green LED one step clockwise
  s / ENTER  save this LED as the camera-forward reference and exit
  q / ESC    quit without saving

Saved to camera_led_ref.json:  {"led_index", "angle_deg", "num_leds"}
"""

import json
import sys

try:
    import msvcrt  # Windows keyboard input
except ImportError:
    msvcrt = None

from respeaker_led_control import find_device, PixelRing, NUM_LEDS

REF_PATH = "camera_led_ref.json"
DEG_PER_LED = 360.0 / NUM_LEDS


def render(ring, idx):
    colors = [(0, 0, 0)] * NUM_LEDS
    colors[idx] = (0, 255, 0)
    ring.customize(colors)
    print(f"\rGreen LED -> index {idx:2d}  ({idx * DEG_PER_LED:6.1f} deg)   "
          "[a/d move, s save, q quit]", end="", flush=True)


def read_key():
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):          # arrow-key prefix
        ext = msvcrt.getch()
        return {b"K": "left", b"M": "right"}.get(ext, "")
    return {b"a": "left", b"d": "right", b"s": "save", b"\r": "save",
            b"q": "quit", b"\x1b": "quit"}.get(ch.lower(), "")


def main():
    if msvcrt is None:
        sys.exit("Interactive alignment needs Windows (msvcrt).")

    dev = find_device()
    ring = PixelRing(dev)
    ring.set_brightness(12)

    idx = 0
    print("Rotate the GREEN LED until it points at the camera (centre of view).")
    render(ring, idx)

    while True:
        action = read_key()
        if action == "left":
            idx = (idx - 1) % NUM_LEDS
            render(ring, idx)
        elif action == "right":
            idx = (idx + 1) % NUM_LEDS
            render(ring, idx)
        elif action == "save":
            ref = {"led_index": idx,
                   "angle_deg": idx * DEG_PER_LED,
                   "num_leds": NUM_LEDS}
            with open(REF_PATH, "w") as f:
                json.dump(ref, f, indent=2)
            print(f"\nSaved camera-forward reference: LED {idx} "
                  f"({idx * DEG_PER_LED:.1f} deg) -> {REF_PATH}")
            return
        elif action == "quit":
            print("\nQuit without saving.")
            return


if __name__ == "__main__":
    main()
