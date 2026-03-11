"""
Generate 4 ArUco markers from the ARUCO_MIP_36h12 dictionary
(6x6 bits, IDs 0-249), save them individually and as a combined sheet.

Usage:
    python generate_aruco_markers.py
    python generate_aruco_markers.py --ids 0 5 10 20 --size 300 --output markers/
"""

import argparse
import os
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ARUCO_MIP_36h12 markers (6x6, IDs 0-249)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ids", type=int, nargs=4, default=[0, 1, 2, 3],
                        metavar="ID",
                        help="4 marker IDs to generate (must be in range 0-249)")
    parser.add_argument("--size", type=int, default=300,
                        help="Side length of each marker image in pixels")
    parser.add_argument("--border", type=int, default=1,
                        help="Width of the black border (in cells)")
    parser.add_argument("--output", type=str, default="aruco_markers",
                        help="Output directory for saved marker PNG files")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip showing the preview window")
    return parser.parse_args()


def get_aruco_mip_36h12_dict():
    """Return the ARUCO_MIP_36h12 dictionary (DICT_6X6_250 in OpenCV)."""
    # OpenCV's DICT_6X6_250 is the ARUCO_MIP_36h12 family:
    # 6x6 bit markers, 250 IDs (0-249), min Hamming distance = 12.
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


def generate_marker(aruco_dict, marker_id: int, size: int, border_bits: int) -> np.ndarray:
    """Draw a single marker and return it as a grayscale numpy array."""
    # generateImageMarker is the up-to-date API (OpenCV >= 4.7)
    # Falls back to drawMarker for older versions.
    try:
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size,
                                            borderBits=border_bits)
    except AttributeError:
        img = aruco_dict.drawMarker(marker_id, size, borderBits=border_bits)

    # Add a white padding so printed markers have a quiet zone
    pad = size // 10
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=255)
    return img_padded


def build_sheet(markers: list[np.ndarray], ids: list[int]) -> np.ndarray:
    """Arrange 4 markers in a 2×2 grid with ID labels."""
    assert len(markers) == 4
    rows = []
    for r in range(2):
        row_imgs = []
        for c in range(2):
            idx = r * 2 + c
            m = cv2.cvtColor(markers[idx], cv2.COLOR_GRAY2BGR)
            # Label below the marker
            label = f"ID: {ids[idx]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = m.shape[0] / 400.0
            thickness = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            bar = np.ones((th + 20, m.shape[1], 3), dtype=np.uint8) * 230
            cv2.putText(bar, label,
                        ((m.shape[1] - tw) // 2, th + 8),
                        font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
            row_imgs.append(np.vstack([m, bar]))
        rows.append(np.hstack(row_imgs))
    sheet = np.vstack(rows)
    return sheet


def main():
    args = parse_args()

    # Validate IDs
    for mid in args.ids:
        if not (0 <= mid <= 249):
            raise ValueError(f"Marker ID {mid} is out of range [0, 249] for ARUCO_MIP_36h12")

    os.makedirs(args.output, exist_ok=True)

    aruco_dict = get_aruco_mip_36h12_dict()
    print(f"Dictionary : ARUCO_MIP_36h12 / DICT_6X6_250  (6×6 bits, IDs 0-249)")
    print(f"Marker size: {args.size}px  |  Border: {args.border} cell(s)")
    print(f"IDs        : {args.ids}")
    print()

    markers = []
    for mid in args.ids:
        img = generate_marker(aruco_dict, mid, args.size, args.border)
        markers.append(img)

        out_path = os.path.join(args.output, f"aruco_6x6_id{mid:03d}.png")
        cv2.imwrite(out_path, img)
        print(f"  Saved marker ID {mid:3d}  →  {out_path}")

    # Save combined 2×2 sheet
    sheet = build_sheet(markers, args.ids)
    sheet_path = os.path.join(args.output, "aruco_6x6_sheet.png")
    cv2.imwrite(sheet_path, sheet)
    print(f"\n  Saved combined sheet  →  {sheet_path}")

    # Display preview
    if not args.no_display:
        win_name = "ARUCO_MIP_36h12  (6×6)  —  press any key to close"
        # Resize for screen if very large
        max_dim = 900
        h, w = sheet.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            sheet_disp = cv2.resize(sheet, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            sheet_disp = sheet
        cv2.imshow(win_name, sheet_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
