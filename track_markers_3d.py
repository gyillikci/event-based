"""
6-DoF pose tracking of an active LED marker array with an event camera.

Each LED on the rig blinks at its own fixed frequency, so the event camera can
label every blob without any appearance matching: the frequency IS the ID. Once
the 2D image position of each identified LED is known, and the 3D position of
that LED on the rig is known from marker_array.json, the pair of them is a
classic PnP problem and the array's full 6-DoF pose follows.

    events -> per-pixel frequency map   (FrequencyMapAsyncAlgorithm, SDK)
           -> blobs                     (FrequencyMapAnalyzer, detect_propeller)
           -> temporal tracks           (PropellerTracker, detect_propeller)
           -> LED identity by frequency (marker_array.json)
           -> 6-DoF pose                (solvePnP)

WHAT THIS NEEDS THAT THE REPO DOES NOT YET HAVE:

  * At least 4 non-collinear LEDs. The Arduino firmware currently drives only 2
    spatially distinct sources (BUILTIN 800 Hz, GREEN 1500 Hz) because
    LEDR/LEDG/LEDB are three dies inside one package at one spot. Two points
    cannot determine a 6-DoF pose - rotation about the line joining them is
    unobservable - so this script will refuse rather than report a pose it
    cannot support. Wire 2 more LEDs to free pins, give them their own
    frequencies, and add them to both active_markers.json and marker_array.json.

  * Real measured LED coordinates in marker_array.json, and camera intrinsics.
    Both are refused while still placeholders: a wrong model does not fail
    loudly, it produces a confident wrong pose, which is worse.

Run --self-test to exercise the geometry and solver end to end with synthetic
data; it needs no camera, no recording and no hardware.

Usage:
    python track_markers_3d.py --self-test
    python track_markers_3d.py                          # live camera
    python track_markers_3d.py -i recording.raw         # from a file
    python track_markers_3d.py --csv pose.csv --no-display
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

DEFAULT_ARRAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marker_array.json")
DEFAULT_MARKERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_markers.json")

# An array whose LEDs are nearly collinear is degenerate for PnP: the pose can
# rotate about the common line without changing the image. Measured as the
# smallest singular value of the centred model points, in mm.
MIN_NONCOLLINEAR_MM = 2.0


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

class ArrayModel:
    """The rig: which LED blinks at which frequency, and where it sits in 3D."""

    def __init__(self, cfg, path):
        self.path = path
        self.name = cfg["array"].get("name", "unnamed")
        self.units = cfg.get("units", "mm")
        self.leds = {}
        for led in cfg["array"]["leds"]:
            self.leds[int(led["id"])] = {
                "id": int(led["id"]),
                "name": str(led.get("name", f"LED{led['id']}")),
                "freq_hz": float(led["freq_hz"]),
                "xyz": np.array(led["xyz_mm"], dtype=np.float64),
                "measured": bool(led.get("measured", False)),
                "color_bgr": tuple(int(c) for c in led.get("color_bgr", [0, 255, 0])),
            }
        if not self.leds:
            raise ValueError(f"{path}: array.leds is empty")

    def match_frequency(self, freq_hz, tol_hz):
        """Return the LED whose nominal frequency is nearest, or None."""
        best, best_diff = None, tol_hz
        for led in self.leds.values():
            diff = abs(freq_hz - led["freq_hz"])
            if diff <= best_diff:
                best, best_diff = led, diff
        return best

    def object_points(self, ids):
        return np.array([self.leds[i]["xyz"] for i in ids], dtype=np.float64)

    def check_usable(self, ids):
        """Raise unless these LEDs can actually support a 6-DoF solve.

        Every failure here is a configuration problem that would otherwise show
        up as a plausible-looking but wrong pose.
        """
        if len(ids) < 4:
            raise ValueError(
                f"need at least 4 identified LEDs for a 6-DoF pose, have {len(ids)}. "
                "3 LEDs give up to 4 candidate poses (P3P) with no way to choose between "
                "them; 2 LEDs leave rotation about their common axis unobservable.")

        unmeasured = [self.leds[i]["name"] for i in ids if not self.leds[i]["measured"]]
        if unmeasured:
            raise ValueError(
                f"{self.path}: LEDs {', '.join(unmeasured)} still have measured=false. "
                "Fill in real caliper coordinates and set measured=true - an unmeasured "
                "model yields a confident wrong pose, not an obvious failure.")

        pts = self.object_points(ids)
        centred = pts - pts.mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        # sv[1] is the spread perpendicular to the dominant axis: near zero means
        # every LED lies on one line.
        if sv[1] < MIN_NONCOLLINEAR_MM:
            raise ValueError(
                f"LED geometry is nearly collinear (perpendicular spread {sv[1]:.2f} mm). "
                "Spread the LEDs across two dimensions.")
        return True

    def is_planar(self, ids, tol_mm=0.5):
        pts = self.object_points(ids)
        centred = pts - pts.mean(axis=0)
        return np.linalg.svd(centred, compute_uv=False)[2] < tol_mm


def load_array_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg, ArrayModel(cfg, path)


def build_intrinsics(cam, width, height):
    """Return (K, dist, description).

    Prefers a calibrated matrix. Falls back to deriving one from the lens focal
    length and the sensor's pixel pitch, which assumes a perfect pinhole with the
    principal point at the image centre and no distortion.
    """
    dist = np.zeros((5, 1), dtype=np.float64)
    if cam.get("dist_coeffs"):
        dist = np.array(cam["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    if cam.get("camera_matrix"):
        K = np.array(cam["camera_matrix"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("camera.camera_matrix must be 3x3")
        return K, dist, "calibrated camera_matrix"

    f_mm = cam.get("lens_focal_length_mm")
    if f_mm:
        pitch_mm = float(cam["pixel_pitch_um"]) / 1000.0
        f_px = float(f_mm) / pitch_mm
        K = np.array([[f_px, 0.0, (width - 1) / 2.0],
                      [0.0, f_px, (height - 1) / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, dist, f"derived from {f_mm} mm lens at {cam['pixel_pitch_um']} um pitch ({f_px:.1f} px)"

    raise ValueError(
        "no camera intrinsics: set camera.camera_matrix (run metavision_mono_calibration "
        "and paste the result) or camera.lens_focal_length_mm in the array config. "
        "PnP converts pixels to angles through the intrinsics, so pose is meaningless without them.")


# --------------------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------------------

def rotation_to_euler_deg(R):
    """ZYX Euler angles (yaw about Z, pitch about Y, roll about X), degrees."""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                     # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([yaw, pitch, roll])


class PoseSolver:
    """PnP with the previous pose used as the initial guess for stability."""

    def __init__(self, model, K, dist, ransac_px=3.0, ambiguity_ratio=2.0):
        self.model = model
        self.K = K
        self.dist = dist
        self.ransac_px = ransac_px
        # Below this ratio between the two planar candidates' reprojection errors,
        # the better fit is not trusted on its own and continuity decides.
        self.ambiguity_ratio = ambiguity_ratio
        self.prev_rvec = None
        self.prev_tvec = None

    def solve(self, ids, image_points):
        """ids: LED ids, image_points: Nx2 pixel coords. Returns a pose dict or None."""
        object_points = self.model.object_points(ids)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

        planar = self.model.is_planar(ids)
        # IPPE is built for planar targets and is both faster and better conditioned
        # there; SQPNP is the general-position choice.
        flags = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
        ambiguity = float("inf")

        if len(ids) >= 6:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, self.dist,
                reprojectionError=self.ransac_px, flags=flags)
            inlier_idx = (inliers.ravel().tolist()
                          if (ok and inliers is not None) else list(range(len(ids))))
            inlier_ids = [ids[i] for i in inlier_idx]
        elif planar:
            # A planar target admits TWO poses that project almost identically - the
            # true one and its mirror through the plane. Near head-on their
            # reprojection errors are nearly equal, so picking the smaller one alone
            # makes the pose flip between frames. Ask for both, then let continuity
            # with the previous pose break the tie when the errors are close.
            ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                object_points, image_points, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok or not rvecs:
                return None
            cands = []
            for rv, tv in zip(rvecs, tvecs):
                rv, tv = cv2.solvePnPRefineLM(object_points, image_points, self.K, self.dist, rv, tv)
                pr, _ = cv2.projectPoints(object_points, rv, tv, self.K, self.dist)
                rms = float(np.sqrt(np.mean(np.linalg.norm(pr.reshape(-1, 2) - image_points, axis=1) ** 2)))
                cands.append((rms, rv, tv))
            cands.sort(key=lambda c: c[0])
            if len(cands) > 1 and cands[0][0] > 1e-9:
                ambiguity = cands[1][0] / cands[0][0]
            best = cands[0]
            if len(cands) > 1 and ambiguity < self.ambiguity_ratio and self.prev_rvec is not None:
                # Too close to call on reprojection error: keep the candidate whose
                # rotation is nearest the previous frame's.
                def rot_delta(rv):
                    R1, _ = cv2.Rodrigues(rv)
                    R0, _ = cv2.Rodrigues(self.prev_rvec)
                    return float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
                best = min(cands, key=lambda c: rot_delta(c[1]))
            _, rvec, tvec = best
            inlier_idx = list(range(len(ids)))
            inlier_ids = list(ids)
            ok = True
        else:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, self.K, self.dist, flags=flags)
            inlier_idx = list(range(len(ids)))
            inlier_ids = list(ids)

        if not ok:
            return None

        # Iterative LM refinement, seeded by the solution just found. Refine on the
        # INLIERS only: refining on every point would pull the pose back towards an
        # outlier that RANSAC had already rejected.
        obj_in = object_points[inlier_idx]
        img_in = image_points[inlier_idx]
        rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, self.K, self.dist, rvec, tvec)

        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, self.K, self.dist)
        residuals = np.linalg.norm(proj.reshape(-1, 2) - img_in, axis=1)
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        self.prev_rvec, self.prev_tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = rotation_to_euler_deg(R)
        t = tvec.ravel()
        return {
            "rvec": rvec, "tvec": tvec, "R": R,
            "x_mm": float(t[0]), "y_mm": float(t[1]), "z_mm": float(t[2]),
            "range_mm": float(np.linalg.norm(t)),
            "yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": float(roll),
            "reproj_rms_px": rms,
            "n_leds": len(ids),
            "n_inliers": len(inlier_ids),
            "planar": planar,
            # Ratio of the runner-up planar solution's reprojection error to the
            # winner's. Near 1.0 the two mirror poses are indistinguishable from this
            # frame alone and the orientation should be treated as suspect.
            "ambiguity": ambiguity,
        }


class TwoLedSolver:
    """What two LEDs of known separation can and cannot tell you.

    With only two points the full 6-DoF pose is out of reach, but three of the six
    degrees of freedom are still directly observable, and they are the useful ones
    for pointing a camera at a target:

      OBSERVABLE   bearing to the pair (azimuth / elevation), exactly - it is just
                   the direction of the rays, and needs no assumption at all.
      OBSERVABLE   roll about the optical axis, from the tilt of the LED pair in
                   the image.
      ASSUMED      range. The angle the pair subtends fixes range ONLY if the
                   baseline is perpendicular to the line of sight. Tilt the board
                   away by an angle a and the pair foreshortens, so the estimate
                   grows by 1/cos(a): 15 deg of tilt reads 3.5% long, 30 deg reads
                   15% long. The bias is always one way - never short.
      LOST         rotation about the line joining the two LEDs, and which way the
                   board is tilted. Nothing recovers these from two points.
    """

    def __init__(self, model, K, dist):
        self.model = model
        self.K = K
        self.dist = dist

    def solve(self, ids, image_points):
        if len(ids) != 2:
            return None
        p0, p1 = self.model.leds[ids[0]]["xyz"], self.model.leds[ids[1]]["xyz"]
        baseline_mm = float(np.linalg.norm(p0 - p1))
        if baseline_mm <= 0:
            return None

        pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
        # Undistort to normalised camera coordinates, so the bearings are correct
        # even off-axis and with a distorting lens.
        norm_pts = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        rays = np.hstack([norm_pts, np.ones((2, 1))])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        cos_theta = float(np.clip(np.dot(rays[0], rays[1]), -1.0, 1.0))
        theta = np.arccos(cos_theta)
        if theta < 1e-9:
            return None
        # Isoceles triangle: the known baseline subtends theta at the camera.
        range_mm = baseline_mm / (2.0 * np.tan(theta / 2.0))

        mid = rays.mean(axis=0)
        mid /= np.linalg.norm(mid)
        pos = range_mm * mid

        d = np.asarray(image_points[1], dtype=np.float64) - np.asarray(image_points[0], dtype=np.float64)
        roll_deg = float(np.degrees(np.arctan2(d[1], d[0])))

        return {
            "mode": "2led",
            "x_mm": float(pos[0]), "y_mm": float(pos[1]), "z_mm": float(pos[2]),
            "range_mm": float(range_mm),
            "azimuth_deg": float(np.degrees(np.arctan2(mid[0], mid[2]))),
            "elevation_deg": float(np.degrees(np.arctan2(-mid[1], mid[2]))),
            "roll_deg": roll_deg,
            "baseline_mm": baseline_mm,
            "pixel_sep": float(np.linalg.norm(d)),
            "subtend_deg": float(np.degrees(theta)),
            "n_leds": 2,
        }


def draw_two_led(img, sol, ids, image_points, model):
    for led_id, pt in zip(ids, image_points):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 7, model.leds[led_id]["color_bgr"], 2)
    a = tuple(int(v) for v in image_points[0])
    b = tuple(int(v) for v in image_points[1])
    cv2.line(img, a, b, (255, 255, 255), 1)
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    cv2.drawMarker(img, mid, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    lines = [
        f"2-LED mode  baseline {sol['baseline_mm']:.2f} mm  sep {sol['pixel_sep']:.1f} px",
        f"range {sol['range_mm']:7.1f} mm   az {sol['azimuth_deg']:+6.2f}  el {sol['elevation_deg']:+6.2f}  "
        f"roll {sol['roll_deg']:+7.2f} deg",
        "range assumes the LED pair is square-on; tilt reads long (1/cos)",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if i < 2 else (0, 200, 255), 1, cv2.LINE_AA)
    return img


def draw_pose(img, pose, K, dist, model, ids, image_points, axis_mm=30.0):
    """Overlay the array axes and the LED reprojection residuals."""
    axes = np.array([[0, 0, 0], [axis_mm, 0, 0], [0, axis_mm, 0], [0, 0, axis_mm]], dtype=np.float64)
    proj, _ = cv2.projectPoints(axes, pose["rvec"], pose["tvec"], K, dist)
    o, x, y, z = [tuple(int(v) for v in p.ravel()) for p in proj]
    cv2.line(img, o, x, (0, 0, 255), 2)     # X red
    cv2.line(img, o, y, (0, 255, 0), 2)     # Y green
    cv2.line(img, o, z, (255, 0, 0), 2)     # Z blue

    obj = model.object_points(ids)
    rep, _ = cv2.projectPoints(obj, pose["rvec"], pose["tvec"], K, dist)
    for led_id, meas, pr in zip(ids, image_points, rep.reshape(-1, 2)):
        col = model.leds[led_id]["color_bgr"]
        cv2.circle(img, (int(meas[0]), int(meas[1])), 6, col, 2)
        cv2.line(img, (int(meas[0]), int(meas[1])), (int(pr[0]), int(pr[1])), (255, 255, 255), 1)

    txt = (f"range {pose['range_mm']:7.1f} mm   "
           f"xyz ({pose['x_mm']:.0f},{pose['y_mm']:.0f},{pose['z_mm']:.0f})   "
           f"ypr ({pose['yaw_deg']:.1f},{pose['pitch_deg']:.1f},{pose['roll_deg']:.1f}) deg   "
           f"{pose['n_leds']} LEDs   reproj {pose['reproj_rms_px']:.2f} px")
    cv2.putText(img, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------------
# Self-test: the whole geometry + solver path, no hardware
# --------------------------------------------------------------------------------------

def self_test():
    """Project a known pose, recover it, and report the error.

    Also checks that the degenerate configurations this script is meant to refuse
    are actually refused.
    """
    print("track_markers_3d self-test")
    print("=" * 64)

    # A 4-LED planar rig, 80 x 60 mm, origin at its centre.
    leds = [(1, 800.0, [-40.0, -30.0, 0.0]),
            (2, 1100.0, [40.0, -30.0, 0.0]),
            (3, 1500.0, [40.0, 30.0, 0.0]),
            (4, 2000.0, [-40.0, 30.0, 0.0])]
    cfg = {"units": "mm",
           "camera": {"width": 1280, "height": 720, "pixel_pitch_um": 4.86,
                      "camera_matrix": None, "dist_coeffs": None, "lens_focal_length_mm": 16.0},
           "array": {"name": "self-test 80x60 planar",
                     "leds": [{"id": i, "name": f"L{i}", "freq_hz": f, "xyz_mm": p, "measured": True}
                              for i, f, p in leds]}}
    model = ArrayModel(cfg, "<self-test>")
    K, dist, how = build_intrinsics(cfg["camera"], 1280, 720)
    print(f"intrinsics: {how}")
    print(f"array     : {model.name}, {len(model.leds)} LEDs, planar={model.is_planar(list(model.leds))}")

    ids = sorted(model.leds)
    model.check_usable(ids)
    obj = model.object_points(ids)

    rng = np.random.default_rng(0)
    print(f"\n{'true pose (mm / deg)':<34} {'recovered':<34} {'err':>18}")
    print("-" * 90)

    worst_pos, worst_ang = 0.0, 0.0
    cases = [
        ("head-on 500 mm", [0.0, 0.0, 0.0], [0.0, 0.0, 500.0]),
        ("yaw 25 deg", [0.0, np.deg2rad(25), 0.0], [20.0, -10.0, 700.0]),
        ("tilted, 1.5 m", [np.deg2rad(15), np.deg2rad(-20), np.deg2rad(8)], [-50.0, 30.0, 1500.0]),
        ("close 250 mm", [np.deg2rad(-10), 0.0, np.deg2rad(30)], [0.0, 0.0, 250.0]),
    ]
    for name, rv, tv in cases:
        rvec = np.array(rv, dtype=np.float64).reshape(3, 1)
        tvec = np.array(tv, dtype=np.float64).reshape(3, 1)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        # 0.5 px of centroid noise: an LED blob in a frequency map is a cluster of
        # pixels, so its centroid is better than 1 px but not exact.
        noisy = proj.reshape(-1, 2) + rng.normal(0.0, 0.5, size=(len(ids), 2))

        solver = PoseSolver(model, K, dist)
        pose = solver.solve(ids, noisy)
        assert pose is not None, f"{name}: solver returned no pose"

        R_true, _ = cv2.Rodrigues(rvec)
        pos_err = float(np.linalg.norm(pose["tvec"].ravel() - tvec.ravel()))
        dR = R_true.T @ pose["R"]
        ang_err = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        worst_pos, worst_ang = max(worst_pos, pos_err), max(worst_ang, ang_err)

        true_s = f"{name}: z={tv[2]:.0f}"
        got_s = f"z={pose['z_mm']:7.1f} reproj={pose['reproj_rms_px']:.2f}px"
        print(f"{true_s:<34} {got_s:<34} {pos_err:7.2f} mm {ang_err:5.2f} deg")

    print("-" * 90)
    print(f"worst position error {worst_pos:.2f} mm, worst angle error {worst_ang:.2f} deg "
          f"(with 0.5 px centroid noise)")

    # A planar rig swept through head-on is where the mirror ambiguity bites: the
    # two IPPE candidates fit equally well, so a per-frame solver flips between
    # them. Track a slow sweep and count sign flips in pitch.
    print("\nplanar mirror ambiguity, pitch sweep -20 deg -> +20 deg through head-on:")
    for label, use_continuity in [("per-frame best fit", False), ("with continuity", True)]:
        solver = PoseSolver(model, K, dist)
        if not use_continuity:
            solver.ambiguity_ratio = 0.0        # never consult the previous pose
        errs, mirrors = [], 0
        for pitch_deg in np.linspace(-20, 20, 41):
            rvec = np.array([0.0, np.deg2rad(pitch_deg), 0.0]).reshape(3, 1)
            tvec = np.array([0.0, 0.0, 600.0]).reshape(3, 1)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            noisy = proj.reshape(-1, 2) + rng.normal(0.0, 0.5, size=(len(ids), 2))
            pose = solver.solve(ids, noisy)
            R_true, _ = cv2.Rodrigues(rvec)
            dR = R_true.T @ pose["R"]
            errs.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            # The mirror solution shows up as a pitch of the wrong sign. Compare
            # against the TRUE sign; the sweep's own crossing at 0 is not an error.
            # Skip near head-on, where the sign is genuinely undetermined.
            if abs(pitch_deg) > 4 and np.sign(pose["pitch_deg"]) != np.sign(pitch_deg):
                mirrors += 1
        print(f"  {label:<20} median angle err {np.median(errs):5.2f} deg, "
              f"max {np.max(errs):5.2f} deg, mirror-flipped frames {mirrors}/34")

    # Cover the other two solver paths: SQPNP for a non-planar rig, and the RANSAC
    # branch that engages at 6+ LEDs (here with one LED deliberately corrupted).
    print("\nother solver paths:")
    variants = [
        ("non-planar 5 LED (SQPNP)",
         [[-40, -30, 0], [40, -30, 0], [40, 30, 0], [-40, 30, 0], [0, 0, -25]], False, None),
        ("planar 6 LED (RANSAC)",
         [[-40, -30, 0], [0, -30, 0], [40, -30, 0], [40, 30, 0], [0, 30, 0], [-40, 30, 0]], False, None),
        ("planar 6 LED, 1 outlier",
         [[-40, -30, 0], [0, -30, 0], [40, -30, 0], [40, 30, 0], [0, 30, 0], [-40, 30, 0]], False, 2),
    ]
    for label, coords, _, outlier in variants:
        vcfg = {"units": "mm", "camera": cfg["camera"],
                "array": {"name": label, "leds": [
                    {"id": i, "name": f"L{i}", "freq_hz": 700.0 + 150 * i,
                     "xyz_mm": [float(v) for v in p], "measured": True}
                    for i, p in enumerate(coords)]}}
        vmodel = ArrayModel(vcfg, f"<{label}>")
        vids = sorted(vmodel.leds)
        vmodel.check_usable(vids)
        vobj = vmodel.object_points(vids)
        rvec = np.array([np.deg2rad(10), np.deg2rad(-15), np.deg2rad(5)]).reshape(3, 1)
        tvec = np.array([15.0, -20.0, 800.0]).reshape(3, 1)
        proj, _ = cv2.projectPoints(vobj, rvec, tvec, K, dist)
        pts = proj.reshape(-1, 2) + rng.normal(0.0, 0.5, size=(len(vids), 2))
        if outlier is not None:
            pts[outlier] += np.array([25.0, -18.0])      # a badly mislocated blob
        pose = PoseSolver(vmodel, K, dist).solve(vids, pts)
        R_true, _ = cv2.Rodrigues(rvec)
        pos_err = float(np.linalg.norm(pose["tvec"].ravel() - tvec.ravel()))
        ang_err = float(np.degrees(np.arccos(
            np.clip((np.trace(R_true.T @ pose["R"]) - 1) / 2, -1, 1))))
        print(f"  {label:<26} planar={str(pose['planar']):<5} "
              f"pos {pos_err:6.2f} mm  ang {ang_err:5.2f} deg  "
              f"inliers {pose['n_inliers']}/{pose['n_leds']}")

    # The 2-LED path, with the real Nano 33 BLE baseline and the 5 mm lens, since
    # that is what the hardware can actually do today.
    cam5 = dict(cfg["camera"], lens_focal_length_mm=5.0)
    K5, dist5, how5 = build_intrinsics(cam5, 1280, 720)
    two_cfg = {"units": "mm", "camera": cam5, "array": {"name": "nano33ble on-board", "leds": [
        {"id": 1, "name": "BUILTIN", "freq_hz": 800, "xyz_mm": [-19.69, 5.79, 0.0], "measured": True},
        {"id": 3, "name": "GREEN", "freq_hz": 1500, "xyz_mm": [5.06, -3.61, 0.0], "measured": True}]}}
    tmodel = ArrayModel(two_cfg, "<2led>")
    tids = sorted(tmodel.leds)
    tobj = tmodel.object_points(tids)
    baseline = float(np.linalg.norm(tobj[0] - tobj[1]))
    solver2 = TwoLedSolver(tmodel, K5, dist5)
    print(f"\n2-LED mode ({how5}, baseline {baseline:.2f} mm):")
    print(f"  {'true range':>11} {'tilt':>6} {'px sep':>7} {'est range':>10} {'error':>9} {'1/cos(tilt)':>12}")
    for true_range in (300.0, 600.0, 1000.0):
        for tilt_deg in (0.0, 15.0, 30.0):
            # Rotate the pair about the vertical axis: at 0 deg it is square-on.
            rvec = np.array([0.0, np.deg2rad(tilt_deg), 0.0]).reshape(3, 1)
            tvec = np.array([0.0, 0.0, true_range]).reshape(3, 1)
            proj, _ = cv2.projectPoints(tobj, rvec, tvec, K5, dist5)
            pts2 = proj.reshape(-1, 2) + rng.normal(0.0, 0.5, size=(2, 2))
            sol = solver2.solve(tids, pts2)
            err_pct = 100.0 * (sol["range_mm"] - true_range) / true_range
            print(f"  {true_range:>9.0f}mm {tilt_deg:>5.0f}d {sol['pixel_sep']:>7.1f} "
                  f"{sol['range_mm']:>9.1f}mm {err_pct:>+8.1f}% {1/np.cos(np.deg2rad(tilt_deg)):>11.3f}")

    # The refusals matter as much as the solve.
    print("\nrefusal checks:")
    for label, test_ids, expect in [("2 LEDs", ids[:2], "at least 4"),
                                    ("3 LEDs", ids[:3], "at least 4")]:
        try:
            model.check_usable(test_ids)
            print(f"  FAIL {label}: accepted, should have been refused")
            return 1
        except ValueError as e:
            assert expect in str(e)
            print(f"  ok   {label} refused: {str(e).split('.')[0]}")

    collinear = {"units": "mm", "camera": cfg["camera"],
                 "array": {"name": "collinear", "leds": [
                     {"id": i, "name": f"L{i}", "freq_hz": 800.0 + 100 * i,
                      "xyz_mm": [20.0 * i, 0.0, 0.0], "measured": True} for i in range(4)]}}
    try:
        ArrayModel(collinear, "<collinear>").check_usable([0, 1, 2, 3])
        print("  FAIL collinear array: accepted, should have been refused")
        return 1
    except ValueError as e:
        assert "collinear" in str(e)
        print(f"  ok   collinear array refused: {str(e).split('.')[0]}")

    unmeasured = {"units": "mm", "camera": cfg["camera"],
                  "array": {"name": "unmeasured", "leds": [
                      {"id": i, "name": f"L{i}", "freq_hz": 800.0 + 100 * i,
                       "xyz_mm": p, "measured": False}
                      for i, (_, _, p) in enumerate(leds)]}}
    try:
        ArrayModel(unmeasured, "<unmeasured>").check_usable([0, 1, 2, 3])
        print("  FAIL unmeasured model: accepted, should have been refused")
        return 1
    except ValueError as e:
        assert "measured=false" in str(e)
        print(f"  ok   unmeasured model refused: {str(e).split('.')[0]}")

    print("\nself-test passed")
    return 0


# --------------------------------------------------------------------------------------
# Live / file tracking
# --------------------------------------------------------------------------------------

def run_tracking(args):
    from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
    from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
    from detect_propeller import FrequencyMapAnalyzer, PropellerTracker

    cfg, model = load_array_config(args.array_config)

    # Cross-check against the frequency table the firmware also uses, so the two
    # files cannot silently drift apart.
    if os.path.exists(args.markers_config):
        with open(args.markers_config, "r", encoding="utf-8") as fh:
            table = {int(m["id"]): float(m["freq_hz"]) for m in json.load(fh)["markers"]}
        for led in model.leds.values():
            if led["id"] in table and abs(table[led["id"]] - led["freq_hz"]) > 1.0:
                print(f"warning: LED {led['id']} is {led['freq_hz']} Hz in {args.array_config} "
                      f"but {table[led['id']]} Hz in {args.markers_config}", file=sys.stderr)

    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=args.delta_t)
    height, width = mv_iterator.get_size()
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    K, dist, how = build_intrinsics(cfg["camera"], width, height)
    print(f"array      : {model.name} ({len(model.leds)} LEDs) from {args.array_config}")
    print(f"intrinsics : {how}")

    freqs = [led["freq_hz"] for led in model.leds.values()]
    min_freq = args.min_freq or max(50.0, min(freqs) - 200.0)
    max_freq = args.max_freq or (max(freqs) + 200.0)

    freq_algo = FrequencyMapAsyncAlgorithm(
        width=width, height=height, filter_length=args.filter_length,
        min_freq=min_freq, max_freq=max_freq, diff_thresh_us=args.max_period_diff)
    freq_algo.update_frequency = args.update_freq

    analyzer = FrequencyMapAnalyzer(
        width=width, height=height, min_freq=min_freq, max_freq=max_freq, num_blades=1,
        min_pixels=args.min_cluster_pixels, dilate_radius=args.dilate_radius,
        max_freq_cv=args.max_freq_cv)
    tracker = PropellerTracker(
        max_distance=args.track_distance, freq_tolerance=args.freq_tolerance,
        min_hits=args.min_hits, max_age=args.max_age,
        confidence_threshold=args.confidence_threshold)
    solver = PoseSolver(model, K, dist, ransac_px=args.ransac_px)
    two_led = TwoLedSolver(model, K, dist)

    csv = None
    if args.csv:
        csv = open(args.csv, "w", encoding="utf-8")
        csv.write("t_us,mode,n_leds,n_inliers,x_mm,y_mm,z_mm,range_mm,azimuth_deg,elevation_deg,"
                  "yaw_deg,pitch_deg,roll_deg,reproj_rms_px,ambiguity\n")

    window = None
    frame = np.zeros((height, width, 3), np.uint8)
    if not args.no_display:
        window = MTWindow(title="Active marker 3D pose", width=width, height=height,
                          mode=BaseWindow.RenderMode.BGR, open_directly=True)

        def keyboard_cb(key, scancode, action, mods):
            if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)

    frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                 fps=args.display_fps, palette=ColorPalette.Dark)
    latest_frame = {"img": None}

    def on_cd_frame(ts, cd_frame):
        latest_frame["img"] = cd_frame.copy()

    frame_gen.set_output_callback(on_cd_frame)

    state = {"pose": None, "ids": [], "pts": [], "checked": False, "refusal": None, "n_solved": 0,
             "n_frames": 0, "announced_2led": False}

    def on_freq_map(ts, freq_map):
        state["n_frames"] += 1
        confirmed = tracker.update(analyzer.analyze(freq_map))

        # One detection per LED id: if two blobs claim the same frequency, keep the
        # bigger one rather than feeding an ambiguous correspondence to PnP.
        best = {}
        for trk in confirmed:
            led = model.match_frequency(trk["freq_hz"], args.match_tolerance)
            if led is None:
                continue
            prev = best.get(led["id"])
            if prev is None or trk["pixels"] > prev["pixels"]:
                best[led["id"]] = trk

        ids = sorted(best)
        pts = [(best[i]["x"], best[i]["y"]) for i in ids]

        # 4+ LEDs -> full 6-DoF pose. Exactly 2 -> the reduced set of observables,
        # which is all that two points can support. Otherwise nothing this frame.
        pose = None
        if len(ids) >= 4:
            try:
                model.check_usable(ids)
                pose = solver.solve(ids, pts)
            except ValueError as e:
                if not state["checked"]:
                    state["checked"] = True
                    state["refusal"] = str(e)
        elif len(ids) == 2 and all(model.leds[i]["measured"] for i in ids):
            pose = two_led.solve(ids, pts)
            if pose is not None and not state["announced_2led"]:
                state["announced_2led"] = True
                print(f"2-LED mode: baseline {pose['baseline_mm']:.2f} mm. Bearing and roll are "
                      "exact; range assumes the pair is square-on and reads long by 1/cos(tilt). "
                      "Rotation about the LED axis is not observable.")

        if pose is not None:
            state["n_solved"] += 1
            if csv:
                if pose.get("mode") == "2led":
                    csv.write(f"{ts},2led,2,2,"
                              f"{pose['x_mm']:.2f},{pose['y_mm']:.2f},{pose['z_mm']:.2f},"
                              f"{pose['range_mm']:.2f},{pose['azimuth_deg']:.3f},"
                              f"{pose['elevation_deg']:.3f},,,{pose['roll_deg']:.2f},,\n")
                else:
                    csv.write(f"{ts},pnp,{pose['n_leds']},{pose['n_inliers']},"
                              f"{pose['x_mm']:.2f},{pose['y_mm']:.2f},{pose['z_mm']:.2f},"
                              f"{pose['range_mm']:.2f},,,"
                              f"{pose['yaw_deg']:.2f},{pose['pitch_deg']:.2f},"
                              f"{pose['roll_deg']:.2f},{pose['reproj_rms_px']:.3f},"
                              f"{pose['ambiguity']:.3f}\n")
        state.update(pose=pose, ids=ids, pts=pts)

    freq_algo.set_output_callback(on_freq_map)

    t_stop_us = args.duration_s * 1e6 if args.duration_s > 0 else None
    for evs in mv_iterator:
        if t_stop_us is not None and evs.size and evs["t"][-1] >= t_stop_us:
            break
        if window is not None:
            EventLoop.poll_and_dispatch()
            if window.should_close():
                break
        frame_gen.process_events(evs)
        freq_algo.process_events(evs)

        if window is not None:
            img = latest_frame["img"]
            frame[:] = img if img is not None else 0
            if state["refusal"]:
                for i, line in enumerate(_wrap(state["refusal"], 78)):
                    cv2.putText(frame, line, (10, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 1, cv2.LINE_AA)
            elif state["pose"] is not None and state["pose"].get("mode") == "2led":
                draw_two_led(frame, state["pose"], state["ids"], state["pts"], model)
            elif state["pose"] is not None:
                draw_pose(frame, state["pose"], K, dist, model, state["ids"], state["pts"],
                          axis_mm=args.axis_mm)
            else:
                cv2.putText(frame, f"{len(state['ids'])} LED(s) seen - need 2 (bearing+range) or 4 (pose)",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
            window.show_async(frame)

    if csv:
        csv.close()
        print(f"wrote {args.csv}")
    if state["refusal"]:
        print(f"\nno pose produced: {state['refusal']}", file=sys.stderr)
        return 2
    print(f"solved {state['n_solved']} of {state['n_frames']} frequency frames")
    return 0


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def parse_args():
    p = argparse.ArgumentParser(
        description="6-DoF pose tracking of an active LED marker array with an event camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="Validate the geometry and solver on synthetic data, then exit.")
    p.add_argument("-i", "--input-event-file", dest="event_file_path", default="",
                   help="RAW/DAT/HDF5 file. Default: live camera.")
    p.add_argument("--array-config", default=DEFAULT_ARRAY, help="3D array geometry + intrinsics (JSON).")
    p.add_argument("--markers-config", default=DEFAULT_MARKERS, help="Frequency table shared with the firmware.")
    p.add_argument("--csv", default="", help="Write the pose track to this CSV.")
    p.add_argument("--no-display", action="store_true", help="Headless.")
    p.add_argument("--duration-s", type=float, default=0.0,
                   help="Stop after this many seconds of stream time (0 = run until closed). "
                        "Use it for bounded test runs so the CSV is closed properly.")
    p.add_argument("--display-fps", type=float, default=25.0)
    p.add_argument("--axis-mm", type=float, default=30.0, help="Length of the drawn pose axes.")

    p.add_argument("--min-freq", type=float, default=0.0, help="0 = derive from the array's LEDs.")
    p.add_argument("--max-freq", type=float, default=0.0, help="0 = derive from the array's LEDs.")
    p.add_argument("--match-tolerance", type=float, default=120.0,
                   help="Max |detected - nominal| Hz to accept an LED identity.")
    p.add_argument("--delta-t", type=int, default=20000)
    p.add_argument("--replay-factor", type=float, default=1.0)
    p.add_argument("--filter-length", type=int, default=4)
    p.add_argument("--max-period-diff", type=int, default=100)
    p.add_argument("--update-freq", type=float, default=50.0)
    p.add_argument("--min-cluster-pixels", type=int, default=3)
    p.add_argument("--dilate-radius", type=int, default=5)
    p.add_argument("--max-freq-cv", type=float, default=0.3)
    p.add_argument("--track-distance", type=float, default=40.0)
    p.add_argument("--freq-tolerance", type=float, default=100.0)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--max-age", type=int, default=5)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--ransac-px", type=float, default=3.0,
                   help="RANSAC reprojection threshold, used when 6+ LEDs are visible.")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.self_test:
        sys.exit(self_test())
    try:
        sys.exit(run_tracking(a))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
