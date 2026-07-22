// Pixel <-> azimuth mapping and visual->acoustic linear calibration.
// Port of the azimuth helpers in detect_drone_fused.py.
#pragma once

#include <optional>
#include <string>

namespace ndrone {

// Loaded visual->acoustic azimuth calibration model.
struct AzimuthCalibration {
    double gain = 1.0;
    double offset_deg = 0.0;
    double camera_hfov_deg = 220.0;
};

// Map an image column to camera azimuth (deg) using a rectilinear (pinhole)
// projection, falling back to a linear map for fisheye-style FOVs (>=179 deg).
double pixel_x_to_visual_azimuth_deg(double center_x_px, int width, double camera_hfov_deg);

// Inverse: map an azimuth back to an image column, clamped to [0, width-1].
double visual_azimuth_deg_to_pixel_x(double azimuth_deg, int width, double camera_hfov_deg);

// Apply the linear calibration az_cal = gain*az_raw + offset (identity if none).
double apply_visual_azimuth_calibration(double raw_azimuth_deg,
                                        const std::optional<AzimuthCalibration>& calib);

// Load a calibration model from JSON (reads gain, offset_deg, camera_hfov_deg).
// Returns nullopt if the path is empty, missing, or malformed.
std::optional<AzimuthCalibration> load_azimuth_calibration(const std::string& path);

}  // namespace ndrone
