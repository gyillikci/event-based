#include "azimuth.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace ndrone {

namespace {
constexpr double kPi = 3.14159265358979323846;
double radians(double deg) { return deg * kPi / 180.0; }
double degrees(double rad) { return rad * 180.0 / kPi; }

// Extract a numeric JSON value for `key` from `text` (minimal, dependency-free
// parser sufficient for the flat calibration files this repo writes).
std::optional<double> json_number(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::nullopt;
    ++pos;
    while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])))) ++pos;
    size_t end = pos;
    while (end < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' ||
            text[end] == '+' || text[end] == '.' || text[end] == 'e' || text[end] == 'E')) {
        ++end;
    }
    if (end == pos) return std::nullopt;
    try {
        return std::stod(text.substr(pos, end - pos));
    } catch (...) {
        return std::nullopt;
    }
}
}  // namespace

double pixel_x_to_visual_azimuth_deg(double center_x_px, int width, double camera_hfov_deg) {
    const double half_hfov_rad = radians(camera_hfov_deg) / 2.0;
    if (camera_hfov_deg >= 179.0 || half_hfov_rad <= 0.0) {
        const double norm_x = (center_x_px / std::max(width - 1, 1)) * 2.0 - 1.0;
        return norm_x * (camera_hfov_deg / 2.0);
    }
    const double f_px = (width / 2.0) / std::tan(half_hfov_rad);
    const double dx = center_x_px - (width - 1) / 2.0;
    return degrees(std::atan2(dx, f_px));
}

double visual_azimuth_deg_to_pixel_x(double azimuth_deg, int width, double camera_hfov_deg) {
    const double half_hfov_rad = radians(camera_hfov_deg) / 2.0;
    double x;
    if (camera_hfov_deg >= 179.0 || half_hfov_rad <= 0.0) {
        const double norm_x = azimuth_deg / (camera_hfov_deg / 2.0);
        x = (norm_x + 1.0) / 2.0 * std::max(width - 1, 1);
    } else {
        const double f_px = (width / 2.0) / std::tan(half_hfov_rad);
        const double dx = f_px * std::tan(radians(azimuth_deg));
        x = dx + (width - 1) / 2.0;
    }
    return std::min(std::max(x, 0.0), static_cast<double>(width - 1));
}

double apply_visual_azimuth_calibration(double raw_azimuth_deg,
                                        const std::optional<AzimuthCalibration>& calib) {
    if (!calib.has_value()) return raw_azimuth_deg;
    return calib->gain * raw_azimuth_deg + calib->offset_deg;
}

std::optional<AzimuthCalibration> load_azimuth_calibration(const std::string& path) {
    if (path.empty()) return std::nullopt;
    std::ifstream f(path);
    if (!f) return std::nullopt;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    auto gain = json_number(text, "gain");
    if (!gain.has_value()) return std::nullopt;

    AzimuthCalibration cal;
    cal.gain = gain.value();
    cal.offset_deg = json_number(text, "offset_deg").value_or(0.0);
    cal.camera_hfov_deg = json_number(text, "camera_hfov_deg").value_or(220.0);
    return cal;
}

}  // namespace ndrone
