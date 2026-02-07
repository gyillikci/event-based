/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

// Example of using Metavision SDK to visualize an EVK3D stream as a depthmap.

#include <sstream>
#include <iomanip>

#include <chrono>
#include <fstream>
#include <map>
#include <numeric>
#include <signal.h>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/persistence.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/highgui/highgui_c.h>
#endif

#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/base/utils/sdk_log.h>
#include <metavision/sdk/base/events/event_pointcloud.h>
#include <metavision/sdk/core/utils/cv_color_map.h>
#include <metavision/sdk/core/utils/misc.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/hal/facilities/i_event_frame_decoder.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/psee_hw_layer/devices/evk3d/evk3d_projector.h>
#include <metavision/psee_hw_layer/devices/evk3d/evk3d_mtr.h>

static const int ESCAPE                                = 27;
static const int SPACE                                 = 32;
static constexpr std::uint16_t kNumDigitsPtCldFilename = 8;

namespace po = boost::program_options;

int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key   = cv::waitKey(delay_ms);
    auto now  = std::chrono::high_resolution_clock::now();
    // cv::waitKey will not wait if no window is opened, so we wait for it, if needed
    std::this_thread::sleep_for(std::chrono::milliseconds(
        delay_ms - std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count()));

    return key;
}

namespace {
std::atomic<bool> signal_caught{false};

[[maybe_unused]] void signalHandler(int s) {
    MV_LOG_TRACE() << "Interrupt signal received." << std::endl;
    signal_caught = true;
}
} // anonymous namespace

class UndistortionProjection {
public:
    UndistortionProjection(const cv::Mat &camera_matrix, const std::vector<float> &dist_coefficients) :
        K_(camera_matrix), D_(dist_coefficients) {}

    inline bool operator()(const Metavision::PointCloud::Point3D &p, Metavision::PointCloud::Point3D &out) const {
        if (std::abs(p.Z) <= 1e-10f) {
            // Avoid division by 0
            return false;
        }

        cv::Point2f evt_undist_norm;
        evt_undist_norm.x = p.X / p.Z;
        evt_undist_norm.y = p.Y / p.Z;

        float x  = evt_undist_norm.x;
        float y  = evt_undist_norm.y;
        float x2 = x * x;
        float y2 = y * y;
        float r2 = x2 + y2, r4 = r2 * r2, r6 = r2 * r4;
        float c_raddist   = 1 + D_[0] * r2 + D_[1] * r4 + D_[4] * r6;
        float x_dist_norm = x * c_raddist + 2 * D_[2] * x * y + D_[3] * (r2 + 2 * x2);
        float y_dist_norm = y * c_raddist + D_[2] * (r2 + 2 * y2) + 2 * D_[3] * x * y;

        out   = p;
        out.X = K_.at<double>(0, 0) * x_dist_norm + K_.at<double>(0, 1) * y_dist_norm + K_.at<double>(0, 2);
        out.Y = K_.at<double>(1, 1) * y_dist_norm + K_.at<double>(1, 2);

        return true;
    }

private:
    cv::Mat K_;
    std::vector<float> D_;
};

class PointCloudExporter {
public:
    PointCloudExporter(const std::string &out_path, const std::string &extension) :
        out_folder_(out_path), extension_(extension) {}

    void process_point_cloud(const Metavision::PointCloud &pc) {
        std::ostringstream oss;
        oss << std::setw(kNumDigitsPtCldFilename) << std::setfill('0') << cnt_++;
        const std::string filename = out_folder_ + oss.str() + "." + extension_;

        std::ofstream pcf;
        pcf.open(filename);
        if (!pcf.is_open()) {
            MV_SDK_LOG_ERROR() << "Impossible to create file" << filename << "aborting write point cloud.";
            return;
        }

        const auto &pts   = pc.points;
        const size_t size = pts.size();

        if (extension_ == ".pcd")
            write_header_pcd(pcf, size);
        else
            write_header_ply(pcf, size);

        pcf.precision(8);
        for (size_t k = 0; k < size; ++k) {
            pcf << pts[k].X << " " << pts[k].Y << " " << pts[k].Z << "\n";
        }
        pcf.close();
    }

private:
    void write_header_pcd(std::ofstream &pcf, int size) {
        pcf << "# .PCD v.7 - Point Cloud Data file format - "
               "http://pointclouds.org/documentation/tutorials/pcd_file_format.php\n";
        pcf << "VERSION .7\n";
        pcf << "FIELDS x y z\n";
        pcf << "SIZE 4 4 4\n";
        pcf << "TYPE F F F\n";
        pcf << "COUNT 1 1 1\n";
        pcf << "WIDTH " << size << "\n";
        pcf << "HEIGHT 1\n";
        pcf << "POINTS " << size << "\n";
        pcf << "DATA ascii\n";
    }

    void write_header_ply(std::ofstream &pcf, int size) {
        pcf << "ply\n";
        pcf << "format ascii 1.0\n";
        pcf << "element vertex " << size << "\n";
        pcf << "property float32 xn";
        pcf << "property float32 y\n";
        pcf << "property float32 z\n";
        pcf << "end_header\n";
    }

    const std::string out_folder_;
    const std::string extension_;
    std::uint16_t cnt_ = 0;
};

class DepthMapGenerator {
public:
    DepthMapGenerator(unsigned fps, float min_depth, float max_depth, unsigned line_thickness,
                      const Metavision::Geometry &g, std::unique_ptr<UndistortionProjection> point_projector) :
        fps_(fps),
        min_depth_(min_depth),
        max_depth_(max_depth),
        line_thickness_(line_thickness),
        geometry_(g),
        point_projector_(std::move(point_projector)),
        color_map_(cv::COLORMAP_JET),
        last_frame_time_(std::chrono::high_resolution_clock::now()) {
        grey_depthmap_frame_.create(geometry_.height(), geometry_.width(), CV_8UC1);
    }

    inline void process_pointcloud(const Metavision::PointCloud &pc, cv::Mat &out) {
        const float ref_depth = max_depth_ > 0 ? max_depth_ : pc.max_depth;
        grey_depthmap_frame_.setTo(0);
        for (auto &pt : pc.points) {
            Metavision::PointCloud::Point3D point;
            if (point_projector_) {
                if (!(*point_projector_)(pt, point)) {
                    continue;
                }
                if (point.X < 0 || point.X >= geometry_.width() || point.Y < 0 || point.Y >= geometry_.height()) {
                    continue;
                }
            } else {
                point = pt;
            }

            const float depth = std::min(std::max(point.Z - min_depth_, 0.f), ref_depth - min_depth_);
            const float val   = 255.f * (1.f - depth / (ref_depth - min_depth_));
            for (std::size_t x = std::max(0.f, point.X - line_thickness_ / 2);
                 x <= std::min<unsigned int>(geometry_.width() - 1, point.X + line_thickness_ / 2); ++x) {
                if (grey_depthmap_frame_.at<uint8_t>(point.Y, x) < val) {
                    grey_depthmap_frame_.at<uint8_t>(point.Y, x) = val;
                }
            }
        }

        color_map_(grey_depthmap_frame_, out);

        wait_for_framerate();
    }

private:
    inline void wait_for_framerate() {
        while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     last_frame_time_)
                   .count() < 1000 / fps_) {}
        last_frame_time_ = std::chrono::high_resolution_clock::now();
    }

    const unsigned fps_;
    const float min_depth_, max_depth_;
    const unsigned line_thickness_;
    const Metavision::Geometry &geometry_;

    std::unique_ptr<UndistortionProjection> point_projector_;

    cv::Mat grey_depthmap_frame_;
    Metavision::CvColorMap color_map_;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time_;
};

int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);

    std::string serial;
    std::string cam_config_path;
    std::string in_file_path;
    std::string out_file_path;
    std::string calib_file_path;
    float max_depth;
    float min_depth;
    unsigned line_thickness;
    std::string mode;
    float x_scale, y_scale, z_scale;
    unsigned fps;
    std::string pt_clouds_path;
    std::string pt_clouds_format;

    const std::vector<std::string> supported_modes = {"MTR", "MTRMetric", "MTRHomogeneous", "MTRScaled",
                                                      "MTRHomogeneousScaled"};
    bool do_retry                                  = false;

    const std::string short_program_desc(
        "Example of using Metavision SDK to visualize an EVK3D stream as a depth map.\n");
    std::string long_program_desc(short_program_desc +
                                  "Press SPACE key while running to record or stop recording raw data\n"
                                  "Press 'q' or Escape key to leave the program.\n"
                                  "Press 'h' to print this help.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",              po::value<std::string>(&serial),"Serial ID of the camera. This flag is incompatible with flag '--input-event-file'.")
        ("input-event-file,i",    po::value<std::string>(&in_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("output-file,o",         po::value<std::string>(&out_file_path)->default_value("data.raw"), "Path to an output file used for data recording. Default value is 'data.raw'. It also works when reading data from a file.")
        ("pt-cloud-folder",       po::value<std::string>(&pt_clouds_path)->default_value(""), "Provide a path to a folder to save the point clouds.")
        ("pt-cloud-format",       po::value<std::string>(&pt_clouds_format)->default_value("PLY"), "Type of file to save (choose between 'PLY' and 'PCD'")
        ("calib-file,c",          po::value<std::string>(&calib_file_path), "JSON file containing EVK3D calibration data")
        ("max-depth,d",           po::value<float>(&max_depth)->default_value(-1), "Max depth in meters expected to be seen on the scene, default is sensor dependent")
        ("min-depth",             po::value<float>(&min_depth)->default_value(0), "Min depth in meters expected to be seen on the scene, default is 0")
        ("line-thickness,l",      po::value<unsigned>(&line_thickness)->default_value(1), "Pixel thickness used to visualize EVK3D projector lines")
        ("mode,m",                po::value<std::string>(&mode)->default_value("MTR"),
         (std::string("Output format mode, should be one of [") + std::accumulate(supported_modes.begin() + 1, supported_modes.end(), supported_modes[0],
                                                                                  [](const std::string &a, const std::string &b) {return a + ", " + b;})
          + "]").c_str())
        ("x-scale",               po::value<float>(&x_scale)->default_value(0.0), "X scaling for MTR scaled modes")
        ("y-scale",               po::value<float>(&y_scale)->default_value(0.0), "Y scaling for MTR scaled modes")
        ("z-scale",               po::value<float>(&z_scale)->default_value(0.0), "z scaling for MTR scaled modes")
        ("fps",                   po::value<unsigned>(&fps)->default_value(60), "FPS rate for depthmaps")
    ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    if (std::find(supported_modes.begin(), supported_modes.end(), mode) == supported_modes.end()) {
        MV_LOG_ERROR() << "Unsupported MTR mode: " + mode;
        MV_LOG_ERROR() << options_desc;
        return 1;
    }

    MV_LOG_INFO() << long_program_desc;

    if (calib_file_path.empty() && in_file_path.empty()) {
        MV_LOG_WARNING() << "No calibration file provided for live camera. Visualization might be incoherent.";
    }

    if (pt_clouds_path != "") {
        if (!std::filesystem::is_directory(pt_clouds_path)) {
            MV_LOG_ERROR() << "Provided path for point cloud export is invalid !";
            return 2;
        }
        if (pt_clouds_format != "PLY" && pt_clouds_format != "PCD") {
            MV_LOG_ERROR() << "The point cloud file extension should be 'PLY' or 'PCD'! Got '" + pt_clouds_format +
                                  "'.";
            return 2;
        }
    }

    do {
        Metavision::Camera camera;
        bool camera_is_opened = false;

        // If the filename is set, then read from the file
        if (!in_file_path.empty()) {
            if (!serial.empty()) {
                MV_LOG_ERROR() << "Options --serial and --input-event-file are not compatible.";
                return 1;
            }

            try {
                Metavision::FileConfigHints hints;
                camera           = Metavision::Camera::from_file(in_file_path, hints);
                camera_is_opened = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
            // Otherwise, set the input source to a camera
        } else {
            try {
                Metavision::DeviceConfig config;
                config.set("format", mode);

                if (!serial.empty()) {
                    camera = Metavision::Camera::from_serial(serial, config);
                } else {
                    camera = Metavision::Camera::from_first_available(config);
                }

                if (!cam_config_path.empty()) {
                    camera.load(cam_config_path);
                }

                camera_is_opened = true;
            } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
        }

        // With the HAL device corresponding to the camera object (file or live camera), we can try to get a facility
        // This gives us access to extra HAL features not covered by the SDK Driver camera API
        try {
            auto *plugin_sw_info = camera.get_device().get_facility<Metavision::I_PluginSoftwareInfo>();
            if (plugin_sw_info) {
                const std::string &plugin_name = plugin_sw_info->get_plugin_name();
                MV_LOG_INFO() << "Plugin used to open the device:" << plugin_name;
            }
        } catch (const Metavision::CameraException &) {
            // we ignore the exception as some devices will not provide this facility (e.g. HDF5 files)
        }

        if (!camera_is_opened) {
            if (do_retry) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                MV_LOG_INFO() << "Trying to reopen camera...";
                continue;
            } else {
                return -1;
            }
        } else {
            MV_LOG_INFO() << "Camera has been opened successfully.";
        }

        std::unique_ptr<PointCloudExporter> pt_cloud_exporter;
        if (pt_clouds_path != "")
            pt_cloud_exporter =
                std::make_unique<PointCloudExporter>(pt_clouds_path, (pt_clouds_format == "PLY") ? "ply" : "pcd");

        auto projector = camera.get_device().get_facility<Metavision::EVK3DProjector>();
        if (projector) {
            projector->enable(true);
            for (unsigned int i = 0; i < projector->get_number_channels(); ++i) {
                projector->enable_channel(i, true);
            }

            // Frame period us = channel period * num_channels * 2 (= pulses per channel)
            unsigned long long channel_period = 1000000ULL / (fps * 16);

            if (channel_period > projector->get_max_channel_period()) {
                channel_period = projector->get_max_channel_period();
            }
            projector->set_channel_period(channel_period);
            projector->set_pulse_level(2);
            projector->set_power(1023);
        }

        std::unique_ptr<UndistortionProjection> point_projector;
        auto mtr = camera.get_device().get_facility<Metavision::EVK3DMTR>();
        if (calib_file_path != "") {
            cv::FileStorage calib_data;
            calib_data.open(calib_file_path, cv::FileStorage::READ);

            std::map<std::string, std::vector<std::vector<float>>> lut_values;
            if (!calib_data["bf_fitted"].isNone()) {
                calib_data["bf_fitted"] >> lut_values["bf_fitted"];
            }
            if (!calib_data["x_inf_fitted"].isNone()) {
                calib_data["x_inf_fitted"] >> lut_values["x_inf_fitted"];
            }
            if (!calib_data["left_or_rights"].isNone()) {
                calib_data["left_or_rights"] >> lut_values["left_or_rights"];
            }

            if (mtr && !mtr->load_calibration(lut_values)) {
                MV_LOG_ERROR() << "Issue loading MTR calibration";
            }

            std::map<std::string, std::vector<float>> undistortion_params;
            cv::Mat camera_mat;
            if (!calib_data["camera_matrix"].isNone()) {
                calib_data["camera_matrix"] >> camera_mat;
            }
            cv::Mat Kinv = camera_mat.inv();
            Kinv.convertTo(Kinv, CV_32F);
            undistortion_params["camera_matrix_inverted"].assign(Kinv.ptr<float>(),
                                                                 Kinv.ptr<float>() + Kinv.total() * Kinv.channels());

            cv::Mat dist_coeffs;
            if (!calib_data["camera_distortion_coefficients"].isNone()) {
                calib_data["camera_distortion_coefficients"] >> dist_coeffs;
                dist_coeffs.convertTo(dist_coeffs, CV_32F);
                undistortion_params["distortion_coefficients"].assign(
                    dist_coeffs.ptr<float>(), dist_coeffs.ptr<float>() + dist_coeffs.total() * dist_coeffs.channels());
            }

            if (camera.get_device()
                        .get_facility<Metavision::I_HW_Identification>()
                        ->get_current_data_encoding_format() != "MTR" &&
                !calib_data["camera_matrix"].isNone() && !calib_data["camera_distortion_coefficients"].isNone()) {
                point_projector = std::make_unique<UndistortionProjection>(
                    camera_mat, undistortion_params["distortion_coefficients"]);
            }

            if (mtr && mtr->has_undistortion()) {
                mtr->set_x_scale_factor(x_scale);
                mtr->set_y_scale_factor(y_scale);
                mtr->set_z_scale_factor(z_scale);

                if (!mtr->load_undistortion_calibration(undistortion_params)) {
                    MV_LOG_ERROR() << "Issue loading MTRU calibration";
                }
            }
        }

        // Add runtime error callback
        camera.add_runtime_error_callback([&do_retry](const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            do_retry = true;
        });

        // Get the geometry of the camera
        auto &geometry = camera.geometry();

        std::string window_name = "Depthmap";

        std::mutex frame_mutex;
        cv::Mat depthmap_frame;
        cv::Mat display_frame;
        DepthMapGenerator depthmap_generator(fps, min_depth, max_depth, line_thickness, geometry,
                                             std::move(point_projector));

        int pointcloud_cb_id = -1;
        // Setup point cloud callbacks
        auto pc_decoder = camera.get_device().get_facility<Metavision::I_EventFrameDecoder<Metavision::PointCloud>>();
        if (!pc_decoder) {
            MV_LOG_ERROR() << "No pointcloud decoder found.";
            return 1;
        }

        pointcloud_cb_id =
            pc_decoder->add_event_frame_callback([&depthmap_frame, &frame_mutex, &depthmap_generator, &display_frame,
                                                  &pt_cloud_exporter](const Metavision::PointCloud &pc) {
                if (pt_cloud_exporter)
                    pt_cloud_exporter->process_point_cloud(pc);

                depthmap_generator.process_pointcloud(pc, depthmap_frame);

                std::lock_guard<std::mutex> lock(frame_mutex);
                depthmap_frame.copyTo(display_frame);
            });

        cv::namedWindow(window_name, CV_GUI_EXPANDED);
        cv::resizeWindow(window_name, geometry.width(), geometry.height());
        cv::moveWindow(window_name, 0, 0);
#if (CV_MAJOR_VERSION == 3 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 408) || \
    (CV_MAJOR_VERSION == 4 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 102)
        cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 1);
#endif

        auto stream = camera.get_device().get_facility<Metavision::I_EventsStream>();

        // Start the camera streaming
        camera.start();

        bool recording  = false;
        bool is_roi_set = true;

        while (!signal_caught && camera.is_running()) {
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (!display_frame.empty()) {
                    cv::imshow(window_name, display_frame);
                }
            }

            // Wait for a pressed key for 33ms, that means that the display is refreshed at 30 FPS
            int key = processUI(33);
            switch (key) {
            case 'q':
            case ESCAPE:
                camera.stop();
                do_retry = false;
                break;
            case SPACE:
                if (!recording) {
                    MV_LOG_INFO() << "Started recording in" << out_file_path;
                    camera.start_recording(out_file_path);
                } else {
                    MV_LOG_INFO() << "Stopped recording in" << out_file_path;
                    camera.stop_recording(out_file_path);
                }
                recording = !recording;
                break;
            case 'h':
                MV_LOG_INFO() << long_program_desc;
                break;
            default:
                break;
            }
        }

        if (pointcloud_cb_id >= 0) {
            pc_decoder->remove_callback(pointcloud_cb_id);
        }

        // Stop the camera streaming, optional, the destructor will automatically do it
        camera.stop();
        if (projector) {
            projector->enable(false);
        }
    } while (!signal_caught && do_retry);

    return signal_caught;
}
