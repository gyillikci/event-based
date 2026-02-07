/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to create a pipeline displaying the results of dense optical flow algorithms.
#include <iostream>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/cv/algorithms/dense_flow_frame_generator_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/plane_fitting_flow_algorithm.h>
#include <metavision/sdk/cv/algorithms/time_gradient_flow_algorithm.h>
#include <metavision/sdk/cv/algorithms/triplet_matching_flow_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

namespace po = boost::program_options;

enum class DenseFlowType { PlaneFitting, TripletMatching, TimeGradient };

const std::unordered_map<DenseFlowType, std::string> kDenseFlowTypeToStr = {
    {DenseFlowType::PlaneFitting, "PlaneFitting"},
    {DenseFlowType::TripletMatching, "TripletMatching"},
    {DenseFlowType::TimeGradient, "TimeGradient"}};
const std::unordered_map<std::string, DenseFlowType> kStrToDenseFlowType = {
    {"PlaneFitting", DenseFlowType::PlaneFitting},
    {"TripletMatching", DenseFlowType::TripletMatching},
    {"TimeGradient", DenseFlowType::TimeGradient}};

std::istream &operator>>(std::istream &is, DenseFlowType &type) {
    std::string s;
    is >> s;
    auto it = kStrToDenseFlowType.find(s);
    if (it == kStrToDenseFlowType.cend())
        throw std::runtime_error("Failed to convert string to DenseFlowType");
    type = it->second;
    return is;
}

std::ostream &operator<<(std::ostream &os, const DenseFlowType &type) {
    auto it = kDenseFlowTypeToStr.find(type);
    if (it == kDenseFlowTypeToStr.cend())
        throw std::runtime_error("Failed to convert DenseFlowType to string");
    os << it->second;
    return os;
}

struct UserArguments {
    std::string cam_config_path;
    std::string event_file_path;
    std::string out_avi_file_path;
    uint32_t sw_stc_threshold;
    DenseFlowType flow_type;
    float receptive_field_radius;
    float min_flow_mag, max_flow_mag;
    bool benchmark  = false;
    bool no_display = false;
    bool no_legend  = false;
    Metavision::timestamp processing_period;
    float visualization_flow_scale;
};

int main(int argc, char *argv[]) {
    UserArguments args;

    const std::string program_desc(
        "Code sample showing how to use Metavision SDK to display results of dense optical flow algorithms.\n");

    po::options_description options_desc;
    po::options_description io_options("Input/Output options");
    // clang-format off
    io_options.add_options()
        ("help,h", "Produce help message.")
        ("input-camera-config,j",  po::value<std::string>(&args.cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("input-event-file,i", po::value<std::string>(&args.event_file_path)->default_value(""), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("output-avi-file,o", po::value<std::string>(&args.out_avi_file_path)->default_value(""), "Path to output AVI file to save the display video to. When the display is disabled, frames are not generated, so the video can't be generated either")
        ;
    // clang-format on

    po::options_description display_options("Display options");
    // clang-format off
    display_options.add_options()
        ("no-display,d", po::bool_switch(&args.no_display), "Disable output display window")
        ("no-legend", po::bool_switch(&args.no_legend), "Disable legend visualization in the display window")
        ("visu-scale", po::value<float>(&args.visualization_flow_scale)->default_value(0.8f), "Scale on flow magnitude visualization, in px/s.")
        ("benchmark", po::bool_switch(&args.benchmark), "Configure pipeline to skip costly visualizations to enable timing of the dense flow algorithm specifically")
        ;
    // clang-format on

    po::options_description processing_options("Processing options");
    // clang-format off
    processing_options.add_options()
        ("sw-stc-threshold,t", po::value<uint32_t>(&args.sw_stc_threshold)->default_value(10000), "Software STC threshold in us (10000 by default, disabled if sensor's STC is enabled or if the threshold is set to 0)")
        ("flow-type", po::value<DenseFlowType>(&args.flow_type)->default_value(DenseFlowType::TripletMatching), "Chosen type of dense flow algorithm to run, in {PlaneFitting, TripletMatching, TimeGradient}")
        ("receptive-field-radius,r", po::value<float>(&args.receptive_field_radius)->default_value(3), "Radius of the receptive field, in pixels, used for flow estimation and converted for each method into the relevant radius.")
        ("min-flow", po::value<float>(&args.min_flow_mag)->default_value(10), "Minimum observable flow magnitude, in px/s.")
        ("max-flow", po::value<float>(&args.max_flow_mag)->default_value(1000), "Maximum observable flow magnitude, in px/s.")
        ("processing-period", po::value<Metavision::timestamp>(&args.processing_period)->default_value(33333), "Period for slicing and processing events and for generating flow visualization, in us.")
        ;
    // clang-format on

    options_desc.add(io_options).add(display_options).add(processing_options);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    const bool enable_display      = !args.no_display && !args.benchmark;
    const bool enable_video_writer = !args.out_avi_file_path.empty() && !args.benchmark;
    const bool enable_visu         = enable_display || enable_video_writer;

    // Initialize the camera
    Metavision::Camera camera;
    if (args.event_file_path.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
        if (!args.cam_config_path.empty()) {
            try {
                camera.load(args.cam_config_path);
            } catch (const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
        }
    } else {
        try {
            camera = Metavision::Camera::from_file(args.event_file_path,
                                                   Metavision::FileConfigHints().real_time_playback(!args.benchmark));
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    }
    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    /// Data flow:
    //
    //  0 (Cam) -->-- 1 (buffer reslicer) -->-- 2 (STC) ----------->-----------  3 (Flow)
    //                                          |                                |
    //                                          v                                v
    //                                          |                                |
    //                                          4 (CD Frame Generator)           5 (Flow Frame Generator)
    //                                          |                                |
    //                                          v                                v
    //                                          |------>------->-<--------<------|
    //                                                          |
    //                                                          v
    //                                                          |
    //                                          |------<-------<->-------->------|
    //                                          |                                |
    //                                          v                                v
    //                                          |                                |
    //                                          6 (Display)                      7 (Video writer)
    //

    // We prepare a software STC algo to use if sensor's hardware STC is not enabled and software threshold not 0
    Metavision::SpatioTemporalContrastAlgorithm stc_algo(width, height, args.sw_stc_threshold);

    auto *event_trail_filter        = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();
    const bool hardware_stc_enabled = event_trail_filter && event_trail_filter->is_enabled();
    const bool bypass_software_stc  = hardware_stc_enabled || args.sw_stc_threshold == 0;

    // Instantiate the flow estimation algorithms
    // The input receptive field radius represents the total area of the neighborhood that is used to estimate flow. We
    // use an algorithm-dependent heuristic to convert this into the search radius to be used for each algorithm.
    std::unique_ptr<Metavision::PlaneFittingFlowAlgorithm> plane_fitting_flow_algo;
    std::unique_ptr<Metavision::TimeGradientFlowAlgorithm> time_gradient_flow_algo;
    std::unique_ptr<Metavision::TripletMatchingFlowAlgorithm> triplet_matching_flow_algo;
    switch (args.flow_type) {
    case DenseFlowType::PlaneFitting: {
        const int radius = cvFloor(args.receptive_field_radius);
        MV_LOG_INFO() << "Instantiating PlaneFittingFlowAlgorithm with radius=" << radius;
        plane_fitting_flow_algo = std::make_unique<Metavision::PlaneFittingFlowAlgorithm>(width, height, radius, -1);
        break;
    }
    case DenseFlowType::TripletMatching: {
        const float radius = 0.5f * args.receptive_field_radius;
        MV_LOG_INFO() << "Instantiating TripletMatchingFlowAlgorithm with radius=" << radius;
        Metavision::TripletMatchingFlowAlgorithmConfig triplet_matching_config(radius, args.min_flow_mag,
                                                                               args.max_flow_mag);
        triplet_matching_flow_algo =
            std::make_unique<Metavision::TripletMatchingFlowAlgorithm>(width, height, triplet_matching_config);
        break;
    }
    case DenseFlowType::TimeGradient: {
        const int radius = static_cast<int>(args.receptive_field_radius);
        MV_LOG_INFO() << "Instantiating TimeGradientFlowAlgorithm with radius=" << radius;
        Metavision::TimeGradientFlowAlgorithmConfig time_gradient_config(radius, args.min_flow_mag, 2);
        time_gradient_flow_algo =
            std::make_unique<Metavision::TimeGradientFlowAlgorithm>(width, height, time_gradient_config);
        break;
    }
    default:
        throw std::runtime_error("Selected DenseFlowType is not implemented!");
    }

    //  Create a video from the generated frames
    std::unique_ptr<cv::VideoWriter> video_writer;
    if (enable_video_writer) {
        video_writer = std::make_unique<cv::VideoWriter>(
            args.out_avi_file_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
    }

    // Initialize the window if not disabled
    std::unique_ptr<Metavision::Window> window;
    if (enable_display) {
        window = std::make_unique<Metavision::Window>("Dense Optical Flow", width, height,
                                                      Metavision::BaseWindow::RenderMode::BGR);
        window->set_keyboard_callback(
            [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window->set_close_flag();
                }
            });
    }

    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> cd_framer;
    std::unique_ptr<Metavision::DenseFlowFrameGeneratorAlgorithm> flow_framer;
    cv::Mat legend_frame, legend_mask;
    if (enable_visu) {
        cd_framer   = std::make_unique<Metavision::OnDemandFrameGenerationAlgorithm>(width, height, 0,
                                                                                   Metavision::ColorPalette::Dark);
        flow_framer = std::make_unique<Metavision::DenseFlowFrameGeneratorAlgorithm>(
            width, height, args.max_flow_mag, args.visualization_flow_scale,
            Metavision::DenseFlowFrameGeneratorAlgorithm::VisualizationMethod::Arrows,
            Metavision::DenseFlowFrameGeneratorAlgorithm::AccumulationPolicy::Average);
        if (enable_display && !args.no_legend) {
            flow_framer->generate_legend_image(legend_frame);
            legend_mask = legend_frame != cv::Scalar(0, 0, 0);
            cv::cvtColor(legend_mask, legend_mask, cv::COLOR_BGR2GRAY);
            legend_mask(cv::Rect((legend_mask.rows >> 1) - 1, (legend_mask.cols >> 1) - 1, 3, 3))
                .setTo(1); // Cover central part of the legend which is black to avoid overlay with arrows
        }
    }

    cv::Mat visu_frame(height, width, CV_8UC3);
    Metavision::EventBufferReslicerAlgorithm reslicer(
        [&](Metavision::EventBufferReslicerAlgorithm::ConditionStatus s, Metavision::timestamp t, std::size_t n) {
            if (!enable_visu)
                return;

            cd_framer->generate(t, visu_frame, false);
            flow_framer->generate(visu_frame, false);
            if (enable_display) {
                if (!args.no_legend)
                    legend_frame.copyTo(visu_frame(cv::Rect(0, visu_frame.rows - 1 - legend_frame.rows,
                                                            legend_frame.cols, legend_frame.rows)),
                                        legend_mask);
                window->show(visu_frame);
            }
            if (enable_video_writer) {
                video_writer->write(visu_frame);
            }
        });
    reslicer.set_slicing_condition(
        Metavision::EventBufferReslicerAlgorithm::Condition::make_n_us(args.processing_period));

    Metavision::timestamp ts_first = -1, ts_last = -1;
    std::vector<Metavision::EventCD> cd_buffer;
    std::vector<Metavision::EventOpticalFlow> flow_buffer;

    using VecIt        = std::vector<Metavision::EventCD>::const_iterator;
    using FlowInserter = std::back_insert_iterator<std::vector<Metavision::EventOpticalFlow>>;
    std::function<void(VecIt begin, VecIt end, FlowInserter inserter)> compute_flow;
    // Define the function to compute the flow
    switch (args.flow_type) {
    case DenseFlowType::PlaneFitting:
        compute_flow = [&](VecIt begin, VecIt end, FlowInserter inserter) {
            plane_fitting_flow_algo->process_events(begin, end, inserter);
        };
        break;
    case DenseFlowType::TripletMatching:
        compute_flow = [&](VecIt begin, VecIt end, FlowInserter inserter) {
            triplet_matching_flow_algo->process_events(begin, end, inserter);
        };
        break;
    case DenseFlowType::TimeGradient:
        compute_flow = [&](VecIt begin, VecIt end, FlowInserter inserter) {
            time_gradient_flow_algo->process_events(begin, end, inserter);
        };
        break;
    default:
        throw std::runtime_error("Selected DenseFlowType is not implemented!");
    }

    auto reslicer_process_events_cb = [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (begin != end) {
            if (ts_first == -1)
                ts_first = begin->t;
            ts_last = std::prev(end)->t;
        }
        // Filter the events
        if (bypass_software_stc) {
            cd_buffer.insert(cd_buffer.end(), begin, end);
        } else {
            stc_algo.process_events(begin, end, std::back_inserter(cd_buffer));
        }

        // Compute the optical flow from the events
        compute_flow(cd_buffer.cbegin(), cd_buffer.cend(), std::back_inserter(flow_buffer));

        // Process the visualization
        if (enable_visu) {
            cd_framer->process_events(cd_buffer.cbegin(), cd_buffer.cend());
            flow_framer->process_events(flow_buffer.cbegin(), flow_buffer.cend());
        }
        cd_buffer.clear();
        flow_buffer.clear();
    };

    // Vector of CD events and vector of OpticalFlow events to store the output of the two algorithms
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        reslicer.process_events(begin, end, reslicer_process_events_cb);
    });

    const auto ts_system_start = std::chrono::high_resolution_clock::now();
    camera.start();
    while (camera.is_running()) {
        Metavision::EventLoop::poll_and_dispatch(20);
        if (enable_display && (window->should_close())) {
            break;
        }
    }
    camera.stop();
    const auto ts_system_stop = std::chrono::high_resolution_clock::now();
    const auto elapsed        = std::chrono::duration_cast<std::chrono::milliseconds>(ts_system_stop - ts_system_start);

    MV_LOG_INFO() << "Ran in" << Metavision::Log::no_space << static_cast<float>(elapsed.count()) / 1000.f << "s";
    MV_LOG_INFO() << "Record duration " << Metavision::Log::no_space << static_cast<float>(ts_last - ts_first) / 1e6f
                  << "s";
    if (enable_video_writer) {
        video_writer->release();
        MV_LOG_INFO() << "Wrote video file to:" << args.out_avi_file_path;
    }

    return 0;
}
