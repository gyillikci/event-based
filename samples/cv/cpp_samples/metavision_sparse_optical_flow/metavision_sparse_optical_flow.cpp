/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to display the results of the sparse optical flow algorithm.
#include <iostream>
#include <functional>
#include <chrono>
#include <boost/program_options.hpp>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/sparse_optical_flow_algorithm.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include "sparse_flow_frame_generation.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string cam_config_path;
    std::string event_file_path;
    std::string out_avi_file_path;
    uint32_t sw_stc_threshold;
    bool benchmark;
    bool no_display;
    bool realtime_playback_speed;

    const std::string program_desc(
        "Code sample showing how to use Metavision SDK to display results of sparse optical flow.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-camera-config,j", po::value<std::string>(&cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("input-event-file,i", po::value<std::string>(&event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("output-avi-file,o", po::value<std::string>(&out_avi_file_path)->default_value(""), "Path to output AVI file.")
        ("sw-stc-threshold,t", po::value<uint32_t>(&sw_stc_threshold)->default_value(10000), "Software STC threshold in us (10000 by default, disabled if sensor's STC is enabled or if the threshold is set to 0)")
        ("benchmark", po::bool_switch(&benchmark)->default_value(false), "Configure pipeline to enable timing of the sparse flow algorithm specifically")
        ("no-display,d", po::bool_switch(&no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
        ;
    // clang-format on

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

    const bool enable_display      = !no_display && !benchmark;
    const bool enable_video_writer = !out_avi_file_path.empty() && !benchmark;
    const bool enable_visu         = enable_display || enable_video_writer;

    // Initialize the camera
    Metavision::Camera camera;
    if (event_file_path.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
        if (!cam_config_path.empty()) {
            try {
                camera.load(cam_config_path);
            } catch (const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
        }

    } else {
        camera = Metavision::Camera::from_file(
            event_file_path, Metavision::FileConfigHints().real_time_playback(realtime_playback_speed && !benchmark));
    }

    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    /// Data flow:
    //
    //  0 (Cam) -->-- 1 (STC) ----------->-----------  2 (Flow)
    //                |                                |
    //                v                                v
    //                |                                |
    //                |------>------->-<--------<------|
    //                                |
    //                                v
    //                                |
    //                                3 (Flow Frame Generator)
    //                                |
    //                                v
    //                                |
    //                |------<-------<->-------->------|
    //                |                                |
    //                v                                v
    //                |                                |
    //                4 (Video writer)                 5 (Display)
    //

    // We prepare a software STC algo to use if sensor's hardware STC is not enabled and software threshold not 0
    Metavision::SpatioTemporalContrastAlgorithm stc_algo(width, height, sw_stc_threshold);
    auto *event_trail_filter        = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();
    const bool hardware_stc_enabled = event_trail_filter && event_trail_filter->is_enabled();
    const bool bypass_software_stc  = hardware_stc_enabled || sw_stc_threshold == 0;

    Metavision::SparseOpticalFlowAlgorithm flow_algo(width, height,
                                                     Metavision::SparseOpticalFlowConfig::Preset::FastObjects);

    std::unique_ptr<Metavision::SparseFlowFrameGeneration> flow_frame_gen;
    if (enable_visu) {
        flow_frame_gen = std::make_unique<Metavision::SparseFlowFrameGeneration>(width, height, 30);
    }

    // Vector of CD events and vector of OpticalFlow events to store the output of the two algorithms
    Metavision::timestamp ts_first = -1, ts_last = -1;
    std::vector<Metavision::EventCD> stc_output;
    std::vector<Metavision::EventOpticalFlow> flow_output;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (begin != end) {
            if (ts_first == -1)
                ts_first = begin->t;
            ts_last = std::prev(end)->t;
        }
        if (bypass_software_stc) {
            stc_output.insert(stc_output.end(), begin, end);
        } else {
            stc_algo.process_events(begin, end, std::back_inserter(stc_output));
        }
        flow_algo.process_events(stc_output.begin(), stc_output.end(), std::back_inserter(flow_output));

        // Call the frame generator on the processed events
        if (flow_frame_gen) {
            flow_frame_gen->process_cd_events(stc_output);
            flow_frame_gen->process_flow_events(flow_output);
        }
        stc_output.clear();
        flow_output.clear();
    });

    //  Create a video from the generated frames
    std::unique_ptr<cv::VideoWriter> video_writer;
    if (enable_video_writer) {
        video_writer = std::make_unique<cv::VideoWriter>(out_avi_file_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                         30, cv::Size(width, height));
    }

    // Initialize the window if not disabled
    std::unique_ptr<Metavision::Window> window;
    if (enable_display) {
        window = std::make_unique<Metavision::Window>("Sparse Optical Flow", width, height,
                                                      Metavision::BaseWindow::RenderMode::BGR);
        window->set_keyboard_callback(
            [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window->set_close_flag();
                }
            });
    }

    if (flow_frame_gen) {
        flow_frame_gen->set_output_callback([&](Metavision::timestamp, const cv::Mat &frame) {
            if (window) {
                window->show(frame);
            }
            if (video_writer) {
                video_writer->write(frame);
            }
        });
    }

    const auto start = std::chrono::high_resolution_clock::now();
    camera.start();
    while (camera.is_running()) {
        Metavision::EventLoop::poll_and_dispatch(20);
        if (window && window->should_close()) {
            break;
        }
    }
    camera.stop();
    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << Metavision::Log::no_space << static_cast<float>(elapsed.count()) / 1000.f << "s";
    MV_LOG_INFO() << "Record duration " << Metavision::Log::no_space << static_cast<float>(ts_last - ts_first) / 1e6f
                  << "s";
    if (video_writer) {
        MV_LOG_INFO() << "Wrote video file:" << out_avi_file_path;
    }

    return 0;
}
