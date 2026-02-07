/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This application applies noise filtering and computes the event rate afterwards.

#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/anti_flicker_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "event_rate_algorithm.h"
#include "event_rate_frame_generation.h"

// Function that applies a filter on events if it's not a nullptr.
// The boolean parameter is_first indicates whether the filter should be applied on the raw events
// or on the events that have already been filtered through other filters.
template<class AlgoPtr, class InputIt, class FilteredIt>
inline void apply_filter(const AlgoPtr &algo, InputIt &begin, InputIt &end, std::vector<FilteredIt> &output_buffer,
                         bool &is_first, bool apply) {
    if (algo && apply) {
        if (is_first) {
            assert(begin != nullptr && end != nullptr);
            output_buffer.clear();
            algo->process_events(begin, end, std::back_inserter(output_buffer));
            is_first = false; // the next filters will have to use output_buffer instead of begin and end
            begin    = nullptr;
            end      = nullptr;
        } else {
            auto end_it = algo->process_events(output_buffer.cbegin(), output_buffer.cend(), output_buffer.begin());
            output_buffer.resize(std::distance(output_buffer.begin(), end_it));
        }
    }
};

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string event_file_path;
    std::string cam_config_path;

    int act_filter_th        = 10000;
    int stc_filter_th        = 10000;
    double event_rate_min_th = 10.;

    bool skip_activity = false;
    bool do_activity   = true;
    bool do_stc        = false;
    bool do_af         = false;

    const std::string short_program_desc("Data Rate Demo\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("skip-activity-filter", po::bool_switch(&skip_activity), "Skip Activity Filter")
        ("activity-th", po::value<int>(&act_filter_th)->default_value(10000), "Activity Filter Threshold")
        ("apply-stc-filter", po::bool_switch(&do_stc), "Apply STC filter (off by default)")
        ("stc-th", po::value<int>(&stc_filter_th)->default_value(10000), "STC Filter Threshold")
        ("apply-anti-flicker-filter", po::bool_switch(&do_af), "Apply anti-flicker filter (off by default)")
        ("event-rate-min-th", po::value<double>(&event_rate_min_th)->default_value(10.), "Minimum Event Rate returned by the Event Rate algorithm")
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

    // Construct a camera from a recording or a live stream
    Metavision::Camera cam;
    if (!event_file_path.empty()) {
        cam = Metavision::Camera::from_file(event_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();

        if (!cam_config_path.empty()) {
            cam.load(cam_config_path);
        }
    }

    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    do_activity = !skip_activity;
    Metavision::FrequencyEstimationConfig anti_flicker_config(7, 95, 125);
    auto activity_filter = std::make_unique<Metavision::ActivityNoiseFilterAlgorithm<>>(width, height, act_filter_th);
    auto stc_filter = std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(width, height, stc_filter_th, true);
    auto af_filter  = std::make_unique<Metavision::AntiFlickerAlgorithm>(width, height, anti_flicker_config);

    Metavision::EventRateAlgorithm event_rate(event_rate_min_th);
    Metavision::FrameGenerationEventRate frame_gen(width, height, 30);

    std::vector<Metavision::EventCD> output;
    std::vector<Metavision::EventRateStruct> er_output;

    // Add callback that will pass the events to the filter (if enabled) and add event rate data. Events are then
    // passed to the EventRateFrameGeneration class
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        bool is_first = true;
        apply_filter(activity_filter, begin, end, output, is_first, do_activity);
        apply_filter(stc_filter, begin, end, output, is_first, do_stc);
        apply_filter(af_filter, begin, end, output, is_first, do_af);

        if (is_first) {
            event_rate.process_events(begin, end, std::back_inserter(er_output));

            std::copy(begin, end, std::back_inserter(output));
        } else {
            event_rate.process_events(output.cbegin(), output.cend(), std::back_inserter(er_output));
        }
        frame_gen.process_cd_events(output);
        frame_gen.process_er_events(er_output);

        output.clear();
        er_output.clear();
    });

    // Set window
    Metavision::Window window("CD events", width, height, Metavision::BaseWindow::RenderMode::BGR);
    window.set_keyboard_callback([&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            switch (key) {
            case Metavision::UIKeyEvent::KEY_N:
                // no filter
                do_activity = false;
                do_stc      = false;
                do_af       = false;
                break;
            case Metavision::UIKeyEvent::KEY_A:
                // toggle activity filter
                MV_LOG_INFO() << "Set ActivityNoiseFilterAlgorithm";
                do_activity = !do_activity;
                break;
            case Metavision::UIKeyEvent::KEY_S:
                // toggle STC filter
                MV_LOG_INFO() << "Set SpatioTemporalContrastAlgorithm";
                do_stc = !do_stc;
                break;
            case Metavision::UIKeyEvent::KEY_F:
                // toggle anti-flicker filter
                MV_LOG_INFO() << "Set AntiFlickerAlgorithm";
                do_af = !do_af;
                break;
            case Metavision::UIKeyEvent::KEY_Q:
            case Metavision::UIKeyEvent::KEY_ESCAPE:
                // quit the application
                window.set_close_flag();
                break;
            }
        }
    });

    frame_gen.set_output_callback([&](Metavision::timestamp, const cv::Mat &frame) { window.show(frame); });

    cam.start();
    while (cam.is_running() && !window.should_close()) {
        Metavision::EventLoop::poll_and_dispatch(20);
    }
    cam.stop();

    MV_LOG_INFO() << "\rFinished processing events ";

    return 0;
}
