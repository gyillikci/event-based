# Native frequency-based active-marker detector.
# C++ port of detect_active_markers.py: FrequencyAlgorithm + FrequencyClustering
# identify blinking LEDs by fixed blink frequency, matched to active_markers.json.
project(active_marker_freq_detector LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base core driver cv REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)

add_executable(active_marker_freq_detector active_marker_freq_detector.cpp)

target_link_libraries(active_marker_freq_detector
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::core
        MetavisionSDK::driver
        MetavisionSDK::cv
        opencv_core
        opencv_highgui
        opencv_imgproc)

# Boost.PropertyTree (header-only) parses active_markers.json. The Metavision SDK
# targets propagate the bundled Boost include dirs; add the discovered ones too
# when a standalone Boost is available.
find_package(Boost QUIET)
if(Boost_FOUND)
    target_include_directories(active_marker_freq_detector PRIVATE ${Boost_INCLUDE_DIRS})
endif()
