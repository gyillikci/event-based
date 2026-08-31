# Native active-marker detector.
# Uses the C++-only ModulatedLightDetectorAlgorithm + ActiveMarkerTrackerAlgorithm
# (no Python bindings exist for these), so it must be built as native C++.
project(active_marker_detector LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base core driver cv REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)

add_executable(active_marker_detector active_marker_detector.cpp)

target_link_libraries(active_marker_detector
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::core
        MetavisionSDK::driver
        MetavisionSDK::cv
        opencv_core
        opencv_highgui
        opencv_imgproc)
