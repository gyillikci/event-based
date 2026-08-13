# Top-level CMake for the native (C++) event-based detectors.
#
# These two tools deliberately use Metavision SDK algorithms that have NO Python
# wrapper and therefore cannot be called from the existing Python scripts:
#   * ModulatedLightDetectorAlgorithm  (metavision_sdk_cv)   -> active markers
#   * ActiveMarkerTrackerAlgorithm     (metavision_sdk_cv)   -> active markers
#   * ProximityFilterAlgorithm         (metavision_sdk_cv)   -> propeller focus
#
# Configure (point CMAKE_PREFIX_PATH at the installed SDK):
#   cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH="C:/Program Files/Prophesee"
#   cmake --build cpp/build --config Release
cmake_minimum_required(VERSION 3.16)

project(event_based_cpp_detectors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_subdirectory(active_marker_detector)
add_subdirectory(propeller_detector)
add_subdirectory(solar_array_monitor)
add_subdirectory(video_to_event)
add_subdirectory(event_to_video)
add_subdirectory(event_feature_tracker)
