# C++ port of the v3 production propeller detector (detect_propeller.py).
# Uses FrequencyMapAsyncAlgorithm (metavision_sdk_analytics) plus a custom
# connected-component analyzer and Bayesian tracker, with MTWindow display.
project(drone_propeller_detector LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base core driver cv analytics ui REQUIRED)
find_package(OpenCV COMPONENTS core imgproc REQUIRED)

add_executable(drone_propeller_detector drone_propeller_detector.cpp)

target_link_libraries(drone_propeller_detector
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::core
        MetavisionSDK::driver
        MetavisionSDK::cv
        MetavisionSDK::analytics
        MetavisionSDK::ui
        opencv_core
        opencv_imgproc)

# ProximityFilterAlgorithm (--focus-*) pulls in Eigen headers.
find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen)
    target_link_libraries(drone_propeller_detector PRIVATE Eigen3::Eigen)
endif()

if(MSVC)
    target_compile_options(drone_propeller_detector PRIVATE /O2)
endif()
