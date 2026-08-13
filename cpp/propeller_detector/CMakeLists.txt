# Native propeller / rotating-source detector.
# Uses FrequencyAlgorithm + FrequencyClusteringAlgorithm (metavision_sdk_cv) and,
# optionally, the C++-only ProximityFilterAlgorithm to focus on a region.
project(propeller_detector LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base core driver cv REQUIRED)

add_executable(propeller_detector propeller_detector.cpp)

target_link_libraries(propeller_detector
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::core
        MetavisionSDK::driver
        MetavisionSDK::cv)

# ProximityFilterAlgorithm pulls in Eigen headers. MetavisionSDK::cv usually
# propagates them; add the imported target too when it is discoverable.
find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen)
    target_link_libraries(propeller_detector PRIVATE Eigen3::Eigen)
endif()
