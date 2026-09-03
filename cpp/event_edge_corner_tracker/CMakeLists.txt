# Real-time edge (Hough on the active-event mask) + corner (eFAST/Arc*) tracking. SDK + HAL + OpenCV.
project(event_edge_corner_tracker LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base driver REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

add_executable(event_edge_corner_tracker event_edge_corner_tracker.cpp)

target_link_libraries(event_edge_corner_tracker
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::driver
        opencv_core
        opencv_imgproc
        opencv_highgui)

# ERC and bias facilities live in Metavision HAL; the driver component usually
# propagates its include path, but link it explicitly when discoverable.
find_package(MetavisionHAL QUIET)
if(TARGET MetavisionHAL::hal)
    target_link_libraries(event_edge_corner_tracker PRIVATE MetavisionHAL::hal)
endif()

if(MSVC)
    target_compile_options(event_edge_corner_tracker PRIVATE /O2)
endif()
