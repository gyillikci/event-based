# Native event corner front-end (eFAST/Arc*) + nearest-neighbour tracking. SDK + OpenCV.
project(event_feature_tracker LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base driver REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

add_executable(event_feature_tracker event_feature_tracker.cpp)

target_link_libraries(event_feature_tracker
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::driver
        opencv_core
        opencv_imgproc
        opencv_highgui)

if(MSVC)
    target_compile_options(event_feature_tracker PRIVATE /O2)
endif()
