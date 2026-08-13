# Native complementary-filter reconstruction (events -> video). SDK + OpenCV.
project(event_to_video LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base driver REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

add_executable(event_to_video event_to_video.cpp)

target_link_libraries(event_to_video
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::driver
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_videoio)

if(MSVC)
    target_compile_options(event_to_video PRIVATE /O2)
endif()
