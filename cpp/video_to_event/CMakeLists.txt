# Native ESIM event-camera simulator (video -> events). OpenCV only; no SDK/ML.
project(video_to_event LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

add_executable(video_to_event video_to_event.cpp)

target_link_libraries(video_to_event
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_videoio)

if(MSVC)
    target_compile_options(video_to_event PRIVATE /O2)
endif()
