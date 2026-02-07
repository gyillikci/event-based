# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_active_marker_2d_tracking)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS core driver cv ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgcodecs imgproc videoio REQUIRED)

set (sample metavision_active_marker_2d_tracking)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample}
                        MetavisionSDK::base
                        MetavisionSDK::core
                        MetavisionSDK::cv
                        MetavisionSDK::driver
                        MetavisionSDK::ui
                        Boost::program_options
                        opencv_core
                        opencv_highgui
                        opencv_imgcodecs
                        opencv_imgproc
                        opencv_videoio)
