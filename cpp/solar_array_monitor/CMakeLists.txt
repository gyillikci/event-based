# Native real-time solar-array vibration monitor.
# Ports the validated median-mode glare guard + event vibrometer (see
# solar_array/*.py) and renders with OpenCV on a decoupled display thread so the
# live view never lags the sensor. Modal identification stays in Python (offline).
project(solar_array_monitor LANGUAGES CXX)

find_package(MetavisionSDK COMPONENTS base driver REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

add_executable(solar_array_monitor solar_array_monitor.cpp)

target_link_libraries(solar_array_monitor
    PRIVATE
        MetavisionSDK::base
        MetavisionSDK::driver
        opencv_core
        opencv_highgui
        opencv_imgproc)

# The hardware ERC facility comes from Metavision HAL; the driver component
# usually propagates its include path, but add it explicitly when discoverable.
find_package(MetavisionHAL QUIET)
if(TARGET MetavisionHAL::hal)
    target_link_libraries(solar_array_monitor PRIVATE MetavisionHAL::hal)
endif()

if(MSVC)
    target_compile_options(solar_array_monitor PRIVATE /O2)
    target_compile_definitions(solar_array_monitor PRIVATE _USE_MATH_DEFINES)
endif()
