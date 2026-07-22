cmake_minimum_required(VERSION 3.16)
project(ndrone_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# --- Options ---------------------------------------------------------------
# WITH_METAVISION: link the Prophesee Metavision C++ SDK for live/recorded
#   event input. OFF here (SDK not present); the synthetic sources build in.
# WITH_PORTAUDIO: link PortAudio for live ReSpeaker capture.
option(WITH_METAVISION "Build against the Metavision C++ SDK" OFF)
option(WITH_PORTAUDIO  "Build acoustic live capture with PortAudio" ON)

# --- Dependencies ----------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# --- Core library (SDK-independent, fully portable) ------------------------
add_library(ndrone_core STATIC
  src/fft.cpp
  src/propeller_detector.cpp
  src/bayesian_fusion.cpp
  src/acoustic_processor.cpp
  src/azimuth.cpp
  src/synthetic.cpp
)
target_include_directories(ndrone_core PUBLIC include)
target_link_libraries(ndrone_core PUBLIC ${OpenCV_LIBS})
target_include_directories(ndrone_core PUBLIC ${OpenCV_INCLUDE_DIRS})

if(WITH_PORTAUDIO)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO portaudio-2.0)
  endif()
  if(PORTAUDIO_FOUND)
    target_compile_definitions(ndrone_core PUBLIC WITH_PORTAUDIO)
    target_include_directories(ndrone_core PUBLIC ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(ndrone_core PUBLIC ${PORTAUDIO_LIBRARIES})
    message(STATUS "PortAudio found: live acoustic capture enabled.")
  else()
    message(STATUS "PortAudio not found: acoustic analysis is offline-only.")
  endif()
endif()

if(WITH_METAVISION)
  # Adjust component names to your installed Metavision SDK version.
  find_package(MetavisionSDK REQUIRED COMPONENTS base stream analytics core)
  target_compile_definitions(ndrone_core PUBLIC WITH_METAVISION)
  target_link_libraries(ndrone_core PUBLIC
    MetavisionSDK::base MetavisionSDK::stream MetavisionSDK::analytics MetavisionSDK::core)
endif()

# --- Applications ----------------------------------------------------------
foreach(app detect_propeller detect_hand_shake detect_drone_fused calibrate_fan_frequency)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE ndrone_core)
endforeach()

# --- Tests -----------------------------------------------------------------
enable_testing()
add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE ndrone_core)
add_test(NAME core_tests COMMAND test_core)
