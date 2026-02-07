# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

project(metavision_active_marker_3d_tracking)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

set (sample metavision_active_marker_3d_tracking)

find_package(MetavisionSDK COMPONENTS base cv cv3d driver REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OGRE COMPONENTS Bites CONFIG REQUIRED)

add_executable(${sample}
    ${sample}.cpp
    viewer_3d.cpp
)

target_link_libraries(${sample} OgreBites MetavisionSDK::base MetavisionSDK::cv MetavisionSDK::cv3d MetavisionSDK::driver Boost::program_options)

if (WIN32)
    # Configure Ogre config files for a developer that is building the app from source
    get_filename_component(USER_DOCUMENTS "[HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders;personal]" ABSOLUTE)
    set(OGRE_INSTALL_DIR ${OGRE_PREFIX_DIR})
    if (CMAKE_BUILD_TYPE MATCHES ^[Dd][Ee][Bb][Uu][Gg]$)
        set(OGRE_INSTALL_DIR ${OGRE_INSTALL_DIR}/debug)
    endif()
    if (OGRE_PLUGIN_DIR)
      set(OGRE_INSTALL_PLUGIN_DIR ${OGRE_PLUGIN_DIR})
    else ()
      set(OGRE_INSTALL_PLUGIN_DIR ${OGRE_INSTALL_DIR}/bin)
    endif ()
    configure_file("resources.cfg.in" "${USER_DOCUMENTS}/Metavision-ActiveMarker3dTracking/resources.cfg" @ONLY)
    configure_file("plugins.cfg.in" "${USER_DOCUMENTS}/Metavision-ActiveMarker3dTracking/plugins.cfg" @ONLY)
endif(WIN32)
