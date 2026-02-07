/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ARUCO_MARKER_DETECTION_ALGORITHM_H
#define METAVISION_SDK_CV_ARUCO_MARKER_DETECTION_ALGORITHM_H

#include <functional>
#include <set>
#include <Eigen/Core>

#include "metavision/sdk/cv/utils/gauss_newton_solver.h"
#include "metavision/sdk/cv/utils/plane_fitting_flow_estimator.h"
#include "metavision/sdk/cv3d/utils/edge_data_association.h"
#include "metavision/sdk/cv3d/utils/edge_ls_problem.h"

namespace Metavision {

template<typename T>
class MostRecentTimestampBufferT;
using MostRecentTimestampBuffer = MostRecentTimestampBufferT<timestamp>;

template<typename T>
class CameraGeometryBase;
using CameraGeometry32f = CameraGeometryBase<float>;

struct Model3d;

/// @brief Structure to represent an ArUco Marker
struct ArucoMarker {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id;                 ///< ID in the ARUCO_MIP_36h12 library
    Model3d model;          ///< Corresponding Model3d structure
    Eigen::Matrix4f T_c_am; ///< Transformation from model to camera coordinate system
};

/// @brief Class to detect ArUco markers. The detection uses the ArUco Nano library and only works on the
/// ARUCO_MIP_36h12 dictionary, limiting markers to those of size 6x6 and with IDs from 0 to 249.
class ArucoMarkerDetectionAlgorithm {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Constructor
    /// @param cam_geometry Camera geometry instance allowing mapping coordinates from camera to image (and vice versa)
    /// @param corners_list List of all the corners of the Aruco marker to detect
    /// @param edges_list List of all the corners of the Aruco marker to detect
    /// @param marker_size Size of the marker, used to scale the camera pose
    ArucoMarkerDetectionAlgorithm(const CameraGeometry32f &cam_geometry,
                                  const std::vector<std::vector<std::vector<int>>> &corners_list,
                                  const std::vector<std::vector<std::vector<int>>> &edges_list, float marker_size);

    /// @brief Tries to detect the 3D model from the input events buffer
    /// @tparam InputIt Read-Only input event iterator type.
    /// @param[in] it_begin Iterator to the first input event
    /// @param[in] it_end Iterator to the past-the-end event
    /// @param[out] detected_markers If detection succeeded, contains the detected @ref ArucoMarker instances
    /// @return True if at least one marker was detected
    /// @warning The detection will only work with markers of the ARUCO_MIP_36h12 dictionary
    template<typename InputIt>
    bool process_events(InputIt it_begin, InputIt it_end, std::vector<ArucoMarker> &detected_markers);

    /// @brief Resets detection internal buffers
    void reset();

private:
    /// @brief Returns true if a marker is detected
    bool process_internal(std::vector<ArucoMarker> &detected_markers);

    /// @brief Creates a @ref Model3d instance corresponding to the detected marker
    /// @param marker_id The ID of the detected marker
    /// @param model The Model3D instance to be filled. Vertices are scaled to the real size of the marker
    void create_model(int marker_id, Model3d &model);

    const CameraGeometry32f &cam_geometry_; ///< Camera Geometry instance
    const std::vector<std::vector<std::vector<int>>> corners_list_,
        edges_list_; ///< Corners and edges of Aruco markers to detect

    float marker_size_;     ///< Marker size
    Eigen::Matrix4f T_c_w_; ///< Detected pose, from model to camera coordinate system

    cv::Mat_<uint8_t> mat_, tmp_;     ///< Matrices used for event integration and detection
    cv::Mat rot_mat_tmp_;             ///< Rotation matrix of detected pose
    Eigen::Matrix3f rot_mat_eigen_;   ///< Rotation matrix in Eigen format
    Eigen::Vector3f trans_mat_eigen_; ///< Translation vector in Eigen format

    std::map<int, ArucoMarker> known_markers_; ///< Known markers
};

template<typename InputIt>
bool ArucoMarkerDetectionAlgorithm::process_events(InputIt it_begin, InputIt it_end,
                                                   std::vector<ArucoMarker> &detected_markers) {
    detected_markers.clear();
    if (it_begin == it_end) {
        return false;
    }

    for (auto it = it_begin; it != it_end; ++it) {
        mat_.at<uint8_t>(it->y, it->x) = it->p * 255;
    }

    if (process_internal(detected_markers)) {
        return true;
    }
    return false;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_ARUCO_MARKER_DETECTION_ALGORITHM_H
