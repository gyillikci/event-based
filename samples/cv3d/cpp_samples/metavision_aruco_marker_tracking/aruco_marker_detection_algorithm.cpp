/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

#include "metavision/sdk/cv/utils/camera_geometry.h"
#include "metavision/sdk/cv/utils/camera_geometry_helpers.h"
#include "metavision/sdk/cv3d/utils/model_3d.h"
#include "metavision/sdk/cv3d/utils/model_3d_processing.h"
#include "aruco_marker_detection_algorithm.h"
#include "aruco_nano.h"

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

namespace Metavision {

ArucoMarkerDetectionAlgorithm::ArucoMarkerDetectionAlgorithm(
    const CameraGeometry32f &cam_geometry, const std::vector<std::vector<std::vector<int>>> &corners_list,
    const std::vector<std::vector<std::vector<int>>> &edges_list, float marker_size) :
    corners_list_(corners_list), edges_list_(edges_list), cam_geometry_(cam_geometry) {
    const auto size = cam_geometry_.get_image_size();
    mat_.create(size(1), size(0));
    mat_.setTo(0);
    T_c_w_.setIdentity();

    marker_size_ = marker_size;
}

void ArucoMarkerDetectionAlgorithm::reset() {
    T_c_w_.setIdentity();
    mat_.setTo(0);
    known_markers_.clear();
}

bool ArucoMarkerDetectionAlgorithm::process_internal(std::vector<ArucoMarker> &detected_markers) {
    cv::GaussianBlur(mat_, tmp_, cv::Size(5, 5), 1);

    // detect marker with tmp_
    auto markers = aruconano::MarkerDetector::detect(tmp_);

    // pose estimation
    for (auto &m : markers) {
        const auto marker_id = m.id;
        MV_LOG_DEBUG() << "Marker ID:" << m.id;

        for (int i = 0; i < 4; ++i) {
            const cv::Vec2f pt_img(m[i]);
            cv::Vec2f pt_un;
            img_to_undist_norm(cam_geometry_, pt_img, pt_un);
            m[i] = cv::Point2f(pt_un);
        }

        // pose of the marker in the camera coordinate system
        const auto r_t = m.estimatePose(cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(5, 1, CV_32F), marker_size_);

        // convert to rotation matrix
        cv::Rodrigues(r_t.first, rot_mat_tmp_);
        cv::cv2eigen(rot_mat_tmp_, rot_mat_eigen_);
        cv::cv2eigen(r_t.second, trans_mat_eigen_);
        T_c_w_.block<3, 3>(0, 0) = rot_mat_eigen_;
        T_c_w_.block<3, 1>(0, 3) = trans_mat_eigen_;

        const auto search = known_markers_.find(marker_id);
        if (search != known_markers_.end()) {
            // known marker
            search->second.T_c_am = T_c_w_;
        } else {
            // adding new marker
            ArucoMarker marker;
            create_model(marker_id, marker.model);
            marker.id     = marker_id;
            marker.T_c_am = T_c_w_;
            known_markers_.insert({marker_id, marker});
        }

        detected_markers.emplace_back(known_markers_[marker_id]);
    }

    return detected_markers.size() > 0;
}

// note: only for 6x6 markers, ID 0 to 249
void ArucoMarkerDetectionAlgorithm::create_model(int marker_id, Model3d &model) {
    // Create model
    for (const auto &k : corners_list_[marker_id]) {
        const auto pt =
            Eigen::Vector3f((k[0] - 4) * marker_size_ / 8, (k[1] - 4) * marker_size_ / 8, 0); // center model
        model.vertices_.emplace_back(pt);
    }
    Model3d::Face face;
    face.normal_ = Eigen::Vector4f(0, 0, 1, 0);
    size_t k     = 0;
    for (const auto &e : edges_list_[marker_id]) {
        model.edges_.emplace_back(Model3d::Edge{static_cast<size_t>(e[0]), static_cast<size_t>(e[1])});
        face.edges_indexes_.emplace_back(k);
        ++k;
    }
    model.faces_.emplace_back(face);
}

} // namespace Metavision
