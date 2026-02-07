/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_ACTIVE_MARKER_3D_TRACKING_SAMPLE_VIEWER_3D_H
#define METAVISION_SDK_CV3D_ACTIVE_MARKER_3D_TRACKING_SAMPLE_VIEWER_3D_H

#include <optional>
#include <Eigen/Core>
#include <OgreApplicationContext.h>
#include <OgreCameraMan.h>

namespace Metavision {

/// @brief Simple 3D viewer to display a 3D model associated with an active marker and the 3D representation of the
/// camera
class Viewer3d : private OgreBites::ApplicationContext {
public:
    /// @brief Enumerate that indicates the target of pose updates (i.e. the camera or the model)
    enum class UpdateMode { Camera, Object };
    using PoseUpdate = std::optional<Eigen::Matrix4f>;

    /// @brief Parameters for configuring the viewer
    struct Params {
        std::uint16_t width;       ///< Width of the window
        std::uint16_t height;      ///< Height of the window
        std::string model_3d_path; ///< Path to a 3D model associated with the active marker
        UpdateMode update_mode;    ///< Update mode of the viewer
    };

    /// @brief Constructor
    /// @param params The parameters for configuring the viewer
    Viewer3d(const Params &params);

    /// @brief Destructor
    ~Viewer3d();

    /// @brief Runs the viewer
    ///
    /// This call is synchronous and will block the calling thread until the viewer stops
    void run();

    /// @brief Processes a pose update
    ///
    /// Depending on the mode of the viewer, will update either the pose of the camera mesh or the pose of the mesh
    /// associated with the active marker
    /// @param T_c_am Pose of the active marker with respect to the camera
    void apply_pose_update(const PoseUpdate &T_c_am);

private:
    bool oneTimeConfig() override;
    void locateResources() override;
    void setup() override;
    bool frameRenderingQueued(const Ogre::FrameEvent &evt) override;

    void setup_base_scene();
    void setup_frustrum_mesh();
    void setup_active_marker_model();
    void setup_lights();
    void try_update_object_pose(const PoseUpdate &update);
    void try_update_camera_pose(const PoseUpdate &update);

    class KeyHandler;

    const Params params_;
    std::unique_ptr<KeyHandler> key_handler_;
    Ogre::SceneManager *scn_mgr_    = nullptr;
    Ogre::SceneNode *object_node_   = nullptr;
    Ogre::SceneNode *frustrum_node_ = nullptr;
    std::unique_ptr<OgreBites::CameraMan> camera_mgr_;
    OgreBites::InputListenerChain listener_chain_;
    std::mutex pose_update_mtx_;
    std::optional<PoseUpdate> pose_update_;
    bool should_close_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV3D_ACTIVE_MARKER_3D_TRACKING_SAMPLE_VIEWER_3D_H
