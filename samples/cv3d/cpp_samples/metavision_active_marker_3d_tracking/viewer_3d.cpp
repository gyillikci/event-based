/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <OgreRoot.h>
#include <OgreRenderWindow.h>
#include <OgreWindowEventUtilities.h>
#include <OgreCameraMan.h>
#include <OgreEntity.h>
#include <OgreMeshManager.h>
#include <OgreTechnique.h>
#include <OgreMesh.h>
#include <OgreViewport.h>

#include "viewer_3d.h"

namespace Metavision {

namespace bfs = boost::filesystem;

class Viewer3d::KeyHandler : public OgreBites::InputListener {
public:
    using OnExitCallback = std::function<void()>;

    KeyHandler(OnExitCallback on_exit = []() {}) : on_exit_(std::move(on_exit)) {}

    void set_on_exit_cb(OnExitCallback cb) {
        on_exit_ = std::move(cb);
    }

private:
    bool keyPressed(const OgreBites::KeyboardEvent &evt) final {
        switch (evt.keysym.sym) {
        case 'q':
        case 'Q':
        case OgreBites::SDLK_ESCAPE:
            on_exit_();
            break;
        }

        return true;
    }

    OnExitCallback on_exit_;
};

Viewer3d::Viewer3d(const Params &params) :
    OgreBites::ApplicationContext("Metavision-ActiveMarker3dTracking"), params_(params) {
    should_close_ = false;
    key_handler_  = std::make_unique<KeyHandler>();
    key_handler_->set_on_exit_cb([&]() { should_close_ = true; });

    initApp();
}

Viewer3d::~Viewer3d() {
    removeInputListener(key_handler_.get());
    shutdown();
}

void Viewer3d::run() {
    getRoot()->startRendering();
}

void Viewer3d::apply_pose_update(const PoseUpdate &T_c_am) {
    std::lock_guard<std::mutex> lock(pose_update_mtx_);

    pose_update_ = T_c_am;
}

bool Viewer3d::oneTimeConfig() {
    auto root              = getRoot();
    Ogre::RenderSystem *rs = root->getRenderSystemByName("OpenGL 3+ Rendering Subsystem");
    if (!rs) {
        throw std::runtime_error("Ogre plugin 'OpenGL 3' not found on the system. \nHave you set your "
                                 "OGRE_CONFIG_DIR=<plugins.cfg> and/or your OGRE_PLUGIN_DIR envvar ?");
    }

    std::ostringstream video_mode_str;
    video_mode_str << params_.width << " x " << params_.height;
    std::map<std::string, std::string> config = {
        //{"Debug Layer", "Off"},
        {"Display Frequency", "N/A"},
        {"FSAA", "0"},
        {"Full Screen", "No"},
        {"RTT Preferred Mode", "FBO"},
        {"Reversed Z-Buffer", "No"},
        {"Separate Shader Objects", "No"},
        {"VSync", "Yes"},
        {"VSync Interval", "1"},
        {"Video Mode", video_mode_str.str()},
        {"sRGB Gamma Conversion", "No"},
    };
    for (auto &conf_it : config) {
        try {
            rs->setConfigOption(conf_it.first, conf_it.second);
        } catch (const std::exception &) {
            /// Configuration key might not exist between 2 different platforms.
            /// Ignore the one that doesn't exist
        }
    }
    root->setRenderSystem(rs);

    return true;
}

void Viewer3d::locateResources() {
    const bfs::path model_path(params_.model_3d_path);

    Ogre::ResourceGroupManager::getSingleton().addResourceLocation(model_path.parent_path().string(), "FileSystem");
    OgreBites::ApplicationContext::locateResources();
}

void Viewer3d::setup() {
    setup_base_scene();
    setup_lights();
    setup_active_marker_model();
    setup_frustrum_mesh();
}

bool Viewer3d::frameRenderingQueued(const Ogre::FrameEvent &evt) {
    std::unique_lock<std::mutex> lock(pose_update_mtx_);
    if (pose_update_) {
        const auto object_pose = *pose_update_;
        pose_update_           = std::nullopt;
        lock.unlock();

        if (params_.update_mode == UpdateMode::Object)
            try_update_object_pose(object_pose);
        else
            try_update_camera_pose(object_pose);
    }

    return !should_close_;
}

void Viewer3d::setup_base_scene() {
    OgreBites::ApplicationContext::setup();

    auto ogre_root = getRoot();
    scn_mgr_       = ogre_root->createSceneManager();

    Ogre::RTShader::ShaderGenerator::initialize();
    Ogre::RTShader::ShaderGenerator::getSingleton().addSceneManager(scn_mgr_);

    auto cam = scn_mgr_->createCamera("MainCamera");
    cam->setNearClipDistance(10);
    cam->setFarClipDistance(10000);
    cam->setAutoAspectRatio(true);

    auto *camera_node = scn_mgr_->getRootSceneNode()->createChildSceneNode();
    camera_node->attachObject(cam);

    auto window    = getRenderWindow();
    auto *viewport = window->addViewport(cam);
    viewport->setBackgroundColour(Ogre::ColourValue(0.f, 0.098f, 0.2078f));

    camera_mgr_ = std::make_unique<OgreBites::CameraMan>(camera_node);
    camera_mgr_->setStyle(OgreBites::CS_ORBIT);

    listener_chain_ = OgreBites::InputListenerChain({key_handler_.get(), camera_mgr_.get()});
    addInputListener(&listener_chain_);
}

void Viewer3d::setup_frustrum_mesh() {
    static constexpr float kDepth = 10.f;
    static constexpr float kWidth = 20.f;
    const float height            = static_cast<float>(params_.height) / static_cast<float>(params_.width) * kWidth;

    auto material = Ogre::MaterialManager::getSingleton().create("CameraFrustrumMaterial", "General");
    material->getTechnique(0)->getPass(0)->setEmissive(0.f, 1.f, 1.f);

    auto *manual_object = scn_mgr_->createManualObject("CameraFrustrum");
    manual_object->estimateVertexCount(5);
    manual_object->estimateIndexCount(16);
    manual_object->begin(material, Ogre::RenderOperation::OT_LINE_LIST);
    manual_object->position(0.f, 0.f, 0.f);
    manual_object->position(kWidth * 0.5f, height * 0.5f, -kDepth);
    manual_object->position(-kWidth * 0.5f, height * 0.5f, -kDepth);
    manual_object->position(kWidth * 0.5f, -height * 0.5f, -kDepth);
    manual_object->position(-kWidth * 0.5f, -height * 0.5f, -kDepth);

    manual_object->index(0);
    manual_object->index(1);
    manual_object->index(0);
    manual_object->index(2);
    manual_object->index(0);
    manual_object->index(3);
    manual_object->index(0);
    manual_object->index(4);
    manual_object->index(1);
    manual_object->index(2);
    manual_object->index(3);
    manual_object->index(4);
    manual_object->index(1);
    manual_object->index(3);
    manual_object->index(2);
    manual_object->index(4);

    manual_object->end();

    frustrum_node_ = scn_mgr_->getRootSceneNode()->createChildSceneNode();
    frustrum_node_->attachObject(manual_object);
}

void Viewer3d::setup_active_marker_model() {
    const bfs::path model_path(params_.model_3d_path);
    auto *object_entity = scn_mgr_->createEntity(model_path.filename().string());
    object_node_        = scn_mgr_->getRootSceneNode()->createChildSceneNode();
    object_node_->attachObject(object_entity);
}

void Viewer3d::setup_lights() {
    auto *directional_light = scn_mgr_->createLight("DirectionalLight");
    directional_light->setType(Ogre::Light::LT_DIRECTIONAL);
    directional_light->setDiffuseColour(Ogre::ColourValue(0.2f, 0.2f, 0.2f));

    auto *directional_light_node = scn_mgr_->getRootSceneNode()->createChildSceneNode();
    directional_light_node->attachObject(directional_light);
    directional_light_node->setDirection(Ogre::Vector3(0.f, -1.f, 1.f));

    auto *point_light = scn_mgr_->createLight("PointLight");
    point_light->setType(Ogre::Light::LT_POINT);

    point_light->setDiffuseColour(0.2f, 0.2f, 0.2f);
    auto *point_light_node = scn_mgr_->getRootSceneNode()->createChildSceneNode();
    point_light_node->attachObject(point_light);
    point_light_node->setPosition(-10.f, 0.f, 100.f);
}

void Viewer3d::try_update_object_pose(const PoseUpdate &update) {
    if (!update) {
        object_node_->setVisible(false);
        return;
    }

    object_node_->setVisible(true);
    Eigen::Matrix4f T_co_ccv;
    T_co_ccv << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;

    const Eigen::Matrix4f T_c_o = T_co_ccv * *update;
    const Eigen::Quaternionf q(T_c_o.block<3, 3>(0, 0));
    const Eigen::Vector3f pos(T_c_o.block<3, 1>(0, 3));
    object_node_->setPosition(pos.x(), pos.y(), pos.z());
    object_node_->setOrientation(q.w(), q.x(), q.y(), q.z());
}

void Viewer3d::try_update_camera_pose(const PoseUpdate &update) {
    if (!update)
        return;

    Eigen::Matrix4f T_co_ccv;
    T_co_ccv << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;

    const Eigen::Matrix4f T_o_c = (T_co_ccv * *update).inverse();
    const Eigen::Quaternionf q(T_o_c.block<3, 3>(0, 0));
    const Eigen::Vector3f pos(T_o_c.block<3, 1>(0, 3));
    frustrum_node_->setPosition(pos.x(), pos.y(), pos.z());
    frustrum_node_->setOrientation(q.w(), q.x(), q.y(), q.z());
}
} // namespace Metavision