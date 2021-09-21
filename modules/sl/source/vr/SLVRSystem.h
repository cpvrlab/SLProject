//#############################################################################
//  File:      SLVRSystem.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRSYSTEM_H
#define SLPROJECT_SLVRSYSTEM_H

#include <openvr.h>

#include <vr/SLVRTrackedDevice.h>
#include <vr/SLVRHmd.h>
#include <vr/SLVRController.h>
#include <vr/SLVRCompositor.h>

#include <SLMat4.h>
#include <SLVec4.h>
#include <SLEnums.h>

#include <vector>

typedef std::vector<SLVRTrackedDevice*> SLVVRTrackedDevices;

//! The SLVRSystem class is the main interface to the OpenVR API
/*! This class is responsible for initializing and shutting down the OpenVR API, for
 * creating framebuffer objects for both eyes, for submitting frames to the device
 * and for returning objects that serve as an interface to the various VR devices.
 */
class SLVRSystem
{
    friend class SLVRTrackedDevice;

private:
    vr::IVRSystem* _system = nullptr;

    SLVVRTrackedDevices _trackedDevices;
    SLVRHmd*            _hmd             = nullptr;
    SLVRController*     _leftController  = nullptr;
    SLVRController*     _rightController = nullptr;

    SLVRCompositor* _compositor = nullptr;

    SLMat4f _globalOffset;

public:
    static SLVRSystem& instance()
    {
        static SLVRSystem instance;
        return instance;
    }

    SLVRSystem();
    ~SLVRSystem();

    void startup();
    void update();
    void resetRenderModels();
    void shutdown();

    // Getters
    vr::IVRSystem* system() { return _system; }
    bool           isRunning() { return _system != nullptr; }

    SLVVRTrackedDevices trackedDevices() { return _trackedDevices; };
    SLVRHmd*            hmd() { return _hmd; };
    SLVRController*     leftController() { return _leftController; };
    SLVRController*     rightController() { return _rightController; };

    SLMat4f& globalOffset() { return _globalOffset; }

    // Setters
    void globalOffset(const SLMat4f& globalOffset) { _globalOffset = globalOffset; }

    SLVRCompositor* compositor() { return _compositor; }

    SLMat4f getProjectionMatrix(SLEyeType eye,
                                float     nearPlane,
                                float     farPlane);
    SLMat4f getEyeMatrix(SLEyeType eye);

private:
    bool checkStartupConditions();
    bool initializeOpenVR();
    bool detectTrackedDevices();
    void registerHmd(vr::TrackedDeviceIndex_t index);
    void registerController(vr::TrackedDeviceIndex_t index);
};

#endif // SLPROJECT_SLVRSYSTEM_H
