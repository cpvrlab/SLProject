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

#include <SLMat4.h>
#include <SLVec4.h>

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
    vr::IVRSystem*      _system = nullptr;
    SLVVRTrackedDevices _trackedDevices;

    SLVRHmd*        _hmd             = nullptr;
    SLVRController* _leftController  = nullptr;
    SLVRController* _rightController = nullptr;

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

    // Getters
    vr::IVRSystem*      system() { return _system; }
    bool                isRunning() { return _system != nullptr; }
    SLVVRTrackedDevices trackedDevices() { return _trackedDevices; };
    SLVRHmd*            hmd() { return _hmd; };
    SLVRController*     leftController() { return _leftController; };
    SLVRController*     rightController() { return _rightController; };

    void fade(float seconds, const SLCol4f& color);
    void shutdown();

private:
    bool checkStartupConditions();
    bool initializeOpenVR();
    bool detectTrackedDevices();
    void registerHmd(vr::TrackedDeviceIndex_t index);
    void registerController(vr::TrackedDeviceIndex_t index);

    SLMat4f openVRMatrixToSLMatrix(vr::HmdMatrix34_t matrix);
};

#endif // SLPROJECT_SLVRSYSTEM_H
