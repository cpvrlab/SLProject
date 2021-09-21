//#############################################################################
//  File:      SLVRSystem.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVRSystem.h>
#include <vr/SLVR.h>
#include <vr/SLVRConvert.h>

//-----------------------------------------------------------------------------
SLVRSystem::SLVRSystem()
{
    VR_LOG("VR system initialized"); // This semicolon prevents strange auto-formatting
}
//-----------------------------------------------------------------------------
SLVRSystem::~SLVRSystem()
{
    VR_LOG("VR system destroyed")
}
//-----------------------------------------------------------------------------
/*! Prepares the system for everything VR
 * The following sequence of events will be triggered:
 * 1. The system checks if the SteamVR runtime is present and if there's a HMD connected
 * 2. The OpenVR API is initialized for scene rendering (this will also launch SteamVR)
 * 3. Tracked devices (such as a HMD or a controller) are detected
 * 4. The compositor is initialized
 */
void SLVRSystem::startup()
{
    if (!checkStartupConditions() || !initializeOpenVR() || !detectTrackedDevices())
    {
        return;
    }

    _compositor = new SLVRCompositor();
    _compositor->startup();
}
//-----------------------------------------------------------------------------
/*! Checks if the conditions for starting OpenVR are met
 * The conditions are that the SteamVR runtime is installed and
 * that the system thinks that a HMD is present
 * @return Are the standard conditions met?
 */
bool SLVRSystem::checkStartupConditions()
{
    if (!vr::VR_IsRuntimeInstalled())
    {
        VR_ERROR("The SteamVR runtime is not installed.\nPlease download SteamVR from: https://store.steampowered.com/app/250820/SteamVR/")
        return false;
    }

    VR_LOG("SteamVR runtime present")

    if (!vr::VR_IsHmdPresent())
    {
        std::cout << "No HMD was detected." << std::endl;
        return false;
    }

    VR_LOG("HMD probably present")

    return true;
}
//-----------------------------------------------------------------------------
/*! Initializes the OpenVR API for scene rendering
 * @return Was the initialization successful?
 */
bool SLVRSystem::initializeOpenVR()
{
    VR_LOG("Initializing OpenVR...")

    vr::HmdError error;
    _system = vr::VR_Init(&error, vr::EVRApplicationType::VRApplication_Scene);

    if (!_system)
    {
        const char* description = vr::VR_GetVRInitErrorAsEnglishDescription(error);
        VR_ERROR("Failed to initialize the OpenVR system: " << description)
        return false;
    }

    VR_LOG("OpenVR system initialized")

    return true;
}
//-----------------------------------------------------------------------------
/*! Detects all tracked devices, creates objects for interfacing with them
 * and stores the objects in a list of all devices and in instance variables
 * for accessing specific devices
 * @return Was a HMD detected?
 */
bool SLVRSystem::detectTrackedDevices()
{
    for (vr::TrackedDeviceIndex_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++)
    {
        // Get the class of the device (HMD, controller, etc.)
        vr::TrackedDeviceClass deviceClass = _system->GetTrackedDeviceClass(i);

        // Register the device depending on its class
        switch (deviceClass)
        {
            case vr::TrackedDeviceClass_HMD:
                registerHmd(i);
                break;
            case vr::TrackedDeviceClass_Controller:
                registerController(i);
                break;
            default:
                break;
        }
    }

    // Report an error if there was no HMD found
    if (!_hmd)
    {
        VR_ERROR("No HMD could be detected")
        return false;
    }

    return true;
}
//-----------------------------------------------------------------------------
/*! Creates an object for a HMD, adds it to the list of all devices and
 * sets the _hmd instance variable
 * @param index The OpenVR index of the HMD
 */
void SLVRSystem::registerHmd(vr::TrackedDeviceIndex_t index)
{
    SLVRHmd* device = new SLVRHmd(index);
    _trackedDevices.push_back(device);

    _hmd = device;
    VR_LOG("Device detected: HMD")
}
//-----------------------------------------------------------------------------
/*! Creates an object for a controller, add it to the list of all devices
 * and set the _leftController or _rightController instance variable depending on
 * which type of controller this is
 * @param index The OpenVR index of the controller
 */
void SLVRSystem::registerController(vr::TrackedDeviceIndex_t index)
{
    SLVRController* device = new SLVRController(index);
    _trackedDevices.push_back(device);

    vr::ETrackedControllerRole role = _system->GetControllerRoleForTrackedDeviceIndex(index);
    if (role == vr::ETrackedControllerRole::TrackedControllerRole_LeftHand)
    {
        _leftController = device;
        VR_LOG("Device detected: Left Controller")
    }
    else if (role == vr::ETrackedControllerRole::TrackedControllerRole_RightHand)
    {
        _rightController = device;
        VR_LOG("Device detected: Right Controller")
    }
}
//-----------------------------------------------------------------------------
/*! Updates the poses of the detected devices and gets the states
 * of buttons, triggers and axes
 * The poses of render models will be updated as well
 */
void SLVRSystem::update()
{
    // Fill all poses into an array
    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

    for (SLVRTrackedDevice* trackedDevice : _trackedDevices)
    {
        // Continue if there's no device at this index
        if (!trackedDevice) continue;

        // Get the pose of the current device and chuck it out if it is invalid
        vr::TrackedDevicePose_t pose = poses[trackedDevice->index()];
        if (!pose.bPoseIsValid) continue;

        // Convert the OpenVR matrix to a SL matrix and set the pose of the corresponding device
        SLMat4f matrix = SLVRConvert::openVRMatrixToSLMatrix(pose.mDeviceToAbsoluteTracking);
        trackedDevice->localPose(matrix);

        // Update the render model pose if it's loaded
        if (trackedDevice->renderModel())
            trackedDevice->renderModel()->node()->om(trackedDevice->pose());
    }

    // Update device states
    if (hmd())
        hmd()->updateState();

    if (leftController())
        leftController()->updateState();

    if (rightController())
        rightController()->updateState();
}
//-----------------------------------------------------------------------------
/*! Deletes all render models without the nodes
 * This method is called when uninitializing a scene so we don't have
 * a dangling pointer to the node in the render model
 */
void SLVRSystem::resetRenderModels()
{
    for (SLVRTrackedDevice* trackedDevice : _trackedDevices)
    {
        if (trackedDevice->renderModel())
            trackedDevice->deleteRenderModelWithoutNode();
    }
}
//-----------------------------------------------------------------------------
/*! Gets the projection matrix for an eye
 * @param eye The eye this projection matrix corresponds to
 * @param nearPlane The near clipping plane of the frustum
 * @param farPlane The far clipping plane of the frustum
 * @return The projection matrix
 */
SLMat4f SLVRSystem::getProjectionMatrix(SLEyeType eye,
                                        float     nearPlane,
                                        float     farPlane)
{
    vr::Hmd_Eye       openVREye    = SLVRConvert::SLEyeTypeToOpenVREye(eye);
    vr::HmdMatrix44_t openVRMatrix = _system->GetProjectionMatrix(openVREye,
                                                                  nearPlane,
                                                                  farPlane);
    return SLVRConvert::openVRMatrixToSLMatrix(openVRMatrix);
}
//-----------------------------------------------------------------------------
/*! Gets the per-eye offset of the camera relative to the HMD
 * @param eye The eye this eye matrix corresponds to
 * @return The eye matrix
 */
SLMat4f SLVRSystem::getEyeMatrix(SLEyeType eye)
{
    vr::Hmd_Eye       openVREye    = SLVRConvert::SLEyeTypeToOpenVREye(eye);
    vr::HmdMatrix34_t openVRMatrix = _system->GetEyeToHeadTransform(openVREye);

    // The matrix is inverted at the end to convert from
    // the eye to head to the head to eye matrix
    return SLVRConvert::openVRMatrixToSLMatrix(openVRMatrix).inverted();
}
//-----------------------------------------------------------------------------
/*! Shuts down the OpenVR API and frees all resources
 */
void SLVRSystem::shutdown()
{
    VR_LOG("Shutting OpenVR down...")

    for (SLVRTrackedDevice* trackedDevice : _trackedDevices)
        delete trackedDevice;
    _trackedDevices.clear();

    delete _compositor;

    _hmd             = nullptr;
    _leftController  = nullptr;
    _rightController = nullptr;
    _compositor      = nullptr;

    vr::VR_Shutdown();
    _system = nullptr;
}
//-----------------------------------------------------------------------------