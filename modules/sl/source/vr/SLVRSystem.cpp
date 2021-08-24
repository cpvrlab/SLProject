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

SLVRSystem::SLVRSystem(){
  VR_LOG("VR system initialized")}

SLVRSystem::~SLVRSystem()
{
    VR_LOG("VR system destroyed")
}

/*! Prepares the system for everything VR
 * The following sequence of events will be triggered:
 * 1. The system checks if the SteamVR runtime is present and if there's a HMD connected
 * 2. The OpenVR API is initialized for scene rendering (this will also launch SteamVR)
 * 3. Tracked devices (such as a HMD or a controller) are detected
 */
void SLVRSystem::startup()
{
    if (!checkStartupConditions() || !initializeOpenVR() || !detectTrackedDevices())
    {
        return;
    }
}

/*! Checks if the conditions for starting OpenVR are met
 * The conditions are that the SteamVR runtime is installed and that the system thinks that a HMD is present
 * @return Are the standard conditions met?
 */
bool SLVRSystem::checkStartupConditions()
{
    if (!vr::VR_IsRuntimeInstalled())
    {
        std::cout << "The SteamVR runtime is not installed.\n";
        std::cout << "Please download SteamVR from: https://store.steampowered.com/app/250820/SteamVR/" << std::endl;
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
        VR_ERROR("Failed to initialize the OpenVR system (error code " << error << ")");
        return false;
    }

    VR_LOG("OpenVR system initialized")

    return true;
}

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

/*! Creates an object for a HMD, adds it to the list of all devices and sets the _hmd instance variable
 * @param index The OpenVR index of the HMD
 */
void SLVRSystem::registerHmd(vr::TrackedDeviceIndex_t index)
{
    SLVRTrackedDevice* device = new SLVRTrackedDevice(index);
    _trackedDevices.push_back(device);

    _hmd = device;
    VR_LOG("Device detected: HMD")
}

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

/*! Updates the poses of the detected devices
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

        // Get the pose of the current and chuck it out if it is invalid
        vr::TrackedDevicePose_t pose = poses[trackedDevice->index()];
        if (!pose.bPoseIsValid) continue;

        // Convert the OpenVR matrix to an SL matrix and set the pose of the corresponding device
        SLMat4f matrix = openVRMatrixToSLMatrix(pose.mDeviceToAbsoluteTracking);
        trackedDevice->pose(matrix);
    }

    // Update controller states
    if (leftController())
        leftController()->updateState();

    if (rightController())
        rightController()->updateState();
}

/*! Converts an OpenVR matrix to a SLProject matrix
 * @param matrix The OpenVR matrix
 * @return The converted SLProject matrix
 */
SLMat4f SLVRSystem::openVRMatrixToSLMatrix(vr::HmdMatrix34_t matrix)
{
    SLMat4f result;

    // First column
    result.m(0, matrix.m[0][0]);
    result.m(1, matrix.m[1][0]);
    result.m(2, matrix.m[2][0]);
    result.m(3, 0);

    // Second column
    result.m(4, matrix.m[0][1]);
    result.m(5, matrix.m[1][1]);
    result.m(6, matrix.m[2][1]);
    result.m(7, 0);

    // Third column
    result.m(8, matrix.m[0][2]);
    result.m(9, matrix.m[1][2]);
    result.m(10, matrix.m[2][2]);
    result.m(11, 0);

    // Fourth column
    result.m(12, matrix.m[0][3]);
    result.m(13, matrix.m[1][3]);
    result.m(14, matrix.m[2][3]);
    result.m(15, 1);

    return result;
}

/*! Fades to the color specified in the amount of time specified
 * @param seconds The number of seconds it takes the display to fade
 * @param color The color the display fades to
 */
void SLVRSystem::fade(float seconds, const SLCol4f& color)
{
    vr::VRCompositor()->FadeToColor(seconds, color.x, color.y, color.z, color.w);
}

/*! Shuts down the OpenVR API and frees all resources
 */
void SLVRSystem::shutdown()
{
    for (SLVRTrackedDevice* trackedDevice : _trackedDevices)
    {
        delete trackedDevice;
    }

    vr::VR_Shutdown();
}