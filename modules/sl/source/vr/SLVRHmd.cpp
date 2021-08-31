//#############################################################################
//  File:      SLVRHmd.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVR.h>
#include <vr/SLVRHmd.h>
#include <vr/SLVRSystem.h>
#include <GlobalTimer.h>

SLVRHmd::SLVRHmd(SLVRTrackedDeviceIndex index) : SLVRTrackedDevice(index)
{
    _lastMovementTime = GlobalTimer::timeS();
}

/*! Updates the state of the controller
 * The state carries information about the buttons and axes and whatnot,
 * but we only need the state of the proximity sensor from it.
 */
void SLVRHmd::updateState()
{
    // Get the state of this HMD and store it in the _state instance variable
    system()->GetControllerState(_index, &_state, sizeof(_state));

    SLfloat now = GlobalTimer::timeS();
    if (SLVRSystem::instance().leftController() && SLVRSystem::instance().rightController() && now - _lastMovementTime > 0.5f)
    {
        _lastMovementTime = now;

        SLVec2f axisLeft = SLVRSystem::instance().leftController()->get2DAxis(SLVRControllerAxis::VRCA_axis_0);
        SLVec2f axisRight = SLVRSystem::instance().rightController()->get2DAxis(SLVRControllerAxis::VRCA_axis_0);

        SLVec3f axisZ = SLVRSystem::instance().leftController()->localPose().axisZ();
        SLVec3f forward2D = SLVec3f(-axisZ.x, 0.0f, -axisZ.z).normalized();
        SLVec3f right2D = SLVec3f(axisZ.z, 0.0f, -axisZ.x).normalized();

        SLVec3f movement = (axisLeft.y * forward2D + axisLeft.x * right2D);
        movement.y = axisRight.y;
        SLVRSystem::instance().globalOffset().translate(movement);
    }
}

/*! Returns whether or not the proximity sensor is activated
 * @return True when the proximity sensor is activated, false otherwise
 */
SLbool SLVRHmd::isProximitySensorActivated()
{
    uint64_t mask = vr::ButtonMaskFromId(vr::EVRButtonId::k_EButton_ProximitySensor);
    return (_state.ulButtonPressed & mask) != 0;
}