//#############################################################################
//  File:      SLVRController.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRCONTROLLER_H
#define SLPROJECT_SLVRCONTROLLER_H

#include <openvr.h>

#include <SL.h>
#include <SLVec2.h>
#include <vr/SLVRTrackedDevice.h>

//-----------------------------------------------------------------------------
enum SLVRControllerButton
{
    VRCB_system          = vr::EVRButtonId::k_EButton_System,
    VRCB_applicationMenu = vr::EVRButtonId::k_EButton_ApplicationMenu,
    VRCB_DPadLeft        = vr::EVRButtonId::k_EButton_DPad_Left,
    VRCB_DPadUp          = vr::EVRButtonId::k_EButton_DPad_Up,
    VRCB_DPadRight       = vr::EVRButtonId::k_EButton_DPad_Right,
    VRCB_DPadDown        = vr::EVRButtonId::k_EButton_DPad_Down,
    VRCB_a               = vr::EVRButtonId::k_EButton_A,
    VRCB_proximitySensor = vr::EVRButtonId::k_EButton_ProximitySensor,
    VRCB_axis0           = vr::EVRButtonId::k_EButton_Axis0,
    VRCB_axis1           = vr::EVRButtonId::k_EButton_Axis1,
    VRCB_axis2           = vr::EVRButtonId::k_EButton_Axis2,
    VRCB_axis3           = vr::EVRButtonId::k_EButton_Axis3,
    VRCB_axis4           = vr::EVRButtonId::k_EButton_Axis4,

    // Aliases
    VRCB_steamVRTouchpad         = vr::EVRButtonId::k_EButton_SteamVR_Touchpad,
    VRCB_steamVRTrigger          = vr::EVRButtonId::k_EButton_SteamVR_Trigger,
    VRCB_dashboardBack           = vr::EVRButtonId::k_EButton_Dashboard_Back,
    VRCB_indexControllerA        = vr::EVRButtonId::k_EButton_IndexController_A,
    VRCB_indexControllerB        = vr::EVRButtonId::k_EButton_IndexController_B,
    VRCB_indexControllerJoystick = vr::EVRButtonId::k_EButton_IndexController_JoyStick
};
//-----------------------------------------------------------------------------
enum SLVRControllerAxis
{
    VRCA_axis_0 = 0,
    VRCA_axis_1 = 1,
    VRCA_axis_2 = 2,
    VRCA_axis_3 = 3,
    VRCA_axis_4 = 4
};
//-----------------------------------------------------------------------------
//! SLVRController provides access to controller input
/*! SLVRController is a subclass of SLVRTrackedDevice with functions for interfacing with controllers.
 * The functions return information about the buttons, the triggers and the axes.
 */
class SLVRController : public SLVRTrackedDevice
{
    friend class SLVRSystem;

public:
    SLbool  isButtonPressed(const SLVRControllerButton& button) const;
    SLbool  isButtonTouched(const SLVRControllerButton& button) const;
    SLfloat getTriggerAxis(const SLVRControllerAxis& axis) const;
    SLVec2f get2DAxis(const SLVRControllerAxis& axis) const;

private:
    SLVRController(SLVRTrackedDeviceIndex index);
    void     updateState() override;
    uint64_t getButtonMask(const SLVRControllerButton& button) const;

    vr::VRControllerState_t _state;
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_SLVRCONTROLLER_H
