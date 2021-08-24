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

SLVRHmd::SLVRHmd(SLVRTrackedDeviceIndex index) : SLVRTrackedDevice(index)
{
}

/*! Updates the state of the controller
 * The state carries information about the buttons and axes and whatnot,
 * but we only need the state of the proximity sensor from it.
 */
void SLVRHmd::updateState()
{
    // Get the state of this HMD and store it in the _state instance variable
    system()->GetControllerState(_index, &_state, sizeof(_state));
}

/*! Returns whether or not the proxmity sensor is activated
 * @return True when the proximity sensor is activated, false otherwise
 */
SLbool SLVRHmd::isProximitySensorActivated()
{
    uint64_t mask = vr::ButtonMaskFromId(vr::EVRButtonId::k_EButton_ProximitySensor);
    return (_state.ulButtonPressed & mask) != 0;
}