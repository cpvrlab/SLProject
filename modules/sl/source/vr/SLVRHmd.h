//#############################################################################
//  File:      SLVRHmd.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRHMD_H
#define SLPROJECT_SLVRHMD_H

#include <openvr.h>

#include <SL.h>
#include <SLVec2.h>
#include <vr/SLVRTrackedDevice.h>

//! SLVRHmd provides access to HMD input
/*! SLVRHmd is a subclass of SLVRTrackedDevice with functions for interfacing with HMDs.
 * The (currently) only function returns whether the proximity sensor is activated.
 */
class SLVRHmd : public SLVRTrackedDevice
{
    friend class SLVRSystem;

private:
    vr::VRControllerState_t _state;

protected:
    SLVRHmd(SLVRTrackedDeviceIndex index);
    void updateState();

public:
    SLbool isProximitySensorActivated();
};

#endif // SLPROJECT_SLVRHMD_H
