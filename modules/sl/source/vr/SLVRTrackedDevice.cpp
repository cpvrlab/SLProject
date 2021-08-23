//#############################################################################
//  File:      SLVRTrackedDevice.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVRTrackedDevice.h>
#include <vr/SLVRSystem.h>

SLVRTrackedDevice::SLVRTrackedDevice(vr::TrackedDeviceIndex_t index) : _index(index)
{
}