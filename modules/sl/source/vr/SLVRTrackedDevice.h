//#############################################################################
//  File:      SLVRTrackedDevice.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRTRACKEDDEVICE_H
#define SLPROJECT_SLVRTRACKEDDEVICE_H

#include <openvr.h>
#include <SLMat4.h>
#include <SL.h>

class SLVRSystem; // Forward declaration of SLVRSystem for friend declaration

//! The main class for interfacing with devices
/*! SLVRTrackedDevice provides access to the properties that all tracked VR devices have in common.
 * These properties are the index and the pose.
 */
class SLVRTrackedDevice
{
    friend class SLVRSystem; // Only SLVRSystem is allowed to instantiate this class

private:
    vr::TrackedDeviceIndex_t _index;
    SLMat4f                  _pose;

    explicit SLVRTrackedDevice(vr::TrackedDeviceIndex_t index);

    SLstring getStringProperty(vr::TrackedDeviceProperty property);

public:
    // Setters
    void pose(const SLMat4f& pose) { _pose = pose; };

    // Getters
    vr::TrackedDeviceIndex_t index() const { return _index; };
    SLMat4f                  pose() { return _pose; };

    SLbool isConnected();
    SLbool isAwake();
    SLstring getManufacturer();
};

#endif // SLPROJECT_SLVRTRACKEDDEVICE_H
