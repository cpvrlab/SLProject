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
#include <SLMesh.h>
#include <SLAssetManager.h>

#include <vr/SLVRRenderModel.h>

typedef vr::TrackedDeviceIndex_t SLVRTrackedDeviceIndex;

//! The main class for interfacing with devices
/*! SLVRTrackedDevice provides access to the properties that all tracked VR devices have in common,
 * such as the index, the pose, whether or not this device is connected, etc.
 */
class SLVRTrackedDevice
{
    friend class SLVRSystem;

protected:
    SLVRTrackedDeviceIndex _index;
    SLMat4f                _pose;
    SLVRRenderModel*       _renderModel = nullptr;

    explicit SLVRTrackedDevice(SLVRTrackedDeviceIndex index);
    ~SLVRTrackedDevice();

    vr::IVRSystem* system();
    SLstring       getStringProperty(vr::TrackedDeviceProperty property);

public:
    // Setters
    void pose(const SLMat4f& pose) { _pose = pose; };

    // Getters
    SLVRTrackedDeviceIndex index() const { return _index; };
    SLMat4f                pose() { return _pose; };
    SLVRRenderModel*       renderModel() { return _renderModel; }

    SLbool   isConnected();
    SLbool   isAwake();
    SLstring getManufacturer();

    SLVRRenderModel* loadRenderModel(SLAssetManager* assetManager);
};

#endif // SLPROJECT_SLVRTRACKEDDEVICE_H
