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

SLVRTrackedDevice::~SLVRTrackedDevice() noexcept
{
    delete _renderModel;
}

/*! Function for accessing vr::IVRSystem* quickly
 * @return The instance of vr::IVRSystem that SLVRSystem uses
 */
vr::IVRSystem* SLVRTrackedDevice::system()
{
    return SLVRSystem::instance().system();
}

/*! Utility function for getting a string property from OpenVR
 * @param property The property whose value will be returned
 * @return The value of the property as a SLstring
 */
SLstring SLVRTrackedDevice::getStringProperty(vr::TrackedDeviceProperty property)
{
    // Create string buffer
    uint32_t bufferSize = vr::k_unMaxPropertyStringSize;
    char*    buffer     = new char[bufferSize];

    // Read property value into buffer
    system()->GetStringTrackedDeviceProperty(_index, property, buffer, bufferSize);

    // Convert char array to string and deallocate buffer
    std::string result = std::string(buffer);
    delete[] buffer;

    return result;
}

/*! Returns whether or not this device is connected
 * @return True if the device is connected, false otherwise
 */
SLbool SLVRTrackedDevice::isConnected()
{
    return system()->IsTrackedDeviceConnected(_index);
}

/*! Returns whether or not this device is awake
 * A device is awake if it has had activity in the last 5 seconds
 * @return True if the device is awake, false otherwise
 */
SLbool SLVRTrackedDevice::isAwake()
{
    vr::EDeviceActivityLevel level = system()->GetTrackedDeviceActivityLevel(_index);
    return level == vr::EDeviceActivityLevel::k_EDeviceActivityLevel_UserInteraction ||
           level == vr::EDeviceActivityLevel::k_EDeviceActivityLevel_UserInteraction_Timeout;
}

/*! Returns the name of the manufacturer
 * @return The name of the manufacturer
 */
SLstring SLVRTrackedDevice::getManufacturer()
{
    return getStringProperty(vr::TrackedDeviceProperty::Prop_ManufacturerName_String);
}

/*! Loads the render model for this device from disk and returns it
 * The render model can also be accessed later through the renderModel getter
 * @param assetManager The asset manager that will own the assets of this render model
 * @return The loaded render model
 */
SLVRRenderModel* SLVRTrackedDevice::loadRenderModel(SLAssetManager* assetManager)
{
    SLstring         renderModelName = getStringProperty(vr::ETrackedDeviceProperty::Prop_RenderModelName_String);
    _renderModel     = new SLVRRenderModel();
    _renderModel->load(renderModelName, assetManager);
    return _renderModel;
}