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

vr::IVRSystem* SLVRTrackedDevice::system()
{
    return SLVRSystem::instance().system();
}

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

SLbool SLVRTrackedDevice::isConnected()
{
    return system()->IsTrackedDeviceConnected(_index);
}

SLbool SLVRTrackedDevice::isAwake()
{
    vr::EDeviceActivityLevel level = system()->GetTrackedDeviceActivityLevel(_index);
    return level == vr::EDeviceActivityLevel::k_EDeviceActivityLevel_UserInteraction ||
           level == vr::EDeviceActivityLevel::k_EDeviceActivityLevel_UserInteraction_Timeout;
}

SLstring SLVRTrackedDevice::getManufacturer()
{
    return getStringProperty(vr::TrackedDeviceProperty::Prop_ManufacturerName_String);
}