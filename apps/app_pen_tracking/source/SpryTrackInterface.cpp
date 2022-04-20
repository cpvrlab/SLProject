//#############################################################################
//  File:      SpryTrackInterface.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SpryTrackInterface.h>

//-----------------------------------------------------------------------------
struct DeviceEnumResult
{
    bool             successful;
    SpryTrackDevice* device;
};
//-----------------------------------------------------------------------------
void SpryTrackInterface::init()
{
    library = ftkInit();
    if (!library)
    {
        SL_EXIT_MSG("SpryTrack: Failed to initialize the library");
    }

    SL_LOG("SpryTrack: Initialized");
}
//-----------------------------------------------------------------------------
bool SpryTrackInterface::isDeviceConnected()
{
    SpryTrackDevice dummyDevice{};
    return tryAccessDevice(&dummyDevice);
}
//-----------------------------------------------------------------------------
SpryTrackDevice SpryTrackInterface::accessDevice()
{
    SpryTrackDevice device{};
    if (!tryAccessDevice(&device))
    {
        SL_EXIT_MSG("SpryTrack: Device is not connected");
    }

    SL_LOG("SpryTrack: Device found");
    device.prepare();
    return device;
}
//-----------------------------------------------------------------------------
bool SpryTrackInterface::tryAccessDevice(SpryTrackDevice* outDevice)
{
    auto deviceEnumCallback = [](uint64 sn, void* user, ftkDeviceType type) {
        auto* result                  = (DeviceEnumResult*)user;
        result->successful            = true;
        result->device->_serialNumber = sn;
        result->device->_type         = type;
    };

    DeviceEnumResult result{false, outDevice};
    ftkError error = ftkEnumerateDevices(library, deviceEnumCallback, &result);
    if (error != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("SpryTrack: Failed to enumerate devices");
    }

    //std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    return result.successful;
}
//-----------------------------------------------------------------------------
void SpryTrackInterface::uninit()
{
    if (ftkClose(&library) != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("SpryTrack: Failed to close the library");
    }

    SL_LOG("SpryTrack: Uninitialized");
}
//-----------------------------------------------------------------------------