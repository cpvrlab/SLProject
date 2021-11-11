//#############################################################################
//  File:      IDSPeakInterface.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "IDSPeakInterface.h"

//-----------------------------------------------------------------------------
void IDSPeakInterface::init()
{
    try
    {
        peak::Library::Initialize();
        initialized = true;
        SL_LOG("IDS Peak: Initialized");
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
void IDSPeakInterface::uninit()
{
    try
    {
        peak::Library::Close();
        initialized = false;
        SL_LOG("IDS Peak: Uninitialized");
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
IDSPeakDevice IDSPeakInterface::openDevice(int index)
{
    if (!initialized)
    {
        init();
    }

    try
    {
        auto& deviceManager = peak::DeviceManager::Instance();

        // Search for available devices
        deviceManager.Update();

        auto devices = deviceManager.Devices();
        SL_LOG("IDS Peak: Found %d available device(s)", devices.size());

        if (index < 0 || index >= devices.size())
        {
            SL_EXIT_MSG("IDS Peak: Invalid device index");
        }

        // Open the first available device
        auto device = devices.at(index)->OpenDevice(peak::core::DeviceAccessType::Control);
        SL_LOG("IDS Peak: Device \"%s\" opened", device->DisplayName().c_str());
        numDevices++;

        IDSPeakDevice result(device, index);
        result.prepare();
        return result;
    }
    catch (const std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }

    return {};
}
//-----------------------------------------------------------------------------
int IDSPeakInterface::numAvailableDevices()
{
    if (!initialized)
    {
        init();
    }

    try
    {
        auto& deviceManager = peak::DeviceManager::Instance();
        deviceManager.Update();
        return (int)deviceManager.Devices().size();
    }
    catch (const std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
void IDSPeakInterface::deviceClosed()
{
    numDevices--;
    if (initialized && numDevices == 0)
    {
        uninit();
    }
}
//-----------------------------------------------------------------------------
