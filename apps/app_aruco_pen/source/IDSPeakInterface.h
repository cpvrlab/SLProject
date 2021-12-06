//#############################################################################
//  File:      IDSPeakInterface.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_IDSPEAKINTERFACE_H
#define SLPROJECT_IDSPEAKINTERFACE_H

#include <peak/peak.hpp>
#include <peak_ipl/peak_ipl.hpp>
#include <peak/converters/peak_buffer_converter_ipl.hpp>

#include <IDSPeakDevice.h>
#include <SL.h>

//-----------------------------------------------------------------------------
class IDSPeakInterface
{
    friend class IDSPeakDevice;

public:
    static IDSPeakInterface& instance()
    {
        static IDSPeakInterface instance;
        return instance;
    }

private:
    bool initialized = false;
    int  numDevices  = 0;

public:
    IDSPeakDevice openDevice(int index,
                             IDSPeakDeviceParams& params);
    int           numAvailableDevices();

private:
    void init();
    void uninit();
    void deviceClosed();
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_IDSPEAKINTERFACE_H
