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

public:
    void          init();
    IDSPeakDevice openDevice(int                  index,
                             IDSPeakDeviceParams& params);
    int           numAvailableDevices();
    void          uninit();
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_IDSPEAKINTERFACE_H
