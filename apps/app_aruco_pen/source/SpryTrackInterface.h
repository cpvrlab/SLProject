//#############################################################################
//  File:      SpryTrackInterface.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_SPRYTRACKINTERFACE_H
#define SRC_SPRYTRACKINTERFACE_H

#include <ftkInterface.h>
#include <SL.h>
#include <SpryTrackDevice.h>

//-----------------------------------------------------------------------------
class SpryTrackInterface
{
    friend class SpryTrackDevice;

public:
    static SpryTrackInterface& instance()
    {
        static SpryTrackInterface instance;
        return instance;
    }

private:
    SpryTrackInterface() {}

public:
    void init();
    bool isDeviceConnected();
    SpryTrackDevice accessDevice();
    void uninit();

private:
    bool tryAccessDevice(SpryTrackDevice* outDevice);

private:
    ftkLibrary library = nullptr;

};
//-----------------------------------------------------------------------------

#endif // SRC_SPRYTRACKINTERFACE_H
