//#############################################################################
//  File:      SpryTrackDevice.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_SPRYTRACKDEVICE_H
#define SRC_SPRYTRACKDEVICE_H

#include <ftkInterface.h>
#include <SL.h>
#include <SpryTrackMarker.h>
#include <cstdint>

//-----------------------------------------------------------------------------
typedef ftkDeviceType SpryTrackDeviceType;
typedef uint64        SpryTrackSerialNumber;
//-----------------------------------------------------------------------------
class SpryTrackDevice
{
    friend class SpryTrackInterface;

public:
    SpryTrackDevice();
    ~SpryTrackDevice();

    SpryTrackSerialNumber serialNumber() const { return _serialNumber; }
    SpryTrackDeviceType   type() const { return _type; }
    vector<SpryTrackMarker*> markers() const { return _markers; }

    void registerMarker(SpryTrackMarker* marker);
    void enableOnboardProcessing();
    void acquireImage(int*      width,
                      int*      height,
                      uint8_t** dataGrayLeft,
                      uint8_t** dataGrayRight);
    void close();

private:
    void prepare();
    void enumerateOptions();

private:
    SpryTrackSerialNumber    _serialNumber;
    SpryTrackDeviceType      _type;
    ftkFrameQuery*           _frame;
    vector<SpryTrackMarker*> _markers;
};
//-----------------------------------------------------------------------------
#endif // SRC_SPRYTRACKDEVICE_H
