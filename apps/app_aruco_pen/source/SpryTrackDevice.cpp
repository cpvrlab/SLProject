//#############################################################################
//  File:      SpryTrackDevice.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SpryTrackDevice.h>
#include <SpryTrackInterface.h>

//-----------------------------------------------------------------------------
SpryTrackDevice::SpryTrackDevice()
  : _serialNumber(-1),
    _type(SpryTrackDeviceType::DEV_UNKNOWN_DEVICE),
    _frame(nullptr)
{
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::prepare()
{
    _frame = ftkCreateFrame();
    if (!_frame)
    {
        SL_EXIT_MSG("SpryTrack: Failed to allocate frame memory");
    }

    ftkError frameOptionsResult = ftkSetFrameOptions(true,
                                                     10u,
                                                     128u,
                                                     128u,
                                                     100u,
                                                     10u,
                                                     _frame);
    if (frameOptionsResult != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("SpryTrack: Failed to set frame options");
    }

    SL_LOG("SpryTrack: Device ready for acquiring frames");
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::acquireImage(int*      width,
                                   int*      height,
                                   uint8_t** dataGray)
{
    ftkError error = ftkGetLastFrame(SpryTrackInterface::instance().library,
                                     _serialNumber,
                                     _frame,
                                     1000);

    for (int i = 0; i < 100; i++)
    {
        if (error != ftkError::FTK_OK)
        {
            char message[1024];
            ftkGetLastErrorString(SpryTrackInterface::instance().library, 1024, message);
            SL_LOG("SpryTrack: Failed to grab frame\n%s", message);
        }
        else
        {
            *width    = _frame->imageHeader->width;
            *height   = _frame->imageHeader->height;
            *dataGray = _frame->imageLeftPixels;
            return;
        }
    }

    SL_EXIT_MSG("SpryTrack: Failed to acquire image after 100 attempts");
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::close()
{
    if (ftkDeleteFrame(_frame) != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("SpryTrack: Failed to deallocate frame memory");
    }
}
//-----------------------------------------------------------------------------