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
constexpr int    MAX_FAILED_ACQUISITION_ATTEMPTS = 100;
constexpr uint32 MAX_FRAME_WAIT_TIMEOUT_MS       = 100;
constexpr uint32 MAX_EVENTS                      = 10;
constexpr uint32 MAX_RAW_DATA                    = 128;
constexpr uint32 MAX_3D_FIDUCIALS                = 100;
constexpr uint32 MAX_MARKERS                     = 10;
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
        ftkDeleteFrame(_frame);
        SL_EXIT_MSG("SpryTrack: Failed to allocate frame memory");
    }

    ftkError frameOptionsResult = ftkSetFrameOptions(true,
                                                     MAX_EVENTS,
                                                     MAX_RAW_DATA,
                                                     MAX_RAW_DATA,
                                                     MAX_3D_FIDUCIALS,
                                                     MAX_MARKERS,
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
                                   uint8_t** dataGrayLeft,
                                   uint8_t** dataGrayRight)
{
    for (int i = 0; i < MAX_FAILED_ACQUISITION_ATTEMPTS; i++)
    {
        ftkError error = ftkGetLastFrame(SpryTrackInterface::instance().library,
                                         _serialNumber,
                                         _frame,
                                         MAX_FRAME_WAIT_TIMEOUT_MS);

        if (error == ftkError::FTK_OK)
        {
            *width         = _frame->imageHeader->width;
            *height        = _frame->imageHeader->height;
            *dataGrayLeft  = _frame->imageLeftPixels;
            *dataGrayRight = _frame->imageRightPixels;
            return;
        }
        else
        {
            SL_LOG("SpryTrack: Failed to grab frame");
        }
    }

    SL_EXIT_MSG(("SpryTrack: Failed to acquire image after " +
                 std::to_string(MAX_FAILED_ACQUISITION_ATTEMPTS) +
                 " attempts")
                  .c_str());
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