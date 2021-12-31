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
SpryTrackDevice::~SpryTrackDevice()
{
    for (SpryTrackMarker* marker : _markers)
    {
        delete marker;
    }
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

    enumerateOptions();
    SL_LOG("SpryTrack: Device ready for acquiring and processing frames");
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::enumerateOptions()
{
    // Who thought it would be a good idea not to have constants for device
    // options but names that need to be converted to option ids first??

    auto optionsEnumCallback = [](uint64 sn, void* user, ftkOptionsInfo* oi) {
        SL_LOG("SpryTrack: Found option: %s", oi->name);
    };

    ftkError error = ftkEnumerateOptions(SpryTrackInterface::instance().library,
                                         _serialNumber,
                                         optionsEnumCallback,
                                         this);
    if (error != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("SpryTrack: Failed to enumerate device options");
    }
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::registerMarker(SpryTrackMarker* marker)
{
    marker->_geometry.geometryId = (uint32)_markers.size();
    _markers.push_back(marker);

    ftkError error = ftkSetGeometry(SpryTrackInterface::instance().library,
                                    _serialNumber,
                                    &marker->_geometry);
    if (error != ftkError::FTK_OK)
    {
        SL_EXIT_MSG("SpryTrack: Failed to register geometry");
    }

    SL_LOG("SpryTrack: Geometry of marker with ID %d registered", marker->id());
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::enableOnboardProcessing()
{
    // Coming soon
    //    ftkError error = ftkSetInt32(SpryTrackInterface::instance().library,
    //                                 _serialNumber,
    //                                 ENABLE_ONBO)
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::acquireFrame(int*      width,
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

        if (error == ftkError::FTK_OK &&
            _frame->imageLeftStat == ftkQueryStatus::QS_OK &&
            _frame->imageRightStat == ftkQueryStatus::QS_OK)
        {
            *width         = _frame->imageHeader->width;
            *height        = _frame->imageHeader->height;
            *dataGrayLeft  = _frame->imageLeftPixels;
            *dataGrayRight = _frame->imageRightPixels;

            processFrame();
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
void SpryTrackDevice::processFrame()
{
    for (SpryTrackMarker* marker : _markers)
    {
        marker->_visible = false;
    }

    for (uint32 j = 0; j < _frame->markersCount; j++)
    {
        ftkMarker marker = _frame->markers[j];

        SpryTrackMarker* registeredMarker = _markers[marker.geometryId];
        registeredMarker->_visible        = true;
        registeredMarker->update(marker);
    }
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