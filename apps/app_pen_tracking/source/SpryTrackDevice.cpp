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

    auto optionsEnumCallback = [](uint64 sn, void* user, ftkOptionsInfo* oi)
    {
        SL_LOG("SpryTrack: Found option: %s", oi->name);

        auto* self = (SpryTrackDevice*)user;
        self->_options.insert({oi->name, oi->id});
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
SpryTrackMarker* SpryTrackDevice::findMarker(SpryTrackMarkerID id) const
{
    auto condition = [id](SpryTrackMarker* marker)
    {
        return marker->id() == id;
    };
    auto it = std::find_if(_markers.begin(), _markers.end(), condition);
    return it == _markers.end() ? nullptr : *it;
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::registerMarker(SpryTrackMarker* marker)
{
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
void SpryTrackDevice::unregisterMarker(SpryTrackMarker* marker)
{
    ftkClearGeometry(SpryTrackInterface::instance().library,
                     _serialNumber,
                     marker->id());
    _markers.erase(std::remove(_markers.begin(), _markers.end(), marker), _markers.end());
    delete marker;
}
//-----------------------------------------------------------------------------
void SpryTrackDevice::enableOnboardProcessing()
{
    ftkError error = ftkSetInt32(SpryTrackInterface::instance().library,
                                 _serialNumber,
                                 _options["New marker reco algorithm"],
                                 1);
    if (error != ftkError::FTK_OK)
    {
        SL_WARN_MSG("SpryTrack: Failed to set recognition algorithm");
        return;
    }

    error = ftkSetInt32(SpryTrackInterface::instance().library,
                        _serialNumber,
                        _options["Enable embedded processing"],
                        1);
    if (error != ftkError::FTK_OK)
    {
        SL_WARN_MSG("SpryTrack: Failed to enable onboard processing");
        return;
    }

//    error = ftkSetFloat32(SpryTrackInterface::instance().library,
//                          _serialNumber,
//                          _options["Distance matching tolerance"],
//                          4.0f);
//    if (error != ftkError::FTK_OK)
//    {
//        SL_WARN_MSG("SpryTrack: Failed to set distance matching tolerance");
//        return;
//    }

    error = ftkSetInt32(SpryTrackInterface::instance().library,
                        _serialNumber,
                        _options["Embedded Matching Maximum Missing Points"],
                        2);
    if (error != ftkError::FTK_OK)
    {
        SL_WARN_MSG("SpryTrack: Failed to set maximum missing points");
        return;
    }

    SL_LOG("SpryTrack: Onboard processing enabled");
}
//-----------------------------------------------------------------------------
SpryTrackFrame SpryTrackDevice::acquireFrame()
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
            SpryTrackFrame frame{};
            frame.width         = _frame->imageHeader->width;
            frame.height        = _frame->imageHeader->height;
            frame.dataGrayLeft  = _frame->imageLeftPixels;
            frame.dataGrayRight = _frame->imageRightPixels;
            processFrame();
            return frame;
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

    if (_frame->markersStat != ftkQueryStatus::QS_OK)
    {
        return;
    }

    for (uint32 i = 0; i < _frame->markersCount; i++)
    {
        ftkMarker marker = _frame->markers[i];

        SpryTrackMarker* registeredMarker = findMarker(marker.geometryId);
        registeredMarker->_visible        = true;
        registeredMarker->_errorMM        = marker.registrationErrorMM;
        registeredMarker->update(marker);
    }

    //    SL_LOG("%d %d", _frame->threeDFiducialsCount, _frame->markersCount);

    //    for (int i = 0; i < _frame->threeDFiducialsCount; i++) {
    //        for (int j = 0; j < _frame->threeDFiducialsCount; j++) {
    //            if(i == j) continue;
    //            SLVec3f a(_frame->threeDFiducials[i].positionMM.x, _frame->threeDFiducials[i].positionMM.y, _frame->threeDFiducials[i].positionMM.z);
    //            SLVec3f b(_frame->threeDFiducials[j].positionMM.x, _frame->threeDFiducials[j].positionMM.y, _frame->threeDFiducials[j].positionMM.z);
    //            std::cout << a.distance(b) << " ";
    //        }
    //        std::cout << ";    ";
    //    }
    //    std::cout << "\n";
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