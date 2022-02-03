//#############################################################################
//  File:      TrackingSystemSpryTrack.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <TrackingSystemSpryTrack.h>
#include <SpryTrackCalibrator.h>
#include <app/AppPenTrackingConst.h>

//-----------------------------------------------------------------------------
constexpr float MAX_MARKER_ERROR_MM = 1.5f;
//-----------------------------------------------------------------------------
bool TrackingSystemSpryTrack::track(CVCaptureProvider* provider)
{
    SpryTrackMarker* marker = getDevice(provider).findMarker(2);
    if (!marker->visible() || marker->errorMM() > MAX_MARKER_ERROR_MM)
    {
        return false;
    }

    _worldMatrix = _extrinsicMat.inv() * marker->objectViewMat() * _markerMat;

    return true;
}
//-----------------------------------------------------------------------------
void TrackingSystemSpryTrack::finalizeTracking()
{
    // Nothing to do, tracking is finalized in TrackingSystemSpryTrack::track
}
//-----------------------------------------------------------------------------
CVMatx44f TrackingSystemSpryTrack::worldMatrix()
{
    return _worldMatrix;
}
//-----------------------------------------------------------------------------
void TrackingSystemSpryTrack::calibrate(CVCaptureProvider* provider)
{
    float               squareSize = AppPenTrackingConst::SQUARE_SIZE;
    CVSize2f            planeSize((float)AppPenTrackingConst::CHESSBOARD_WIDTH * squareSize,
                       (float)AppPenTrackingConst::CHESSBOARD_HEIGHT * squareSize);
    SpryTrackCalibrator calibrator(getDevice(provider), planeSize);
    calibrator.calibrate();
    _extrinsicMat = calibrator.extrinsicMat();

    SpryTrackMarker* marker = getDevice(provider).findMarker(2);
//    CVMatx44f markerMat = _extrinsicMat.inv() * marker->objectViewMat();
//    _markerMat = markerMat.inv();
//    _markerMat.val[3] -= 0.03f;
//    _markerMat.val[11] += 0.03f;
    _markerMat = CVMatx44f(1.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f, 0.01f,
                           0.0f, 0.0f, 1.0f, 0.01f + 0.01f,
                           0.0f, 0.0f, 0.0f, 1.0f);

    std::cout << _extrinsicMat << std::endl;
}
//-----------------------------------------------------------------------------
bool TrackingSystemSpryTrack::isAcceptedProvider(CVCaptureProvider* provider)
{
    return typeid(*provider) == typeid(CVCaptureProviderSpryTrack);
}
//-----------------------------------------------------------------------------
SpryTrackDevice& TrackingSystemSpryTrack::getDevice(CVCaptureProvider* provider)
{
    auto* providerSpryTrack = dynamic_cast<CVCaptureProviderSpryTrack*>(provider);
    if (!providerSpryTrack)
    {
        SL_EXIT_MSG("Warning: TrackingSystemSpryTrack requires a CVCaptureProviderSpryTrack");
    }

    return providerSpryTrack->device();
}
//-----------------------------------------------------------------------------