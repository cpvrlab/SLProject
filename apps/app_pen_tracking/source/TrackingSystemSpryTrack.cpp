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
constexpr float MAX_MARKER_ERROR_MM = 0.4f;
//-----------------------------------------------------------------------------
bool TrackingSystemSpryTrack::track(CVCaptureProvider* provider)
{
    // clang-format off
//    _extrinsicMat = CVMatx44f(0.99671555, 0.0017340181, -0.080963552, -0.03965205,
//                              0.0040809154, 0.99742502, 0.071600921, -0.18748917,
//                              0.080879226, -0.071696162, 0.99414194, -0.4222953,
//                              0, 0, 0, 1);
    // clang-format on

    SpryTrackMarker* marker = getDevice(provider).markers()[0];
    if (!marker->visible() || marker->errorMM() > MAX_MARKER_ERROR_MM)
    {
        return false;
    }

    _worldMatrix = _extrinsicMat.inv() * marker->objectViewMat();

    CVMatx44f translation(1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, -0.032f,
                          0.0f, 0.0f, 0.0f, 1.0f);
    _worldMatrix = translation * _worldMatrix;

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
    float               squareSize = AppPenTrackingConst::CALIB_SQUARE_SIZE;
    CVSize2f            planeSize((float)AppPenTrackingConst::CALIB_CHESSBOARD_WIDTH * squareSize,
                       (float)AppPenTrackingConst::CALIB_CHESSBOARD_HEIGHT * squareSize);
    SpryTrackCalibrator calibrator(getDevice(provider), planeSize);
    calibrator.calibrate();
    _extrinsicMat = calibrator.extrinsicMat();

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