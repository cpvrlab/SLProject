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

//-----------------------------------------------------------------------------
bool TrackingSystemSpryTrack::track(CVCaptureProvider* provider)
{
    SpryTrackDevice& device = getDevice(provider);
    SpryTrackMarker*       marker = device.markers()[0];
    if (!marker->visible())
    {
        SL_LOG("Marker not detected");
        return false;
    }

    SL_LOG("Marker detected");

    _worldMatrix = _extrinsicMat.inv() * marker->objectViewMat();
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
    SpryTrackCalibrator calibrator(getDevice(provider));
    calibrator.calibrate();
    _extrinsicMat = calibrator.extrinsicMat();
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