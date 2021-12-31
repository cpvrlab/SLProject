//#############################################################################
//  File:      TrackingSystemSpryTrack.cpp
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <TrackingSystemSpryTrack.h>
#include <CVCaptureProviderSpryTrack.h>

//-----------------------------------------------------------------------------
bool TrackingSystemSpryTrack::track(CVCaptureProvider* provider)
{
    auto* providerSpryTrack = dynamic_cast<CVCaptureProviderSpryTrack*>(provider);
    if (!providerSpryTrack)
    {
        SL_LOG("Warning: TrackingSystemSpryTrack requires a CVCaptureProviderSpryTrack");
        return false;
    }

    const SpryTrackDevice& device = providerSpryTrack->device();
    SpryTrackMarker*       marker = device.markers()[0];
    if (!marker->visible()) return false;

    _worldMatrix = marker->objectViewMat();
    return true;
}
//-----------------------------------------------------------------------------
void TrackingSystemSpryTrack::finalizeTracking()
{
}
//-----------------------------------------------------------------------------
CVMatx44f TrackingSystemSpryTrack::worldMatrix()
{
    return _worldMatrix;
}
//-----------------------------------------------------------------------------