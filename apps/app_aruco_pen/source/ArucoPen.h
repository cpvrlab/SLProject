//#############################################################################
//  File:      ArucoPen.h
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLARUCOPEN_H
#define SLPROJECT_SLARUCOPEN_H

#include <CVTrackedArucoCube.h>
#include <CVCaptureProvider.h>
#include <TrackingSystem.h>
#include <TrackingSystemSpryTrack.h>

#include <SLEventHandler.h>
#include <SLVec3.h>
#include <SLQuat4.h>

//-----------------------------------------------------------------------------
class ArucoPen : public SLEventHandler
{
public:
    enum State
    {
        Idle,
        Tracing
    };

    ~ArucoPen();

    SLbool onKeyPress(SLKey key,
                      SLKey mod) override;
    SLbool onKeyRelease(SLKey key,
                        SLKey mod) override;

    SLVec3f  tipPosition();
    SLQuat4f orientation();

    SLVec3f  rosPosition();
    SLQuat4f rosOrientation();

    SLfloat liveDistance();
    SLfloat lastDistance() const;

    State state() { return _state; }

    TrackingSystem* trackingSystem() { return _trackingSystem; }
    void            trackingSystem(TrackingSystem* trackingSystem);

private:
    State           _state          = Idle;
    TrackingSystem* _trackingSystem = new TrackingSystemSpryTrack();

    SLVec3f _lastPrintedPosition;
    SLbool  _positionPrintedOnce = false;
    SLfloat _lastDistance;
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_SLARUCOPEN_H
