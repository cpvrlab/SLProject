//#############################################################################
//  File:      SLArucoPen.hpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLARUCOPEN_H
#define SLPROJECT_SLARUCOPEN_H

#include <cv/CVTrackedArucoCube.h>

#include <SLEventHandler.h>
#include <SLVec3.h>

class SLArucoPen : public CVTrackedArucoCube
  , public SLEventHandler
{
public:
    SLArucoPen(string calibIniPath, float edgeLength);

    SLbool onKeyPress(const SLKey key,
                      const SLKey mod) override;

    SLVec3f tipPosition();

    SLfloat liveDistance();
    SLfloat lastDistance();

private:
    SLVec3f _lastPrintedPosition;
    SLbool _positionPrintedOnce = false;
    SLfloat _lastDistance;

};

#endif // SLPROJECT_SLARUCOPEN_H
