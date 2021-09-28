//#############################################################################
//  File:      SLArucoPen.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "SLArucoPen.h"

//-----------------------------------------------------------------------------
SLArucoPen::SLArucoPen(string calibIniPath,
                       float  edgeLength)
  : CVTrackedArucoCube(calibIniPath, edgeLength)
{
}
//-----------------------------------------------------------------------------
SLbool SLArucoPen::onKeyPress(const SLKey key,
                              const SLKey mod)
{
    switch (key)
    {
        case K_F6: {
            SLVec3f position = tipPosition();
            float   distance = liveDistance();

            SL_LOG("ArUco Pen");
            SL_LOG("\tPosition: %s", position.toString(", ", 4).c_str());

            if (_positionPrintedOnce)
            {
                SL_LOG("\tDistance: %.2fcm", distance * 100.0f);
            }
            else
            {
                SL_LOG("\tTake a second measurement to calculate the distance");
                _positionPrintedOnce = true;
            }

            _lastPrintedPosition = position;
            _lastDistance = distance;

            return true;
        }
        default: return false;
    }
}
//-----------------------------------------------------------------------------
SLVec3f SLArucoPen::tipPosition()
{
    float tipOffset = -(0.147f - 0.025f + 0.002f);

    float offsetX = _objectViewMat.val[1] * tipOffset;
    float offsetY = _objectViewMat.val[5] * tipOffset;
    float offsetZ = _objectViewMat.val[9] * tipOffset;

    SLVec3f position(_objectViewMat.val[3] + offsetX,
                     _objectViewMat.val[7] + offsetY,
                     _objectViewMat.val[11] + offsetZ);
    return position;
}
//-----------------------------------------------------------------------------
SLfloat SLArucoPen::liveDistance()
{
    if (!_positionPrintedOnce)
    {
        return 0.0f;
    }

    SLVec3f position = tipPosition();
    return position.distance(_lastPrintedPosition);
}
//-----------------------------------------------------------------------------
SLfloat SLArucoPen::lastDistance()
{
    return _lastDistance;
}
//-----------------------------------------------------------------------------