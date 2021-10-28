//#############################################################################
//  File:      SLArucoPen.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "SLArucoPen.h"
#include <app/AppArucoPen.h>

//-----------------------------------------------------------------------------
SLArucoPen::SLArucoPen(string calibIniPath,
                       float  edgeLength)
  : CVMultiTracked(new CVTrackedArucoCube(calibIniPath, edgeLength))
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
            _lastDistance        = distance;

            return true;
        }
        default: return false;
    }
}
//-----------------------------------------------------------------------------
SLVec3f SLArucoPen::tipPosition()
{
    float tipOffset = -(0.147f - 0.025f + 0.002f);

    CVMatx44f          worldMatrix = CVMatx44f(_objectViewMat);
    CVCaptureProvider* provider    = AppArucoPen::instance().currentCaptureProvider();
    CVCalibration      calibration = provider->camera().calibration;

    if (!calibration.rvec.empty() && !calibration.tvec.empty())
    {
        CVMatx44f extrinsic = CVTracked::createGLMatrix(calibration.tvec, calibration.rvec);
        // clang-format off
        extrinsic = CVMatx44f(-extrinsic.val[ 1],  extrinsic.val[ 2], -extrinsic.val[ 0],  extrinsic.val[3],
                              -extrinsic.val[ 5],  extrinsic.val[ 6], -extrinsic.val[ 4],  extrinsic.val[7],
                              -extrinsic.val[ 9],  extrinsic.val[10], -extrinsic.val[ 8],  extrinsic.val[11],
                               0.0f,                0.0f,               0.0f,               1.0f);
        // clang-format on
        worldMatrix         = extrinsic.inv() * worldMatrix;
    }

    float offsetX = worldMatrix.val[1] * tipOffset;
    float offsetY = worldMatrix.val[5] * tipOffset;
    float offsetZ = worldMatrix.val[9] * tipOffset;

    SLVec3f position(worldMatrix.val[3] + offsetX,
                     worldMatrix.val[7] + offsetY,
                     worldMatrix.val[11] + offsetZ);
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
SLfloat SLArucoPen::lastDistance() const
{
    return _lastDistance;
}
//-----------------------------------------------------------------------------
