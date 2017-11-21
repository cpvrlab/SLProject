//#############################################################################
//  File:      SLDeviceRotation.cpp
//  Author:    Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#ifdef SL_MEMLEAKDETECT     // set in SL.h for debug config only
#include <debug_new.h>      // memory leak detector
#endif

#include <SLDeviceRotation.h>

//-----------------------------------------------------------------------------
void SLDeviceRotation::init()
{
    _rotation.identity();
    _pitchRAD = 0.0f;
    _yawRAD = 0.0f;
    _rollRAD = 0.0f;
    _zeroYawAtStart = true;
    _startYawRAD = 0.0f;
    _hasStarted = false;
    _isUsed = false;
}
//-----------------------------------------------------------------------------
/*! Event handler for rotation change of a mobile device with Euler angles for
pitch, yaw and roll. This function will only be called in an Android or iOS
project. See onRotationPYR in GLES3Activity.java in the Android project.
This handler is only called if the flag SLScene::_usesRotation is true. If so
the mobile device turns on it's IMU sensor system. The device rotation is so
far only used in SLCamera::setViev if the cameras animation is on CA_deciveRotYUp.
If _zeroYawAfterStart is true the start yaw value is subtracted. This means
that the magnetic north will be ignored.
The angles should be:\n
Pitch from -halfpi (down)  to zero (horizontal) to +halfpi (up)\n
Yaw   from -pi     (south) to zero (north)      to +pi     (south)\n
Roll  from -halfpi (ccw)   to zero (horizontal) to +halfpi (clockwise)\n
*/
void SLDeviceRotation::onRotationPYR(SLfloat pitchRAD,
                                     SLfloat yawRAD,
                                     SLfloat rollRAD)
{
    _pitchRAD = pitchRAD;
    _yawRAD   = yawRAD;
    _rollRAD  = rollRAD;
}
//-----------------------------------------------------------------------------
/*! onRotationQUAT: Event handler for rotation change of a mobile device with
rotation quaternion.
*/
void SLDeviceRotation::onRotationQUAT(SLfloat quatX,
                                      SLfloat quatY,
                                      SLfloat quatZ,
                                      SLfloat quatW)
{
    SLQuat4f quat(quatX, quatY, quatZ, quatW);
    _rotation = quat.toMat3();

    if (_zeroYawAtStart)
    {
        if (_hasStarted  )
        {
            //store initial rotation in yaw for referencing of initial alignment
            _startYawRAD = _yawRAD;
            _hasStarted = false;
        }
    }
}
//-----------------------------------------------------------------------------
//! Setter that turns on the device rotation sensor
void SLDeviceRotation::isUsed (SLbool use)
{
    if (!_isUsed && use==true)
        _hasStarted = true;

    _isUsed = use;
}
//------------------------------------------------------------------------------
