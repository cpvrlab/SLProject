//#############################################################################
//  File:      SLDeviceRotation.cpp
//  Author:    Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLDeviceRotation.h>

//-----------------------------------------------------------------------------
void SLDeviceRotation::init()
{
    _rotation.identity();
    _pitchRAD           = 0.0f;
    _yawRAD             = 0.0f;
    _rollRAD            = 0.0f;
    _zeroYawAtStart     = true;
    _startYawRAD        = 0.0f;
    _isFirstSensorValue = false;
    _isUsed             = false;
}
//-----------------------------------------------------------------------------
/*! onRotationQUAT: Event handler for rotation change of a mobile device from a
rotation quaternion. This function will only be called in an Android or iOS
project. See e.g. onSensorChanged in GLES3Activity.java in the Android project.
This handler is only called if the flag SLScene::_usesRotation is true. If so
the mobile device turns on it's IMU sensor system. The device rotation is so
far only used in SLCamera::setViev if the cameras animation is on CA_deciveRotYUp.
If _zeroYawAfterStart is true the start yaw value is subtracted. This means
that the magnetic north will be ignored.
The angles should be:\n
Roll  from -halfpi (ccw)   to zero (horizontal) to +halfpi (clockwise)\n
Pitch from -halfpi (down)  to zero (horizontal) to +halfpi (up)\n
Yaw   from -pi     (south) to zero (north)      to +pi     (south)\n
*/
void SLDeviceRotation::onRotationQUAT(SLfloat quatX,
                                      SLfloat quatY,
                                      SLfloat quatZ,
                                      SLfloat quatW)
{
    _quaternion = SLQuat4f(quatX, quatY, quatZ, quatW);
    _rotation = _quaternion.toMat3();
    _quaternion.toEulerAnglesXYZ(_rollRAD, _pitchRAD, _yawRAD);

    //_rotation.print("Rotation:\n");

    /*   Up   North
         |  /                       Rotation Matrix:
         | /
         |/                       iOS           Android
         +------ East        +- E  N  U -+   +-         -+
        +-------------+      |  1  0  0  |   |  0 -1  0  |
       / +-------+   /       |  0  1  0  |   |  1  0  0  |
      / /       / 0 /        |  0  0  1  |   |  0  0  1  |
     / +-------+   /         +-         -+   +-         -+
    +-------------+

        Up   North
         |  /                       Rotation Matrix:
         | /
         |/                       iOS           Android
         +------ East        +-         -+   +-         -+
         +------------+      |  0  0 -1  |   |  0  0  1  |
         | +------+   |      |  0  1  0  |   |  0 -1  0  |
         | |      | 0 |      |  1  0  0  |   |  1  0  0  |
         | +------+   |      +-         -+   +-         -+
         +------------+

        Up   North
         |  /                       Rotation Matrix:
         | /
         |/                       iOS           Android
         +------ East        +-         -+   +-         -+
        +---------+          |  0  1  0  |   |           |
       / +-----+ /           | -1  0  0  |   |           |
      / /     / /            |  0  0  1  |   |           |
     / /     / /             +-         -+   +-         -+
    / +-----+ /
   /    0    /
  +---------+

        Up   North
         |  /                       Rotation Matrix:
         | /
         |/                       iOS           Android
         +------ East        +-         -+   +-         -+
         +---------+         |  1  0  0  |   |           |
         | +-----+ |         |  0  0 -1  |   |           |
         | |     | |         |  0  1  0  |   |           |
         | |     | |         +-         -+   +-         -+
         | +-----+ |
         |    0    |
         +---------+
     */

    if (_zeroYawAtStart)
    {
        if (_isFirstSensorValue)
        {
            //store initial rotation in yaw for referencing of initial alignment
            _startYawRAD        = _yawRAD;
            _isFirstSensorValue = false;
        }
    }
}
//-----------------------------------------------------------------------------
//! Setter that turns on the device rotation sensor
void SLDeviceRotation::isUsed(SLbool use)
{
    if (!_isUsed && use == true)
        _isFirstSensorValue = true;

    _isUsed = use;
}
//------------------------------------------------------------------------------
