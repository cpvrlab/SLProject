//#############################################################################
//  File:      SLDeviceRotation.cpp
//  Authors:   Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLDeviceRotation.h>

//-----------------------------------------------------------------------------
SLDeviceRotation::SLDeviceRotation()
{
    init();
}
//-----------------------------------------------------------------------------
void SLDeviceRotation::init()
{
    _rotation.identity();
    _pitchRAD = 0.0f;
    _yawRAD   = 0.0f;
    _rollRAD  = 0.0f;
    _rotationAvg.init(3, SLMat3f());
    _zeroYawAtStart     = true;
    _startYawRAD        = 0.0f;
    _isFirstSensorValue = false;
    _isUsed             = false;
    _offsetMode         = ROM_none;
    _updateRPY          = true;
}
//-----------------------------------------------------------------------------
/*! onRotationQUAT: Event handler for rotation change of a mobile device from a
 rotation quaternion. This function will only be called in an Android or iOS
 project. See e.g. onSensorChanged in GLES3Activity.java in the Android project.
 This handler is only called if the flag SLScene::_usesRotation is true. If so
 the mobile device turns on it's IMU sensor system. The device rotation is so
 far only used in SLCamera::setView if the cameras animation is on CA_deciveRotYUp.
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
    _rotation   = _quaternion.toMat3();
    _rotationAvg.set(_rotation);

    if (_updateRPY)
        _quaternion.toEulerAnglesXYZ(_pitchRAD, _rollRAD, _yawRAD);

    /*
     Android sensor coordinate system:
     (https://developer.android.com/guide/topics/sensors/sensors_overview)

    Up = z   North = y
         |  /
         | /
         |/
         +------ East = x
        +---------+
       / +-----+ /
      / /     / /
     / /     / /
    / +-----+ /
   /    0    /
  +---------+

     iOS sensor coordinate system:
     (https://developer.apple.com/documentation/coremotion/getting_processed_device-motion_data/understanding_reference_frames_and_device_attitude)
     In iOS we configure CMMotionManager with xMagneticNorthZVertical which means its a frame, where x points north, y points west and z points up (NWU).
     In the iOS code, we add rotation of 90 deg. around z-axis to relate the sensor rotation to an ENU-frame (as in Android).

     Up = z   West = y
          |  /
          | /
          |/
          +------ North = x
         +---------+
        / +-----+ /
       / /     / /
      / /     / /
     / +-----+ /
    /    0    /
   +---------+

     */

    if (_zeroYawAtStart && _isFirstSensorValue)
    {
        // store initial rotation in yaw for referencing of initial alignment
        _startYawRAD        = _yawRAD;
        _isFirstSensorValue = false;
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
//! Returns the device rotation averaged over multple frames
void SLDeviceRotation::numAveraged(SLint numAvg)
{
    assert(numAvg > 0 && "Num. of averaged values must be greater than zero");
    _rotationAvg.init(numAvg, _rotationAvg.average());
}
//------------------------------------------------------------------------------
//! Returns the rotation offset mode as string
SLstring SLDeviceRotation::offsetModeStr() const
{
    switch (_offsetMode)
    {
        case ROM_none: return "None";
        case ROM_oneFingerX: return "OneFingerX";
        case ROM_oneFingerXY: return "OneFingerXY";
        default: return "Unknown";
    }
}
//------------------------------------------------------------------------------
