//#############################################################################
//  File:      SLDeviceRotation.h
//  Purpose:   Mobile device rotation class declaration
//  Author:    Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLDEVICEROTATION_H
#define SLDEVICEROTATION_H

#include <SL.h>
#include <SLMat3.h>
#include <SLQuat4.h>
#include <Averaged.h>

//-----------------------------------------------------------------------------
//! Device rotation offset mode
enum SLOffsetMode
{
    OM_none = 0,
    OM_fingerX,
    OM_fingerXY,
    OM_fingerYTrans,
    OM_fingerXRotYTrans,
    OM_autoX,
    OM_autoXY,
};
//-----------------------------------------------------------------------------
//! Encapsulation of a mobile device rotation set by the device's IMU sensor
/*! This class is only used if SLProject runs on a mobile device. Check out the
 app-Demo-SLProject/android and app_demo_slproject/ios how the sensor data is
 generated and passed to this object hold by SLApplication.
 It stores the devices rotation that it gets from its IMU (inertial measurement
 unit) sensor. This is a fused orientation that is calculated from the
 magnetometer, the accelerometer and the gyroscope. The device rotation can
 be used in the active camera to apply it to the scene camera
 (s. SLCamera::setView).
*/
class SLDeviceRotation
{
public:
    SLDeviceRotation();
    void init();
    void onRotationQUAT(SLfloat quatX,
                        SLfloat quatY,
                        SLfloat quatZ,
                        SLfloat quatW);
    // Setters
    void isUsed(SLbool isUsed);
    void hasStarted(SLbool started) { _isFirstSensorValue = started; }
    void zeroYawAtStart(SLbool zeroYaw) { _zeroYawAtStart = zeroYaw; }
    void numAveraged(SLint numAvg);
    void offsetMode(SLOffsetMode om) { _offsetMode = om; }
    void updateRPY(SLbool doUpdate) { _updateRPY = doUpdate; }

    // Getters
    SLbool       isUsed() const { return _isUsed; }
    SLMat3f      rotation() const { return _rotation; }
    SLMat3f      rotationAveraged() { return _rotationAvg.average(); }
    SLQuat4f     quaternion() const { return _quaternion; }
    SLfloat      pitchRAD() const { return _pitchRAD; }
    SLfloat      pitchDEG() const { return _pitchRAD * Utils::RAD2DEG; }
    SLfloat      yawRAD() const { return _yawRAD; }
    SLfloat      yawDEG() const { return _yawRAD * Utils::RAD2DEG; }
    SLfloat      rollRAD() const { return _rollRAD; }
    SLfloat      rollDEG() const { return _rollRAD * Utils::RAD2DEG; }
    SLbool       zeroYawAtStart() const { return _zeroYawAtStart; }
    SLfloat      startYawRAD() const { return _startYawRAD; }
    SLint        numAveraged() { return (int)_rotationAvg.size(); }
    SLOffsetMode offsetMode() { return _offsetMode; }
    SLstring     offsetModeStr() const;
    SLbool       updateRPY() const { return _updateRPY; }

private:
    SLbool            _isUsed;             //!< Flag if device rotation is used
    SLbool            _isFirstSensorValue; //!< Flag for the first sensor values
    SLfloat           _pitchRAD;           //!< Device pitch angle in radians
    SLfloat           _yawRAD;             //!< Device yaw angle in radians
    SLfloat           _rollRAD;            //!< Device roll angle in radians
    SLMat3f           _rotation;           //!< Mobile device rotation as matrix
    Averaged<SLMat3f> _rotationAvg;        //!< Component wise averaged rotation matrix
    SLQuat4f          _quaternion;         //! Quaternion rotation that is set by IMU
    SLbool            _zeroYawAtStart;     //!< Flag if yaw angle should be zeroed at sensor start
    SLfloat           _startYawRAD;        //!< Initial yaw angle after _zeroYawAfterSec in radians
    SLOffsetMode      _offsetMode;         //!< Rotation offset mode
    SLbool            _updateRPY;          //!< Calculate roll pitch yaw in onRotationQUAT
};
//-----------------------------------------------------------------------------
#endif