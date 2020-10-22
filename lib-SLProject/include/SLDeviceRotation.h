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
//! Encapsulation of a mobile device rotation set by the device's IMU sensor
/*! This class is only used if SLProject runs on a mobile device. Check out the
 app-Demo-SLProject/android and app-Demo-SLProject/iOS how the sensor data is
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
    SLDeviceRotation() { init(); }
    void init();
    void onRotationQUAT(SLfloat quatX,
                        SLfloat quatY,
                        SLfloat quatZ,
                        SLfloat quatW);
    // Setters
    void isUsed(SLbool isUsed);
    void hasStarted(SLbool started) { _isFirstSensorValue = started; }
    void zeroYawAtStart(SLbool zeroYaw) { _zeroYawAtStart = zeroYaw; }
    void rotationOffset(const SLMat3f& rotOffset) { _rotationOffset = rotOffset; }
    void numAveragedPYR(SLint numAvgPYR)
    {
        assert(numAvgPYR > 0 && "Num. of averaged values must be greater than zero");
        _numAveragedPYR = numAvgPYR;
        _pitchAvgRAD.init(numAvgPYR, _pitchAvgRAD.average());
        _yawAvgRAD.init(numAvgPYR, _yawAvgRAD.average());
        _rollAvgRAD.init(numAvgPYR, _rollAvgRAD.average());
    }

    // Getters
    SLbool  isUsed() const { return _isUsed; }
    SLMat3f rotation() const { return _rotation; }
    SLMat3f rotationAveraged()
    {
        SLMat3f rx(_pitchAvgRAD.average()*Utils::RAD2DEG, 1,0,0);
        SLMat3f ry(_yawAvgRAD.average()*Utils::RAD2DEG, 0,1,0);
        SLMat3f rz(_rollAvgRAD.average()*Utils::RAD2DEG, 0,0,1);
        return rx * ry * rz;
        //return SLMat3f(_pitchAvgRAD.average(),
        //               _yawAvgRAD.average(),
        //               _rollAvgRAD.average());
    }
    SLMat3f  rotationOffset() const { return _rotationOffset; }
    SLQuat4f quaternion() const { return _quaternion; }
    SLfloat  pitchRAD() const { return _pitchRAD; }
    SLfloat  pitchDEG() const { return _pitchRAD * Utils::RAD2DEG; }
    SLfloat  pitchAvgDEG() { return _pitchAvgRAD.average() * Utils::RAD2DEG; }
    SLfloat  yawRAD() const { return _yawRAD; }
    SLfloat  yawDEG() const { return _yawRAD * Utils::RAD2DEG; }
    SLfloat  yawAvgDEG() { return _yawAvgRAD.average() * Utils::RAD2DEG; }
    SLfloat  rollRAD() const { return _rollRAD; }
    SLfloat  rollDEG() const { return _rollRAD * Utils::RAD2DEG; }
    SLfloat  rollAvgDEG() { return _rollAvgRAD.average() * Utils::RAD2DEG; }
    SLbool   zeroYawAtStart() const { return _zeroYawAtStart; }
    SLfloat  startYawRAD() const { return _startYawRAD; }
    SLint    numAveragedPYR() const { return _numAveragedPYR; }

private:
    SLbool   _isUsed;             //!< Flag if device rotation is used
    SLbool   _isFirstSensorValue; //!< Flag for the first sensor values
    SLfloat  _pitchRAD;           //!< Device pitch angle in radians
    SLfloat  _yawRAD;             //!< Device yaw angle in radians
    SLfloat  _rollRAD;            //!< Device roll angle in radians
    SLint    _numAveragedPYR;     //!< Number of averaged PYR angles
    AvgFloat _pitchAvgRAD;        //!< Averaged device pitch angle in radians
    AvgFloat _yawAvgRAD;          //!< Averaged device yaw angle in radians
    AvgFloat _rollAvgRAD;         //!< Averaged device roll angle in radians
    SLMat3f  _rotation;           //!< Mobile device rotation as matrix
    SLMat3f  _rotationOffset;     //! Offset rotation matrix for slight correction
    SLQuat4f _quaternion;         //! Quaternion rotation that is set by IMU
    SLbool   _zeroYawAtStart;     //!< Flag if yaw angle should be zeroed at sensor start
    SLfloat  _startYawRAD;        //!< Initial yaw angle after _zeroYawAfterSec in radians
};
//-----------------------------------------------------------------------------
#endif
