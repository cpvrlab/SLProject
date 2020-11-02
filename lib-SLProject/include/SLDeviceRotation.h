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
    //void pitchOffsetRAD(SLfloat pitchRAD) { _pitchOffsetRAD = pitchRAD; }
    //void yawOffsetRAD(SLfloat yawRAD) { _yawOffsetRAD = yawRAD; }
    void numAveraged(SLint numAvg)
    {
        assert(numAvg > 0 && "Num. of averaged values must be greater than zero");
        _rotationAvg.init(numAvg, _rotationAvg.average());
    }
    void offsetMode(SLOffsetMode om)
    {
        _offsetMode = om;
        //_pitchOffsetRAD = 0.0f;
        //_yawOffsetRAD = 0.0f;
    }
    void offsetScale(SLfloat os) { _offsetScale = os; }

    // Getters
    SLbool       isUsed() const { return _isUsed; }
    SLMat3f      rotation() const { return _rotation; }
    SLMat3f      rotationAveraged() { return _rotationAvg.average(); }
    SLQuat4f     quaternion() const { return _quaternion; }
    SLfloat      pitchRAD() const { return _pitchRAD; }
    SLfloat      pitchDEG() const { return _pitchRAD * Utils::RAD2DEG; }
    //SLfloat      pitchOffsetRAD() const { return _pitchOffsetRAD; };
    //SLfloat      pitchOffsetDEG() const { return _pitchOffsetRAD * Utils::RAD2DEG; };
    SLfloat      yawRAD() const { return _yawRAD; }
    SLfloat      yawDEG() const { return _yawRAD * Utils::RAD2DEG; }
    //SLfloat      yawOffsetRAD() const { return _yawOffsetRAD; };
    //SLfloat      yawOffsetDEG() const { return _yawOffsetRAD * Utils::RAD2DEG; };
    SLfloat      rollRAD() const { return _rollRAD; }
    SLfloat      rollDEG() const { return _rollRAD * Utils::RAD2DEG; }
    SLbool       zeroYawAtStart() const { return _zeroYawAtStart; }
    SLfloat      startYawRAD() const { return _startYawRAD; }
    SLint        numAveraged() { return _rotationAvg.size(); }
    SLOffsetMode offsetMode() { return _offsetMode; }
    SLstring     offsetModeStr()
    {
        switch(_offsetMode)
        {
            case OM_none: return "None";
            case OM_fingerX: return "Finger X";
            case OM_fingerXY: return "Finger X&Y";
            case OM_autoX: return "Auto X";
            case OM_autoXY: return "auto X&Y";
            default: return "Unknown";
        }
    }
    SLfloat      offsetScale() { return _offsetScale; }

    const SLMat3f enucorrRenu() const { return _enucorrRenu; }
    void addRotationToEnucorrRenu(const SLMat3f& rot)
    {
        //we have to right multiply new rotation because new rotations are estimated w.r.t. enu coordinate frame
        _enucorrRenu = _enucorrRenu * rot;
    }
private:
    SLbool            _isUsed;             //!< Flag if device rotation is used
    SLbool            _isFirstSensorValue; //!< Flag for the first sensor values
    SLfloat           _pitchRAD;           //!< Device pitch angle in radians
    SLfloat           _yawRAD;             //!< Device yaw angle in radians
    SLfloat           _rollRAD;            //!< Device roll angle in radians
    SLMat3f           _rotation;           //!< Mobile device rotation as matrix
    Averaged<SLMat3f> _rotationAvg;        //!< Component wise averaged rotation matrix
    //SLfloat           _pitchOffsetRAD;     //!< Additional pitch offset angle in radians
    //SLfloat           _yawOffsetRAD;       //!< Additional yaw offset angle in radians
    SLQuat4f          _quaternion;         //! Quaternion rotation that is set by IMU
    SLbool            _zeroYawAtStart;     //!< Flag if yaw angle should be zeroed at sensor start
    SLfloat           _startYawRAD;        //!< Initial yaw angle after _zeroYawAfterSec in radians
    SLOffsetMode      _offsetMode;         //!< Rotation offset mode
    SLfloat           _offsetScale;        //!< Rotation offset scale factor
    
    SLMat3f _enucorrRenu;
};
//-----------------------------------------------------------------------------
#endif
