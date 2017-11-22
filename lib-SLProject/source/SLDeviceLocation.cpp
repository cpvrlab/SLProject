//#############################################################################
//  File:      SLDeviceLocation.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#ifdef SL_MEMLEAKDETECT     // set in SL.h for debug config only
#include <debug_new.h>      // memory leak detector
#endif

#include <SLDeviceLocation.h>

//-----------------------------------------------------------------------------
void SLDeviceLocation::init()
{
    _isUsed = false;
    _isFirstSensorValue = false;
    _locLLA.set(0,0,0);
    _locECEF.set(0,0,0);
    _locENU.set(0,0,0);
    _locAccuracyM = 0.0f;
    _originLLA.set(0,0,0);
    _originECEF.set(0,0,0);
    _originENU.set(0,0,0);
    _originAccuracyM = FLT_MAX;
    _wRecef.identity();
    _hasOrigin = false;
    _useOriginAltitude = true;
    _improveOrigin = true;
    _improveTimeSEC = 8.0f;
}
//-----------------------------------------------------------------------------
// Setter for hasOrigin flag.
void SLDeviceLocation::hasOrigin(SLbool hasOrigin)
{
    if (hasOrigin == false)
    {   _improveTimer.start();
        _originAccuracyM = FLT_MAX;
    }
    _hasOrigin = hasOrigin;
}
//-----------------------------------------------------------------------------
//! Event handler for mobile device location update.
/*! Global event handler for device GPS location with longitude and latitude in
degrees and altitude in meters. This location uses the World Geodetic System
1984 (WGS 84). The accuracy in meters is a radius in which the location is with
a probability of 68% (2 sigma). The altitude in m is the most inaccurate
information. The option _useOriginAltitude allows to overwrite the current
altitude with the origins altitude.
*/
void SLDeviceLocation::onLocationLLA(double latDEG,
                                     double lonDEG,
                                     double altM,
                                     float  accuracyM)
{
    // Init origin if it is not set yet or if the origin should be improved
    if (!_hasOrigin || _improveOrigin)
    {
        // The first sensor value can appear after a few seconds.
        if (_isFirstSensorValue)
        {   _improveTimer.start();
            _isFirstSensorValue = false;
        }

        // Only improve if accuracy is higher and the improve time has not elapsed
        if (accuracyM < _originAccuracyM || _improveTimer.elapsedTimeInSec() < _improveTimeSEC)
        {   _originAccuracyM = accuracyM;
            originLLA(latDEG, lonDEG, altM);
        }
    }

    _locLLA.set(latDEG, lonDEG, _useOriginAltitude ? _originLLA.alt : altM);

    _locAccuracyM = accuracyM;

    // Convert to cartesian ECEF coordinates
    _locECEF.lla2ecef(_locLLA);

    // Transform to local east-north-up frame
    _locENU = _wRecef * _locECEF;
}
//-----------------------------------------------------------------------------
//! Initialize global origin in latitude, longitude and altitude.
/*! The calculated values can be used for global camera positioning via GPS
sensor.
*/
void SLDeviceLocation::originLLA(double latDeg, double lonDeg, double altM)
{
    _originLLA = SLVec3d(latDeg, lonDeg, altM);
    _originECEF.lla2ecef(_originLLA);

    //calculation of ecef to world (scene) rotation matrix
    //definition of rotation matrix for ECEF to world frame rotation:
    //world frame (scene) w.r.t. ENU frame
    double phiRad = latDeg * SL_DEG2RAD;  //   phi == latitude
    double lamRad = lonDeg * SL_DEG2RAD;  //lambda == longitude
    double sinPhi = sin(phiRad);
    double cosPhi = cos(phiRad);
    double sinLam = sin(lamRad);
    double cosLam = cos(lamRad);

    SLMat3d enuRecef(-sinLam,                cosLam,      0,
                     -cosLam*sinPhi, -sinLam*sinPhi, cosPhi,
                      cosLam*cosPhi,  sinLam*cosPhi, sinPhi);

    //world frame (scene) w.r.t. ENU frame
    SLMat3d wRenu; //same as before
    wRenu.rotation(-90, 1, 0, 0);

    //world frame (scene) w.r.t. ECEF
    _wRecef = wRenu * enuRecef;
    _originENU = _wRecef * _originECEF;

    //Indicate that origin is set. Otherwise it would be reset on each update
    _hasOrigin = true;
}

//-----------------------------------------------------------------------------
//! Setter that turns on the device rotation sensor
void SLDeviceLocation::isUsed (SLbool use)
{
    if (!_isUsed && use==true)
        _isFirstSensorValue = true;

    _isUsed = use;
}
//------------------------------------------------------------------------------
