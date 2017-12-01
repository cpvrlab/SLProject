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
#include <spa.h>

//-----------------------------------------------------------------------------
void SLDeviceLocation::init()
{
    _isUsed = false;
    _isFirstSensorValue = false;
    _locLLA.set(0,0,0);
    _locECEF.set(0,0,0);
    _locENU.set(0,0,0);
    _locAccuracyM = 0.0f;
    _locMaxDistanceM = 1000.0f;
    _defaultLLA.set(0,0,0);
    _defaultENU.set(0,0,0);
    _originLLA.set(0,0,0);
    _originECEF.set(0,0,0);
    _originENU.set(0,0,0);
    _originAccuracyM = FLT_MAX;
    _originSolarZenit = 45.0f;
    _originSolarAzimut = 0.0f; 
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
            defaultLLA(latDEG, lonDEG, altM);
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
//! Set global origin in latitude, longitude and altitude.
/*! The calculated values can be used for global camera positioning via GPS
sensor.
*/
void SLDeviceLocation::originLLA(double latDEG, double lonDEG, double altM)
{
    _originLLA = SLVec3d(latDEG, lonDEG, altM);
    _originECEF.lla2ecef(_originLLA);

    //calculation of ecef to world (scene) rotation matrix
    //definition of rotation matrix for ECEF to world frame rotation:
    //world frame (scene) w.r.t. ENU frame
    double phiRad = latDEG * SL_DEG2RAD;  //   phi == latitude
    double lamRad = lonDEG * SL_DEG2RAD;  //lambda == longitude
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
//! Sets the default location in latitude, longitude and altitude.
/*! It must be called after setting the origin. If no origin is set with it
will be automatically set in onLocationLLA. The default location is used by
the camera in SLCamera::setView if the current distance between _locENU and
_originENU is greater than _locMaxDistanceM.
*/
void SLDeviceLocation::defaultLLA(double latDEG, double lonDEG, double altM)
{
    _defaultLLA.set(latDEG, lonDEG, _useOriginAltitude ? _originLLA.alt : altM);

    // Convert to cartesian ECEF coordinates
    SLVec3d defaultECEF;
    defaultECEF.lla2ecef(_defaultLLA);

    // Transform to local east-north-up frame
    _defaultENU = _wRecef * defaultECEF;
}
//-----------------------------------------------------------------------------
//! Setter that turns on the device rotation sensor
void SLDeviceLocation::isUsed (SLbool use)
{
    if (!_isUsed && use==true)
        _isFirstSensorValue = true;

    _isUsed = use;
}
//-----------------------------------------------------------------------------
//! Calculates the solar angles at origin at local time
SLbool SLDeviceLocation::calculateSolarAngles()
{
    // leave default angles if origin has not been set
    //if (!_hasOrigin) return;

    std::time_t t = std::time(nullptr);
    tm ut; memcpy(&ut, std::gmtime(&t), sizeof(tm));
    tm lt; memcpy(&lt, std::localtime(&t), sizeof(tm));

    cout << '\n';
    cout << "Universal time  : " << put_time(&ut, "%c %Z") << '\n';
    cout << "Local time      : " << put_time(&lt, "%c %Z") << '\n';
    cout << "Timezone        : " << lt.tm_hour - ut.tm_hour << '\n';

    spa_data spa;  //declare the SPA structure
    int result;
    float min, sec;

    //enter required input values into SPA structure
    spa.year            = lt.tm_year;
    spa.month           = lt.tm_mon;
    spa.day             = lt.tm_mday;
    spa.hour            = lt.tm_hour;
    spa.minute          = lt.tm_min;
    spa.second          = lt.tm_sec;
    spa.timezone        = lt.tm_hour - ut.tm_hour;
    spa.delta_ut1       = 0;
    spa.delta_t         = 0;
    spa.longitude       = _originLLA.lon;
    spa.latitude        = _originLLA.lat;
    spa.elevation       = _originLLA.alt;
    spa.pressure        =  1013.25 * pow((1.0 - 0.0065*_originLLA.alt), 5.255); // http://systemdesign.ch/wiki/Barometrische_H%C3%B6henformel
    spa.temperature     = 15.0;
    spa.slope           = 0;
    spa.azm_rotation    = 0;
    spa.atmos_refract   = 0.5667;
    spa.function        = SPA_ALL;

    /////////////////////////////
    result = spa_calculate(&spa);
    /////////////////////////////

    if (result == 0)  //check for SPA errors
    {
        _originSolarZenit = spa.zenith;
        _originSolarAzimut = spa.azimuth;

        printf("Zenith          : %.6f degrees\n", _originSolarZenit);
        printf("Azimuth         : %.6f degrees\n", _originSolarAzimut);

        min = 60.0*(spa.sunrise - (int)(spa.sunrise));
        sec = 60.0*(min - (int)min);
        printf("Sunrise         : %02d:%02d:%02d Local Time\n", (int)(spa.sunrise), (int)min, (int)sec);

        min = 60.0*(spa.sunset - (int)(spa.sunset));
        sec = 60.0*(min - (int)min);
        printf("Sunset          : %02d:%02d:%02d Local Time\n", (int)(spa.sunset), (int)min, (int)sec);
        cout << '\n';

    }
    else printf("SPA Error Code: %d\n", result);

    return (result == 0);
}
//------------------------------------------------------------------------------
