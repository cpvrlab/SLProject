//#############################################################################
//  File:      SLDeviceLocation.cpp
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAlgo.h>
#include <SLDeviceLocation.h>
#include <SLImporter.h>
#include <spa.h>

//-----------------------------------------------------------------------------
void SLDeviceLocation::init()
{
    _isUsed             = false;
    _isFirstSensorValue = false;
    _locLatLonAlt.set(0, 0, 0);
    _locECEF.set(0, 0, 0);
    _locENU.set(0, 0, 0);
    _locAccuracyM    = 0.0f;
    _locMaxDistanceM = 1000.0f;
    _defaultLatLonAlt.set(0, 0, 0);
    _defaultENU.set(0, 0, 0);
    _originLatLonAlt.set(0, 0, 0);
    _originENU.set(0, 0, 0);
    _offsetENU.set(0, 0, 0);
    _originAccuracyM    = FLT_MAX;
    _originSolarZenith  = 45.0f;
    _originSolarAzimuth = 0.0f;
    _originSolarSunrise = 0.0f;
    _originSolarSunset  = 0.0f;
    _wRecef.identity();
    _hasOrigin         = false;
    _useOriginAltitude = true;
    _improveOrigin     = true;
    _improveTimeSEC    = 10.0f;
    _sunLightNode      = nullptr;
    _altDemM           = 0.0f;
    _altGpsM           = 0.0f;
    _cameraHeightM     = 1.6f;
    _offsetMode        = LOM_none;
    _nameLocations.clear();
    _activeNamedLocation = -1;
}
//-----------------------------------------------------------------------------
// Setter for hasOrigin flag.
void SLDeviceLocation::hasOrigin(SLbool hasOrigin)
{
    if (!hasOrigin)
    {
        _improveTimer.start();
        _originAccuracyM = FLT_MAX;
    }
    _hasOrigin = hasOrigin;
}
//-----------------------------------------------------------------------------
//! Event handler for mobile device location update.
/*! Global event handler for device GPS location with longitude and latitude in
 degrees and altitude in meters. This location uses the World Geodetic System
 1984 (WGS84). The accuracy in meters is a radius in which the location is with
 a probability of 68% (2 sigma). The altitude in m is the most inaccurate
 information. The option _useOriginAltitude allows to overwrite the current
 altitude with the origins altitude. If a geoTiff is available the altitude is
 is take from it.
 /param latDEG Latitude (vertical) on WGS84 geoid in degrees
 /param lonDEG Longitude (horizontal) on WGS84 geoid in degrees
 /param altM Altitude over WGS84 geoid in meters
 /param accuracyM Accuracy in meters is a radius
*/
void SLDeviceLocation::onLocationLatLonAlt(SLdouble latDEG,
                                           SLdouble lonDEG,
                                           SLdouble altM,
                                           SLfloat  accuracyM)
{
    // Use altitude either from DEM (best), origin (static) or GPS (worst)
    _altGpsM       = (float)altM;
    float altToUse = (float)altM;
    if (geoTiffIsAvailableAndValid() && posIsOnGeoTiff(latDEG, lonDEG))
    {
        _altDemM = _demGeoTiff.getAltitudeAtLatLon(latDEG,
                                                   lonDEG);
        altToUse = _altDemM + _cameraHeightM;
    }
    else
    {
        altToUse = _useOriginAltitude ? (float)_originLatLonAlt.alt : _altGpsM;
    }

    // Init origin if it is not set yet or if the origin should be improved
    if (!_hasOrigin || _improveOrigin)
    {
        // The first sensor value can appear after a few seconds.
        if (_isFirstSensorValue)
        {
            _improveTimer.start();
            _isFirstSensorValue = false;
        }

        // Only improve if accuracy is higher and the improve time has not elapsed
        if (accuracyM < _originAccuracyM ||
            _improveTimer.elapsedTimeInSec() < _improveTimeSEC)
        {
            _originAccuracyM = accuracyM;
            originLatLonAlt(latDEG, lonDEG, altToUse);
            defaultLatLonAlt(latDEG, lonDEG, altToUse);
        }
    }

    _locLatLonAlt.set(latDEG, lonDEG, altToUse);

    _locAccuracyM = accuracyM;

    // Convert to cartesian ECEF coordinates
    _locECEF.latlonAlt2ecef(_locLatLonAlt);

    // Transform to local east-north-up frame
    _locENU = _wRecef * _locECEF;
}
//-----------------------------------------------------------------------------
//! Origin coordinate setter in WGS84 Lat-Lon in degrees, minutes and seconds
/* Swisstopo coordinates at https://map.geo.admin.ch in degrees, minutes and
 * seconds are preciser than their decimal degrees.
 */
void SLDeviceLocation::originLatLonAlt(int    degreesLat,
                                       int    minutesLat,
                                       double secondsLat,
                                       int    degreesLon,
                                       int    minutesLon,
                                       double secondsLon,
                                       double altitudeM)
{
    SLVec3d originWGS84Decimal = SLAlgo::geoDegMinSec2Decimal(degreesLat,
                                                              minutesLat,
                                                              secondsLat,
                                                              degreesLon,
                                                              minutesLon,
                                                              secondsLon,
                                                              altitudeM);
    originLatLonAlt(originWGS84Decimal.lat,
                    originWGS84Decimal.lon,
                    originWGS84Decimal.alt);
}
//-----------------------------------------------------------------------------
//! Set global origin in latitude, longitude and altitude at the ground level.
/*! The calculated values can be used for global camera positioning via GPS
 sensor. The origin is the zero point of the model. The origin should be defined
 in the model on the ground.
 /param latDEG Latitude (vertical) on WGS84 geoid in decimal degrees
 /param lonDEG Longitude (horizontal) on WGS84 geoid in decimal degrees
 /param altM Altitude over WGS84 geoid in meters
*/
void SLDeviceLocation::originLatLonAlt(SLdouble latDEG,
                                       SLdouble lonDEG,
                                       SLdouble altM)
{
    _originLatLonAlt = SLVec3d(latDEG, lonDEG, altM);
    SLVec3d originECEF;
    originECEF.latlonAlt2ecef(_originLatLonAlt);

    // calculation of ECEF to world (scene) rotation matrix
    // definition of rotation matrix for ECEF to world frame rotation:
    // world frame (scene) w.r.t. ENU frame
    double phiRad = latDEG * Utils::DEG2RAD; // phi == latitude
    double lamRad = lonDEG * Utils::DEG2RAD; // lambda == longitude
    double sinPhi = sin(phiRad);
    double cosPhi = cos(phiRad);
    double sinLam = sin(lamRad);
    double cosLam = cos(lamRad);

    SLMat3d enuRecef(-sinLam,
                     cosLam,
                     0,
                     -cosLam * sinPhi,
                     -sinLam * sinPhi,
                     cosPhi,
                     cosLam * cosPhi,
                     sinLam * cosPhi,
                     sinPhi);

    // ENU frame w.r.t. world frame (scene)
    SLMat3d wRenu; // same as before
    wRenu.rotation(-90, 1, 0, 0);

    // ECEF w.r.t. world frame (scene)
    _wRecef    = wRenu * enuRecef;
    _originENU = _wRecef * originECEF;

    // Indicate that origin is set. Otherwise it would be reset on each update
    _hasOrigin = true;

    calculateSolarAngles(_originLatLonAlt, std::time(nullptr));
}
//-----------------------------------------------------------------------------
//! Default coordinate setter in WGS84 Lat-Lon in degrees, minutes and seconds
/* Swisstopo coordinates at https://map.geo.admin.ch in degrees, minutes and
 * seconds are preciser than their decimal degrees.
 */
void SLDeviceLocation::defaultLatLonAlt(int    degreesLat,
                                        int    minutesLat,
                                        double secondsLat,
                                        int    degreesLon,
                                        int    minutesLon,
                                        double secondsLon,
                                        double altitudeM)
{
    SLVec3d defaultWGS84Decimal = SLAlgo::geoDegMinSec2Decimal(degreesLat,
                                                               minutesLat,
                                                               secondsLat,
                                                               degreesLon,
                                                               minutesLon,
                                                               secondsLon,
                                                               altitudeM);
    defaultLatLonAlt(defaultWGS84Decimal.lat,
                     defaultWGS84Decimal.lon,
                     defaultWGS84Decimal.alt);
}
//-----------------------------------------------------------------------------
//! Sets the default location in latitude, longitude and altitude.
/*! It must be called after setting the origin. If no origin is set with it
 will be automatically set in onLocationLatLonAlt. The default location is used by
 the camera in SLCamera::setView if the current distance between _locENU and
 _originENU is greater than _locMaxDistanceM. Witch means that you are in real
 not near the location.
 /param latDEG Latitude (vertical) on WGS84 geoid in degrees
 /param lonDEG Longitude (horizontal) on WGS84 geoid in degrees
 /param altM Altitude over WGS84 geoid in meters
 */
void SLDeviceLocation::defaultLatLonAlt(SLdouble latDEG,
                                        SLdouble lonDEG,
                                        SLdouble altM)
{
    _defaultLatLonAlt.set(latDEG,
                          lonDEG,
                          _useOriginAltitude ? _originLatLonAlt.alt : altM);
    _locLatLonAlt = _defaultLatLonAlt;

    // Convert to cartesian ECEF coordinates
    SLVec3d defaultECEF;
    defaultECEF.latlonAlt2ecef(_defaultLatLonAlt);

    // Transform to local east-north-up frame
    _defaultENU = _wRecef * defaultECEF;
}
//-----------------------------------------------------------------------------
//! Setter that turns on the device rotation sensor
void SLDeviceLocation::isUsed(SLbool use)
{
    if (!_isUsed && use)
        _isFirstSensorValue = true;

    _isUsed = use;
}
//-----------------------------------------------------------------------------
//! Calculates the solar angles at origin at local time
/*! Calculates the zenith and azimuth angle in deg. of the sun at the origin at
 the local time using the Solar Position Algorithm from:
 https://midcdmz.nrel.gov/spa/ that is part of libsl_external.
*/
SLbool SLDeviceLocation::calculateSolarAngles(SLVec3d     locationLatLonAlt,
                                              std::time_t time)
{
    // leave default angles if origin has not been set
    if (!_hasOrigin) return false;

    // transform time
    tm ut{}, lt{};
    memcpy(&ut, std::gmtime(&time), sizeof(tm));
    memcpy(&lt, std::localtime(&time), sizeof(tm));

    ut.tm_year += 1900;
    lt.tm_year += 1900;
    ut.tm_mon++;
    lt.tm_mon++;

    SL_LOG("Universal time  : %02d.%02d.%02d %02d:%02d:%02d",
           ut.tm_mday,
           ut.tm_mon,
           ut.tm_year,
           ut.tm_hour,
           ut.tm_min,
           ut.tm_sec);
    SL_LOG("Local time      : %02d.%02d.%02d %02d:%02d:%02d",
           lt.tm_mday,
           lt.tm_mon,
           lt.tm_year,
           lt.tm_hour,
           lt.tm_min,
           lt.tm_sec);
    SL_LOG("Timezone        : %d", lt.tm_hour - ut.tm_hour);

    spa_data spa; // declare the SPA structure
    SLint    result;

    // enter required input values into SPA structure
    spa.year      = lt.tm_year;
    spa.month     = lt.tm_mon;
    spa.day       = lt.tm_mday;
    spa.hour      = lt.tm_hour;
    spa.minute    = lt.tm_min;
    spa.second    = lt.tm_sec;
    spa.timezone  = lt.tm_hour - ut.tm_hour;
    spa.delta_ut1 = 0;
    spa.delta_t   = 0;
    spa.longitude = locationLatLonAlt.lon;
    spa.latitude  = locationLatLonAlt.lat;
    spa.elevation = locationLatLonAlt.alt;
    // http://systemdesign.ch/wiki/Barometrische_Hoehenformel
    spa.pressure      = 1013.25 * pow((1.0 - 0.0065 * locationLatLonAlt.alt / 288.15), 5.255);
    spa.temperature   = 15.0;
    spa.slope         = 0;
    spa.azm_rotation  = 0;
    spa.atmos_refract = 0.5667;
    spa.function      = SPA_ALL;

    /////////////////////////////
    result = spa_calculate(&spa);
    /////////////////////////////

    if (result == 0) // check for SPA errors
    {
        _originSolarZenith  = (SLfloat)spa.zenith;
        _originSolarAzimuth = (SLfloat)spa.azimuth;
        _originSolarSunrise = (SLfloat)spa.sunrise;
        _originSolarSunset  = (SLfloat)spa.sunset;

        SLfloat SRh = _originSolarSunrise;
        SLfloat SRm = (SLfloat)(60.0f * (SRh - (int)(SRh)));
        SLfloat SRs = (SLfloat)(60.0 * (SRm - floor(SRm)));
        SLfloat SSh = _originSolarSunset;
        SLfloat SSm = (SLfloat)(60.0f * (SSh - (int)(SSh)));
        SLfloat SSs = (SLfloat)(60.0f * (SSm - floor(SSm)));

        SL_LOG("Zenith          : %.6f degrees", _originSolarZenith);
        SL_LOG("Azimuth         : %.6f degrees", _originSolarAzimuth);
        SL_LOG("Sunrise         : %02d:%02d:%02d Local Time", (int)(SRh), (int)SRm, (int)SRs);
        SL_LOG("Sunset          : %02d:%02d:%02d Local Time", (int)(SSh), (int)SSm, (int)SSs);
    }
    else
        SL_LOG("SPA Error Code: %d", result);

    if (_sunLightNode)
    {
        // The azimuth is from north eastwards
        _sunLightNode->rotation(180.0f - _originSolarAzimuth, SLVec3f::AXISY);

        // The zenith angle is from up downwards
        _sunLightNode->rotate(90.0f - _originSolarZenith, -SLVec3f::AXISX);
    }

    return (result == 0);
}
//------------------------------------------------------------------------------
//! Converter method: the transferred wgs84 coordinate is converted to ENU frame and returned (does not change SLDeviceLocation)
SLVec3d SLDeviceLocation::convertLatLonAlt2ENU(SLVec3d locLatLonAlt) const
{
    if (geoTiffIsAvailableAndValid() && posIsOnGeoTiff(locLatLonAlt.x, locLatLonAlt.y))
        locLatLonAlt.z = _demGeoTiff.getAltitudeAtLatLon(locLatLonAlt.x, locLatLonAlt.y);

    // Convert to cartesian ECEF coordinates
    SLVec3d locECEF;
    locECEF.latlonAlt2ecef(locLatLonAlt);

    // Transform to local east-north-up frame
    SLVec3d locENU = _wRecef * locECEF;

    return locENU;
}
//------------------------------------------------------------------------------
//! Loads a GeoTiff DEM (Digital Elevation Model) Image
/* Loads a GeoTiff DEM (Digital Elevation Model) Image that must be in WGS84
 coordinates. For more info see CVImageGeoTiff.
 If the 32-bit image file and its JSON info file gets successfully loaded,
 we can set the altitudes from the _originLatLonAlt and _defaultLatLonAlt by the DEM.
 */
void SLDeviceLocation::loadGeoTiff(const SLstring& geoTiffFile)
{
    try
    {
        assert(!_defaultLatLonAlt.isZero() &&
               !_originLatLonAlt.isZero() &&
               "Set first defaultLatLonAlt and originLatLonAlt before you add a GeoTiff.");

        _demGeoTiff.loadGeoTiff(geoTiffFile);

        // Check that default and origin location is withing the GeoTiff extends
        if (geoTiffIsAvailableAndValid())
        {
            // Overwrite the altitudes of origin
            SLfloat altOriginM = _demGeoTiff.getAltitudeAtLatLon(_originLatLonAlt.lat,
                                                                 _originLatLonAlt.lon);
            originLatLonAlt(_originLatLonAlt.lat,
                            _originLatLonAlt.lon,
                            altOriginM);

            // Overwrite the altitudes of default with the additional camera height
            SLfloat altDefaultM = _demGeoTiff.getAltitudeAtLatLon(_defaultLatLonAlt.lat,
                                                                  _defaultLatLonAlt.lon);
            defaultLatLonAlt(_defaultLatLonAlt.lat,
                             _defaultLatLonAlt.lon,
                             altDefaultM + _cameraHeightM);
        }
        else
        {
            string msg = "SLDeviceLocation::loadGeoTiff: Either the geotiff file ";
            msg += "could not be loaded or the origin or default position lies ";
            msg += "not within the extends of the geotiff file.";
            throw std::runtime_error(msg.c_str());
        }
    }
    catch (std::exception& e)
    {
        SL_WARN_MSG(e.what());
    }
    catch (...)
    {
        SL_WARN_MSG("SLDeviceLocation::loadGeoTiff: Unknown exception catched.");
    }
}
//-----------------------------------------------------------------------------
/* Returns true if a geoTiff files is loaded and the origin and default
 positions are within the extends of the image.
*/
bool SLDeviceLocation::geoTiffIsAvailableAndValid() const
{
    return (!_demGeoTiff.cvMat().empty() &&
            _originLatLonAlt.lat < _demGeoTiff.upperLeftLatLonAlt()[0] &&
            _originLatLonAlt.lat > _demGeoTiff.lowerRightLatLonAlt()[0] &&
            _originLatLonAlt.lon > _demGeoTiff.upperLeftLatLonAlt()[1] &&
            _originLatLonAlt.lon < _demGeoTiff.lowerRightLatLonAlt()[1] &&
            _defaultLatLonAlt.lat < _demGeoTiff.upperLeftLatLonAlt()[0] &&
            _defaultLatLonAlt.lat > _demGeoTiff.lowerRightLatLonAlt()[0] &&
            _defaultLatLonAlt.lon > _demGeoTiff.upperLeftLatLonAlt()[1] &&
            _defaultLatLonAlt.lon < _demGeoTiff.lowerRightLatLonAlt()[1]);
}
//-----------------------------------------------------------------------------
//! Return true if the current GPS location is within the GeoTiff boundaries
bool SLDeviceLocation::posIsOnGeoTiff(SLdouble latDEG, SLdouble lonDEG) const
{
    return (!_demGeoTiff.empty() &&
            latDEG < _demGeoTiff.upperLeftLatLonAlt()[0] &&
            latDEG > _demGeoTiff.lowerRightLatLonAlt()[0] &&
            lonDEG > _demGeoTiff.upperLeftLatLonAlt()[1] &&
            lonDEG < _demGeoTiff.lowerRightLatLonAlt()[1]);
}
//------------------------------------------------------------------------------
//! Returns the device location offset mode as string
SLstring SLDeviceLocation::offsetModeStr() const
{
    switch (_offsetMode)
    {
        case LOM_none: return "None";
        case LOM_twoFingerY: return "TwoFingerY";
        default: return "Unknown";
    }
}
//-----------------------------------------------------------------------------
