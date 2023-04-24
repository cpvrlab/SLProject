//#############################################################################
//  File:      SLDeviceLocation.h
//  Purpose:   Mobile device location class declaration
//  Authors:   Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLDEVICELOCATION_H
#define SLDEVICELOCATION_H

#include <SLLightDirect.h>
#include <SLAlgo.h>
#include <HighResTimer.h>
#include <CVImageGeoTiff.h>

class SLNode;

//-----------------------------------------------------------------------------
//! class for a named location with lat-lon-Alt WGS84 position
class SLLocation
{
public:
    SLLocation(string locName,
               int    degreesLat,
               int    minutesLat,
               double secondsLat,
               int    degreesLon,
               int    minutesLon,
               double secondsLon,
               double altitudeM)
    {
        name             = std::move(locName);
        SLVec3d posWGS84 = SLAlgo::geoDegMinSec2Decimal(degreesLat,
                                                        minutesLat,
                                                        secondsLat,
                                                        degreesLon,
                                                        minutesLon,
                                                        secondsLon,
                                                        altitudeM);
        posWGS84LatLonAlt.set(posWGS84.lat,
                              posWGS84.lon,
                              posWGS84.alt);
    }

    string  name;
    SLVec3d posWGS84LatLonAlt;
};
typedef vector<SLLocation> SLVLocation;
//-----------------------------------------------------------------------------
//! Device location offset mode
enum SLLocOffsetMode
{
    LOM_none = 0,
    LOM_twoFingerY
};
//-----------------------------------------------------------------------------
//! Encapsulation of a mobile device location set by the device's GPS sensor
/*! This class is only used if SLProject runs on a mobile device. Check out the
 app-Demo-SLProject/android and app_demo_slproject/ios how the sensor data is generated
 and passed to this object hold by AppDemo. The class stores the devices location
 that it gets from its GPS (global positioning system) sensor. The device location can
 be used in the active camera to apply it to the scene camera
 (s. SLCamera::setView).\n
  - LatLonAlt: The device location from GPS comes as a latitude (deg. north-south),
  longitude (deg. east-west) and altitude (m) LatLonAlt-tripple.
  These two angles are a position and height on the WGS84 ellipsoid
  (World Geodetic System 1984).\n
  - ECEF (Earth Centered Earth Fixed) are right-handed cartesian world
 coordinates with the z-axis at the north pole the x-axis at the prime meridian
 (0 deg. longitude) and the y-axis at 90 deg. longitude. x- and y-axis form the
 equator plane.\n
  - ENU (East North Up) is the local frame (= right-handed coordinate system)
 on the surface of the ellipsoid with the E=East tangential vector, the N=North
 tangential vector and U=Up as the ellipsoid's normal vector. Be aware that the
 up vector is normal to the ellipsoid and not to the terrain above. This normal
 does not point the center of the ellipsoid.\n
 If we want to show a local scene on the earth, we do this always in the ENU
 frame because in the ECEF frame with have not enough precision for float
 coordinates. Therefore we have define a local origin in the ENU frame and
 convert all locations from LatLonAlt to ECEF and the with the wRecef rotation
 matrix to the ENU frame.
*/
class SLDeviceLocation
{
public:
    SLDeviceLocation() { init(); }
    void init();
    void onLocationLatLonAlt(SLdouble latDEG,
                             SLdouble lonDEG,
                             SLdouble altM,
                             SLfloat  accuracyM);

    SLbool calculateSolarAngles(SLVec3d     locationLatLonAlt,
                                std::time_t time);

    //! Converter method: the transferred wgs84 coordinate is converted to ENU frame and returned (does not change SLDeviceLocation)
    SLVec3d convertLatLonAlt2ENU(SLVec3d locLatLonAlt) const;

    // Setters
    void isUsed(SLbool isUsed);
    void useOriginAltitude(SLbool useGLA) { _useOriginAltitude = useGLA; }
    void improveOrigin(SLbool impO) { _improveOrigin = impO; }
    void hasOrigin(SLbool hasOL);
    void originLatLonAlt(int    degreesLat,
                         int    minutesLat,
                         double secondsLat,
                         int    degreesLon,
                         int    minutesLon,
                         double secondsLon,
                         double altitudeM);
    void originLatLonAlt(SLdouble latDEG,
                         SLdouble lonDEG,
                         SLdouble altM);
    void originLatLonAlt(SLVec3d lla) { originLatLonAlt(lla.lat, lla.lon, lla.alt); }
    void defaultLatLonAlt(int    degreesLat,
                          int    minutesLat,
                          double secondsLat,
                          int    degreesLon,
                          int    minutesLon,
                          double secondsLon,
                          double altitudeM);
    void defaultLatLonAlt(SLdouble latDEG,
                          SLdouble lonDEG,
                          SLdouble altM);
    void defaultLatLonAlt(SLVec3d lla) { defaultLatLonAlt(lla.lat, lla.lon, lla.alt); }
    void locMaxDistanceM(SLfloat maxDist) { _locMaxDistanceM = maxDist; }
    void sunLightNode(SLLightDirect* sln) { _sunLightNode = sln; }
    void loadGeoTiff(const SLstring& geoTiffFile);
    bool geoTiffIsAvailableAndValid() const;
    bool posIsOnGeoTiff(SLdouble latDEG, SLdouble lonDEG) const;
    void cameraHeightM(float camHeightM) { _cameraHeightM = camHeightM; }
    void offsetMode(SLLocOffsetMode lom) { _offsetMode = lom; }
    void offsetENU(SLVec3d offsetENU) { _offsetENU = offsetENU; }
    void activeNamedLocation(SLint locIndex)
    {
        _activeNamedLocation = locIndex;
        _isUsed              = false;
        defaultLatLonAlt(_nameLocations[locIndex].posWGS84LatLonAlt);
    }

    // Getters
    SLbool          isUsed() const { return _isUsed; }
    SLVec3d         locLatLonAlt() const { return _locLatLonAlt; }
    SLVec3d         locECEF() const { return _locECEF; }
    SLVec3d         locENU() const { return _locENU; }
    SLfloat         locAccuracyM() const { return _locAccuracyM; }
    SLfloat         locMaxDistanceM() const { return _locMaxDistanceM; }
    SLVec3d         defaultENU() const { return _defaultENU; }
    SLVec3d         originLatLonAlt() const { return _originLatLonAlt; }
    SLVec3d         defaultLatLonAlt() const { return _defaultLatLonAlt; }
    SLVec3d         originENU() const { return _originENU; }
    SLVec3d         offsetENU() const { return _offsetENU; }
    SLbool          hasOrigin() const { return _hasOrigin; }
    SLbool          useOriginAltitude() const { return _useOriginAltitude; }
    SLMat3d         wRecef() const { return _wRecef; }
    SLfloat         improveTime() { return std::max(_improveTimeSEC - _improveTimer.elapsedTimeInSec(), 0.0f); }
    SLfloat         originSolarZenit() const { return _originSolarZenith; }
    SLfloat         originSolarAzimut() const { return _originSolarAzimuth; }
    SLfloat         originSolarSunrise() const { return _originSolarSunrise; }
    SLfloat         originSolarSunset() const { return _originSolarSunset; }
    SLNode*         sunLightNode() const { return _sunLightNode; }
    SLfloat         altDemM() const { return _altDemM; }
    SLfloat         altGpsM() const { return _altGpsM; }
    SLfloat         cameraHeightM() const { return _cameraHeightM; };
    SLLocOffsetMode offsetMode() { return _offsetMode; }
    SLstring        offsetModeStr() const;
    SLVLocation&    nameLocations() { return _nameLocations; }
    SLint           activeNamedLocation() const { return _activeNamedLocation; }

private:
    SLbool          _isUsed;              //!< Flag if the devices GPS Sensor is used
    SLbool          _isFirstSensorValue;  //!< Flag for the first sensor values
    SLVec3d         _locLatLonAlt;        //!< Earth location in latitudeDEG, longitudeDEG & AltitudeM on WGS84 geoid
    SLVec3d         _locECEF;             //!< Cartesian location in ECEF
    SLVec3d         _locENU;              //!< Cartesian location in ENU frame
    SLfloat         _locAccuracyM;        //!< Horizontal accuracy radius in m with 68% probability
    SLfloat         _locMaxDistanceM;     //!< Max. allowed distance from origin. If higher it is ignored.
    SLVec3d         _defaultLatLonAlt;    //!< Default location of scene in LatLonAlt.
    SLVec3d         _defaultENU;          //!< Default location in ENU frame used if real location is too far away from origin
    SLVec3d         _originLatLonAlt;     //!< Global origin location of scene in LatLonAlt
    SLVec3d         _originENU;           //!< Origin location in ENU frame
    SLVec3d         _offsetENU;           //!< Offset vector in ENU frame
    SLfloat         _originAccuracyM;     //!< Accuracy radius of origin point
    SLfloat         _originSolarZenith;   //!< Zenith angle of the sun in deg. (from up dir.) at origin at local time
    SLfloat         _originSolarAzimuth;  //!< Azimuth angle of the sun in deg. (eastward from north) at origin at local time
    SLfloat         _originSolarSunrise;  //!< Sunrise local time at origin
    SLfloat         _originSolarSunset;   //!< Sunset local time at origin
    SLbool          _hasOrigin;           //!< Flag if this scene has a global reference location
    SLbool          _useOriginAltitude;   //!< Flag if global reference altitude should be used
    SLfloat         _altDemM;             //!< Altitude in m from Digital Elevation Model
    SLfloat         _altGpsM;             //!< Altitude in m from GPS
    SLfloat         _cameraHeightM;       //!< Height from ground to the mobile camera in m
    SLbool          _improveOrigin;       //!< Flag if origin should be improved over time & accuracy
    SLfloat         _improveTimeSEC;      //!< Max. time in seconds for the origin improvement.
    HighResTimer    _improveTimer;        //!< Timer to measure the improve time.
    SLMat3d         _wRecef;              //!< ECEF frame to world frame rotation: rotates a point defined in ecef
    SLNode*         _sunLightNode;        //!< Pointer to directional light node to be changed if solar angles are calculated
    CVImageGeoTiff  _demGeoTiff;          //!< Digital Elevation Model from a Geo Tiff image
    SLLocOffsetMode _offsetMode;          //!< Location offset mode
    SLVLocation     _nameLocations;       //!< Vector of fix locations for default view points
    SLint           _activeNamedLocation; //!< Index of the active named location as defaultENU;
};
//-----------------------------------------------------------------------------
#endif
