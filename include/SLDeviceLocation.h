//#############################################################################
//  File:      SLDeviceLocation.h
//  Purpose:   Mobile device location class declaration
//  Author:    Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLDEVICELOCATION_H
#define SLDEVICELOCATION_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLLightDirect.h>

//-----------------------------------------------------------------------------
//! Encapsulation of a mobile device location set by the device's GPS sensor
/*! This class is only used if SLProject runs on a mobile device. Check out the
app-Demo-Android and app-Demo-iOS how the sensor data is generated and passed
to this object hold by SLScene. The class stores the devices location that it
gets from its GPS (global positioning system) sensor. The device location can
be used in the active camera to apply it to the scene camera
(s. SLCamera::setView).\n
  - LLA: The device location from GPS comes as a latitude (deg.), longitude
(deg.) and altitude (m) LLA-tripple. These two angles are a position and height
on the WGS84 ellipsoid (World Geodetic System 1984).\n
  - ECEF (Earth Centered Earth Fixed) are right-handed cartesian world
coordinates with the z-axis at the north pole the x-axis at the prime meridian
(0 deg. longitude) and the y-axis at 90 deg. longitude. x- and y-axis form the
 equator plane.\n
  - ENU (East North Up) is the local frame (= right-handed coordinate system)
on the surface of the ellipsoid with the E=East tangential vector, the N=North
tangential vector and U=Up as the ellipsoid's normal vector. Be aware that the
up vector is normal to the ellipsoid and not to the terrain above. This normal
does not point the center of the ellipsoid.\n
If we want to show a local scene on the earth, we do this allways in the ENU
frame because in the ECEF frame with have not enough precision for float
coordinates. Therefore we have define a local origin in the ENU frame and
convert all locations from LLA to ECEF and the with the wRecef rotation matrix
to the ENU frame.
*/
class SLDeviceLocation
{
    public:             SLDeviceLocation    (){init();}
            void        init                ();
            void        onLocationLLA       (SLdouble latDEG,
                                             SLdouble lonDEG,
                                             SLdouble altM,
                                             SLfloat AccuracyM);

            SLbool      calculateSolarAngles(SLdouble latDEG,
                                             SLdouble lonDEG,
                                             SLdouble altM);

            // Setters
            void        isUsed              (SLbool isUsed);
            void        useOriginAltitude   (SLbool useGLA) {_useOriginAltitude = useGLA;}
            void        improveOrigin       (SLbool impO) {_improveOrigin = impO;}
            void        hasOrigin           (SLbool hasOL);
            void        originLLA           (SLdouble latDEG,
                                             SLdouble lonDEG,
                                             SLdouble altM);
            void        defaultLLA          (SLdouble latDEG,
                                             SLdouble lonDEG,
                                             SLdouble altM);
            void        locMaxDistanceM     (SLfloat maxDist) {_locMaxDistanceM = maxDist;}
            void        sunLightNode        (SLLightDirect* sln) {_sunLightNode = sln;}

            // Getters
            SLbool      isUsed              () const {return _isUsed;}
            SLVec3d     locLLA              () const {return _locLLA;}
            SLVec3d     locECEF             () const {return _locECEF;}
            SLVec3d     locENU              () const {return _locENU;}
            SLfloat     locAccuracyM        () const {return _locAccuracyM;}
            SLfloat     locMaxDistanceM     () const {return _locMaxDistanceM;}
            SLVec3d     defaultENU          () const {return _defaultENU;}
            SLVec3d     originLLA           () const {return _originLLA;}
            SLVec3d     originENU           () const {return _originENU;}
            SLVec3d     originECEF          () const {return _originECEF;}
            SLbool      hasOrigin           () const {return _hasOrigin;}
            SLbool      useOriginAltitude   () const {return _useOriginAltitude;}
            SLMat3d     wRecef              () const {return _wRecef;}
            SLfloat     improveTime         () {return SL_max(_improveTimeSEC - _improveTimer.elapsedTimeInSec(), 0.0f);}
            SLfloat     originSolarZenit    () const {return _originSolarZenit;}
            SLfloat     originSolarAzimut   () const {return _originSolarAzimut;}

   private:
            SLbool      _isUsed;            //!< Flag if the devices GPS Sensor is used
            SLbool      _isFirstSensorValue;//!< Flag for the first sensor values
            SLVec3d     _locLLA;            //!< Earth location in latitudeDEG, longitudeDEG & AltitudeM on WGS84 geoid
            SLVec3d     _locECEF;           //!< Cartesian location in ECEF
            SLVec3d     _locENU;            //!< Cartesian location in ENU frame
            SLfloat     _locAccuracyM;      //!< Horizontal accuracy radius in m with 68% probability
            SLfloat     _locMaxDistanceM;   //!< Max. allowed distance from origin. If higher it is ignored.
            SLVec3d     _defaultLLA;        //!< Default location of scene in LLA.
            SLVec3d     _defaultENU;        //!< Default location in ENU frame used if real location is too far away from origin
            SLVec3d     _originLLA;         //!< Global origin location of scene in LLA
            SLVec3d     _originECEF;        //!< Global origin location of scene in ECEF (cartesian)
            SLVec3d     _originENU;         //!< Origin location in ENU frame
            SLfloat     _originAccuracyM;   //!< Accuracy radius of origin point
            SLfloat     _originSolarZenit;  //!< Zenit angle of the sun in deg. (from up dir.) at origin at local time
            SLfloat     _originSolarAzimut; //!< Azimut angle of the sun in deg. (eastward from north) at origin at local time
            SLbool      _hasOrigin;         //!< Flag if this scene has a global reference location
            SLbool      _useOriginAltitude; //!< Flag if global reference altitude should be used
            SLbool      _improveOrigin;     //!< Flag if origin should be improved over time & accuracy
            SLfloat     _improveTimeSEC;    //!< Max. time in seconds for the origin improvement.
            SLTimer     _improveTimer;      //!< Timer to measure the improve time.
            SLMat3d     _wRecef;            //!< ECEF frame to world frame rotation: rotates a point defined in ecef
            SLNode*     _sunLightNode;      //!< Pointer to directional light node to be changed if solar angles are calculated
};
//-----------------------------------------------------------------------------
#endif
