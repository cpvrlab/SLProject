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

//-----------------------------------------------------------------------------
//! Encapsulation of a mobile device location set by the device's GPS sensor
/*! This class is only used if SLProject runs on a mobile device. Check out the
app-Demo-Android and app-Demo-iOS how the sensor data is generated and passed
to this object hold by SLScene. The class stores the devices location that it
gets from its GPS (global positioning system) sensor. The device location can
be used in the active camera to apply it to the scene camera
(s. SLCamera::setView).\n
The device location from GPS comes as a latitude (deg.), longitude (deg.) and
altitude (m) tripple. These two angles are a position and height on the WGS84
ellipsoid. Please check the following terms on Wikipedia:\n
- WGS84 (World Geodetic System 1984)
- ECEF (Earth Centered Earth Fixed)
- ENU (East North Up)
*/
class SLDeviceLocation
{
    public:             SLDeviceLocation    (){init();}
            void        init                ();
            void        onLocationLLA       (double latitudeDEG,
                                             double longitudeDEG,
                                             double altitudeM,
                                             float accuracyM);
            // Setters
            void        isUsed              (SLbool isUsed) {_isUsed = isUsed;}
            void        useOriginAltitude   (SLbool useGLA) {_useOriginAltitude = useGLA;}
            void        hasOrigin           (SLbool hasOL) {_hasOrigin = hasOL;}
            void        originLLA           (double latDeg,
                                             double lonDeg,
                                             double altM);
            // Getters
            SLbool      isUsed              () const {return _isUsed;}
            SLVec3d     locLLA              () const {return _locLLA;}
            SLVec3d     locECEF             () const {return _locECEF;}
            SLVec3d     locENU              () const {return _locENU;}
            SLfloat     accuracyM           () const {return _accuracyM;}
            SLVec3d     originLLA           () const {return _originLLA;}
            SLVec3d     originENU           () const {return _originENU;}
            SLVec3d     originECEF          () const {return _originECEF;}
            SLbool      hasOrigin           () const {return _hasOrigin;}
            SLbool      useOriginAltitude   () const {return _useOriginAltitude;}
            SLMat3d     wRecef              () const {return _wRecef;}

   private:
            SLbool      _isUsed;            //!< Flag if the devices GPS Sensor is used
            SLbool      _deviceLocStarted;  //!< Flag for the first sensor values
            SLVec3d     _locLLA;            //!< GPS location in latitudeDEG, longitudeDEG & AltitudeM
            SLVec3d     _locECEF;           //!< Cartesian location in ECEF
            SLVec3d     _locENU;            //!< Cartesian location in ENU frame
            SLfloat     _accuracyM;         //!< Horizontal accuracy radius in m with 68% probability
            SLVec3d     _originLLA;         //!< Global origin location of scene in LLA
            SLVec3d     _originECEF;        //!< Global origin location of scene in ECEF (cartesian)
            SLVec3d     _originENU;         //!< Origin location in ENU frame
            SLbool      _hasOrigin;         //!< Flag if this scene has a global reference location
            SLbool      _useOriginAltitude; //!< Flag if global reference altitude should be used
            SLMat3d     _wRecef;            //!< ECEF frame to world frame rotation: rotates a point defined in ecef
};
//-----------------------------------------------------------------------------
#endif
