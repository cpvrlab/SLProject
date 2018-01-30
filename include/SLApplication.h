//#############################################################################
//  File:      SLApplication.h
//  Author:    Marcus Hudritsch
//  Date:      Februar 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLAPPLICATION_H
#define SLAPPLICATION_H

#include <stdafx.h>
#include <SLInputManager.h>
#include <SLCVCalibration.h>
#include <SLDeviceRotation.h>
#include <SLDeviceLocation.h>

class SLScene;
//-----------------------------------------------------------------------------
//! Top level class for an SLProject application.
/*!      
The SLApplication holds static instances of top-level items such as the scene
pointer, the camera calibration objects and the device rotation and location
information. The static function createAppAndScene is called by the C-interface
functions slCreateScene and slTerminate.
*/
class SLApplication: public SLObject
{
    public:
    static void             createAppAndScene   (SLstring name);
    static void             deleteAppAndScene   ();
    
    static SLScene*         scene;          //!< scene pointer
    static SLInputManager   inputManager;   //!< Input events manager
    static SLCVCalibration* activeCalib;    //!< Pointer to the active calibration
    static SLCVCalibration  calibMainCam;   //!< OpenCV calibration for main video camera
    static SLCVCalibration  calibScndCam;   //!< OpenCV calibration for secondary video camera
    static SLDeviceRotation devRot;         //!< Mobile device rotation from IMU
    static SLDeviceLocation devLoc;         //!< Mobile device location from GPS
};
//-----------------------------------------------------------------------------
#endif
