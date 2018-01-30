//#############################################################################
//  File:      SLApplication.h
//  Author:    Marcus Hudritsch
//  Date:      Februar 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#ifdef SL_MEMLEAKDETECT     // set in SL.h for debug config only
#include <debug_new.h>      // memory leak detector
#endif

#include <SLApplication.h>
#include <SLScene.h>
#include <SLCVCapture.h>
#include <SLCVTracked.h>
#include <SLCVTrackedAruco.h>

//-----------------------------------------------------------------------------
//! Global static objects
//-----------------------------------------------------------------------------
//! static instance variable declaration
SLInputManager      SLApplication::inputManager;
SLScene*            SLApplication::scene        = nullptr;
SLCVCalibration*    SLApplication::activeCalib  = nullptr;
SLCVCalibration     SLApplication::calibMainCam;
SLCVCalibration     SLApplication::calibScndCam;
SLDeviceRotation    SLApplication::devRot;
SLDeviceLocation    SLApplication::devLoc;
//-----------------------------------------------------------------------------
void SLApplication::createAppAndScene(SLstring name)
{
    assert(SLApplication::scene == nullptr &&
           "You can create only one SLApplication");
    
    scene = new SLScene(name);
    
    // load opencv camera calibration for main and secondary camera
    #if defined(SL_USES_CVCAPTURE)
    calibMainCam.load("cam_calibration_main.xml", true, false);
    calibMainCam.loadCalibParams();
    activeCalib = &calibMainCam;
    SLCVCapture::hasSecondaryCamera = false;
    #else
    calibMainCam.load("cam_calibration_main.xml", false, false);
    calibMainCam.loadCalibParams();
    calibScndCam.load("cam_calibration_scnd.xml", true, false);
    calibScndCam.loadCalibParams();
    activeCalib = &calibMainCam;
    SLCVCapture::hasSecondaryCamera = true;
    #endif
}
//-----------------------------------------------------------------------------
void SLApplication::deleteAppAndScene()
{
    assert(SLApplication::scene != nullptr &&
           "You can delete an  only once");
    
    if (scene)
        delete scene;
    
    scene = nullptr;
}
//-----------------------------------------------------------------------------
