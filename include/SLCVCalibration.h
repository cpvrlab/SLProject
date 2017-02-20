//#############################################################################
//  File:      SLCVCalibration.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCALIBRATION_H
#define SLCVCALIBRATION_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/

#include <stdafx.h>
#include <SLCV.h>
using namespace std;

//-----------------------------------------------------------------------------
//! Live video camera calibration class with OpenCV an OpenCV calibration.
/*! For the calibration internals see the OpenCV documentation:
http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
After a successufull calibration the parameters are stored in a config file on
the SLCVCalibration::defaultPath. If it exists, it is loaded from there at
startup. If doesn't exist a calibration can be done with the calibration scene 
(Load Scene > Using Video > Calibrate Camera).\n
\n
The different calibration states are handled within SLScene::onUpdate:
\n
\nCS_uncalibrated:     The camera is not calibrated (no or invalid calibration found)
\nCS_calibrateStream:  The calibration is running with live video stream
\nCS_calibrateGrab:    The calibration is running and an image should be grabbed
\nCS_startCalculating: The calibration starts during the next frame
\nCS_calibrated:       The camera is calibrated
\nCS_estimate:         The camera intrinsics are set from an estimated FOV angle
\n
A simple calibration can be approximated from standard field of view angle.
For a good calibration we have to make 15-20 images from a chessboard pattern.
The chessboard pattern can be printed from the CalibrationChessboard_8x5_A4.pdf
in the folder _data/calibration. It is important that one side has an odd number
of inner corners. Like this it is unambiguous and can be rotated in any direction.
\n
The SLScene instance has two video camera calibrations, one for a main camera
(SLScene::_calibMainCam) and one for the selfie camera on mobile devices
(SLScene::_calibScndCam). The member SLScene::_activeCalib references the active
one and is set by the SLScene::videoType (VT_NONE, VT_MAIN, VT_SCND) during the
scene assembly in SLScene::onLoad.
*/
class SLCVCalibration
{
public:
                    SLCVCalibration     ();

    bool            load                (SLstring calibFileName, SLbool mirror);
    bool            loadCalibParams     ();
    void            setCalibrationState ();
    bool            calculate           ();
    void            clear               ();
    bool            findChessboard      (SLCVMat imageColor,
                                         SLCVMat imageGray,
                                         bool drawCorners = true);
    void            estimate            (SLint imageWidthPX, SLint imageHeightPX);

    static SLstring calibIniPath;       //!< calibration init parameters file path
    static void     calcBoardCorners3D  (SLCVSize boardSize, 
                                         SLfloat squareSize, 
                                         SLCVVPoint3f& objectPoints3D);
    // Setters
    void            state               (SLCVCalibState s) {_state = s;}
    void            isMirrored          (SLbool iM) {_isMirrored = iM;}
    void            showUndistorted     (SLbool sU) {_showUndistorted = sU;}

    // Getters
    SLCVSize        imageSize           () {return _imageSize;}
    SLfloat         imageAspectRatio    () {return (float)_imageSize.width/(float)_imageSize.height;}
    SLCVMat&        intrinsics          () {return _intrinsics;}
    SLCVMat&        distortion          () {return _distortion;}
    SLfloat         cameraFovDeg        () {return _cameraFovDeg;}
    SLbool          isMirrored          () {return _isMirrored;}
    SLfloat         fx                  () {return _intrinsics.cols==3 && _intrinsics.rows==3 ? (SLfloat)_intrinsics.at<double>(0,0) : 0.0f;}
    SLfloat         fy                  () {return _intrinsics.cols==3 && _intrinsics.rows==3 ? (SLfloat)_intrinsics.at<double>(1,1) : 0.0f;}
    SLfloat         cx                  () {return _intrinsics.cols==3 && _intrinsics.rows==3 ? (SLfloat)_intrinsics.at<double>(0,2) : 0.0f;}
    SLfloat         cy                  () {return _intrinsics.cols==3 && _intrinsics.rows==3 ? (SLfloat)_intrinsics.at<double>(1,2) : 0.0f;}
    SLfloat         k1                  () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(0,0) : 0.0f;}
    SLfloat         k2                  () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(1,0) : 0.0f;}
    SLfloat         p1                  () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(2,0) : 0.0f;}
    SLfloat         p2                  () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(3,0) : 0.0f;}
    SLCVCalibState  state               () {return _state;}
    SLint           numImgsToCapture    () {return _numOfImgsToCapture;}
    SLint           numCapturedImgs     () {return _numCaptured;}
    SLfloat         reprojectionError   () {return _reprojectionError;}
    SLbool          showUndistorted     () {return _showUndistorted;}
    SLCVSize        boardSize           () {return _boardSize;}
    SLfloat         boardSquareMM       () {return _boardSquareMM;}
    SLfloat         boardSquareM        () {return _boardSquareMM * 0.001f;}
    SLstring        calibrationTime     () {return _calibrationTime;}
    SLstring        stateStr            ()
    {   switch(_state)
        {   case CS_uncalibrated:       return "CS_uncalibrated";
            case CS_calibrated:         return "CS_calibrated";
            case CS_estimated:          return "CS_estimated";
            case CS_calibrateStream:    return "CS_calibrateStream";
            case CS_calibrateGrab:      return "CS_calibrateGrab";
            case CS_startCalculating:   return "CS_startCalculating";
            default:                    return "unknown";
        }
    }

private:
    void            calcCameraFOV       ();

    //////////////////////////////////////////////////////////////////////////////////
    SLCVMat         _intrinsics;            //!< 3x3 Matrix with intrisic camera paramters           
    SLCVMat         _distortion;            //!< 5x1 Matrix with distortion parameters
    //////////////////////////////////////////////////////////////////////////////////

    SLfloat         _cameraFovDeg;          //!< Vertical field of view in degrees
    SLCVCalibState  _state;                 //!< calibration state enumeration 
    SLstring        _calibFileName;         //!< name for calibration file
    SLstring        _calibParamsFileName;   //!< name of calibration paramters file
    SLbool          _isMirrored;            //!< Flag if input image must be mirrored
    SLCVSize        _boardSize;             //!< NO. of inner chessboard corners.
    SLfloat         _boardSquareMM;         //!< Size of chessboard square in mm
    SLint           _numOfImgsToCapture;    //!< NO. of images to capture
    SLint           _numCaptured;           //!< NO. of images captured
    SLfloat         _reprojectionError;     //!< Reprojection error after calibration
    SLCVVVPoint2f   _imagePoints;           //!< 2D vector of corner points in chessboard
    SLCVSize        _imageSize;             //!< Input image size in pixels
    SLbool          _showUndistorted;       //!< Flag if image should be undistorted
    SLstring        _calibrationTime;       //!< Time stamp string of calibration
};
//-----------------------------------------------------------------------------
#endif // SLCVCalibration_H
