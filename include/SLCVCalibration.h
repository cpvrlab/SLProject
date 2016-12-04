//#############################################################################
//  File:      SLCVCalibration.cpp
//  Author:    Michael Göttlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCALIBRATION_H
#define SLCVCALIBRATION_H

/* 
If an application uses live video processing you have to define 
the preprocessor contant SL_HAS_OPENCV in the project settings.
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/
#ifdef SL_HAS_OPENCV

#include <stdafx.h>
#include <SLCV.h>
using namespace std;

//-----------------------------------------------------------------------------
//! Live video camera calibration class with OpenCV an OpenCV calibration.
/* For the calibration internals see the OpenCV documentation:
http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
After a successufull calibration the parameters are stored in a config file on
the SLCVCalibration::defaultPath. If it exists, it is loaded from there at
startup. If doesn't exist a calibration can be done with the calibration scene 
(Load Scene > Augmented Reality > Calibrate Camera).\n
\n
The different calibration states are handled within SLScene::onUpdate:
\n
\nCS_uncalibrated:     The camera is not calibrated (no or invalid calibration found)
\nCS_calibrateStream:  The calibration is running with live video stream
\nCS_calibrateGrab:    The calibration is running and an image should be grabbed
\nCS_startCalculating: The calibration starts during the next frame
\nCS_calibrated:       The camera is calibrated
\n
The chessboard pattern can be printed from the CalibrationChessboard_8x5_A4.pdf
in the folder _data/calibration. It is important that one side has an odd number
of inner corners. Like this it is unambiguous and can be rotated in any direction.
*/
class SLCVCalibration
{
public:
                    SLCVCalibration     ();

    bool            loadCamParams       ();
    bool            loadCalibParams     ();
    void            setCalibrationState ();
    void            calculate           ();
    void            clear               ();
    void            showUndistorted     (bool su) {_showUndistorted = su;}
    bool            findChessboard      (SLCVMat imageColor,
                                         SLCVMat imageGray,
                                         bool drawCorners = true);

    static SLstring calibIniPath;       //!< calibration init parameters file path
    static void     calcBoardCorners3D  (SLCVSize boardSize, 
                                         SLfloat squareSize, 
                                         SLCVVPoint3f& objectPoints3D);

    // Setters
    void            state               (SLCVCalibState s) {_state = s;}

    // Getters
    SLCVSize        imageSize           () {return _imageSize;}
    SLfloat         imageAspectRatio    () {return (float)_imageSize.width/(float)_imageSize.height;}
    SLCVMat&        intrinsics          () {return _intrinsics;}
    SLCVMat&        distortion          () {return _distortion;}
    SLfloat         cameraFovDeg        () {return _cameraFovDeg;}
    SLCVCalibState  state               () {return _state;}
    SLint           numImgsToCapture    () {return _numOfImgsToCapture;}
    SLint           numCapturedImgs     () {return _numCaptured;}
    SLfloat         reprojectionError   () {return _reprojectionError;}
    SLbool          showUndistorted     () {return _showUndistorted;}
    SLCVSize        boardSize           () {return _boardSize;}
    SLfloat         boardSquareMM       () {return _boardSquareMM;}
    SLfloat         boardSquareM        () {return _boardSquareMM * 0.001f;}

private:
    void            calcCameraFOV       ();

    SLCVMat         _intrinsics;            //!< Matrix with intrisic camera paramters           
    SLCVMat         _distortion;            //!< Matrix with distortion parameters
    SLfloat         _cameraFovDeg;          //!< Field of view in degrees
    SLCVCalibState  _state;                 //!< calibration state enumeration 
    string          _calibFileName;         //!< name for calibration file
    string          _calibParamsFileName;   //!< name of calibration paramters file
    SLCVSize        _boardSize;             //!< NO. of inner chessboard corners.
    SLfloat         _boardSquareMM;         //!< Size of chessboard square in mm
    SLint           _numOfImgsToCapture;    //!< NO. of images to capture
    SLint           _numCaptured;           //!< NO. of images captured
    SLfloat         _reprojectionError;     //!< Reprojection error after calibration
    SLCVVVPoint2f   _imagePoints;           //!< 2D vector of corner points in chessboard
    SLCVSize        _imageSize;             //!< Input image size in pixels
    SLbool          _showUndistorted;       //!< Flag if image should be undistorted
};
//-----------------------------------------------------------------------------
#endif // SLCVCalibration_H
#endif // SL_HAS_OPENCV
