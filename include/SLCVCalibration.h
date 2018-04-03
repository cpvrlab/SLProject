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
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <stdafx.h>
#include <SLCV.h>
#include <SLCVCalibration.h>
using namespace std;

//-----------------------------------------------------------------------------
//! Live video camera calibration class with OpenCV an OpenCV calibration.
/*! The camera calibration can determine the inner or intrinsic parameters such
as the focal length and the lens distortion and external or extrinsic parameter
such as the camera pose towards a known geometry.
\n
For a good calibration we have to make 15-20 images from a chessboard pattern.
The chessboard pattern can be printed from the CalibrationChessboard_8x5_A4.pdf
in the folder _data/calibration. It is important that one side has an odd number
of inner corners. Like this it is unambiguous and can be rotated in any direction.
\n
The different calibration states are handled within SLScene::onUpdate:
\n
- CS_uncalibrated:     The camera is not calibrated (no calibration found found)
- CS_calibrateStream:  The calibration is running with live video stream
- CS_calibrateGrab:    The calibration is running and an image should be grabbed
- CS_startCalculating: The calibration starts during the next frame
- CS_calibrated:       The camera is calibrated
- CS_estimate:         The camera intrinsics are set from an estimated FOV angle
\n
The core of the intrinsic calibration is stored in the members _cameraMat and
_distortion. For the calibration internals see the OpenCV documentation:
http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
After a successufull calibration the parameters are stored in a config file on
the SL::configPath. If it exists, it is loaded from there at startup.
If doesn't exist a simple calibration from a default field of view angle is
estimated.
\n
The SLScene instance has two video camera calibrations, one for a main camera
(SLScene::_calibMainCam) and one for the selfie camera on mobile devices
(SLScene::_calibScndCam). The member SLScene::_activeCalib references the active
one and is set by the SLScene::videoType (VT_NONE, VT_MAIN, VT_SCND) during the
scene assembly in SLScene::onLoad. On mobile devices the front camera is the
selfie camera (our secondary) and the back camera is the our main camera.
*/
class SLCVCalibration
{
public:
                    SLCVCalibration         ();
                   ~SLCVCalibration         (){;}
    bool            load                    (SLstring calibFileName,
                                             SLbool mirrorHorizontally,
                                             SLbool mirrorVertically);
    void            save                    ();
    bool            loadCalibParams         ();
    bool            calculate               ();
    void            clear                   ();
    SLfloat         calcReprojectionErr     (const SLCVVVPoint3f& objectPoints,
                                             const SLCVVMat& rvecs,
                                             const SLCVVMat& tvecs,
                                             SLVfloat& perViewErrors);
    bool            findChessboard          (SLCVMat imageColor,
                                             SLCVMat imageGray,
                                             bool drawCorners = true);
    void            buildUndistortionMaps   ();
    void            remap                   (SLCVMat &inDistorted,
                                             SLCVMat &outUndistorted);
    void            createFromGuessedFOV    (SLint imageWidthPX,
                                             SLint imageHeightPX);

    static SLstring calibIniPath;           //!< calibration init parameters file path
    static void     calcBoardCorners3D      (SLCVSize boardSize,
                                             SLfloat squareSize,
                                             SLCVVPoint3f& objectPoints3D);
    // Setters
    void            state                   (SLCVCalibState s) {_state = s;}
    void            toggleMirrorH           () {clear(); _isMirroredH = !_isMirroredH; save();}
    void            toggleMirrorV           () {clear(); _isMirroredV = !_isMirroredV; save();}
    void            toggleFixPrincipalPoint () {clear(); _calibFixPrincipalPoint = !_calibFixPrincipalPoint;}
    void            toggleFixAspectRatio    () {clear(); _calibFixAspectRatio = !_calibFixAspectRatio;}
    void            toggleZeroTangentDist   () {clear(); _calibZeroTangentDist = !_calibZeroTangentDist;}
    void            showUndistorted         (SLbool su) {_showUndistorted = su;}

    // Getters
    SLCVSize        imageSize               () {return _imageSize;}
    SLfloat         imageAspectRatio        () {return (float)_imageSize.width/(float)_imageSize.height;}
    SLCVMat&        cameraMat               () {return _cameraMat;}
    SLCVMat&        distortion              () {return _distortion;}
    SLfloat         cameraFovDeg            () {return _cameraFovDeg;}
    SLbool          calibFixPrincipalPoint  () {return _calibFixPrincipalPoint;}
    SLbool          calibFixAspectRatio     () {return _calibFixAspectRatio;}
    SLbool          calibZeroTangentDist    () {return _calibZeroTangentDist;}
    SLbool          isMirroredH             () {return _isMirroredH;}
    SLbool          isMirroredV             () {return _isMirroredV;}
    SLfloat         fx                      () {return _cameraMat.cols==3 && _cameraMat.rows==3 ? (SLfloat)_cameraMat.at<double>(0,0) : 0.0f;}
    SLfloat         fy                      () {return _cameraMat.cols==3 && _cameraMat.rows==3 ? (SLfloat)_cameraMat.at<double>(1,1) : 0.0f;}
    SLfloat         cx                      () {return _cameraMat.cols==3 && _cameraMat.rows==3 ? (SLfloat)_cameraMat.at<double>(0,2) : 0.0f;}
    SLfloat         cy                      () {return _cameraMat.cols==3 && _cameraMat.rows==3 ? (SLfloat)_cameraMat.at<double>(1,2) : 0.0f;}
    SLfloat         k1                      () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(0,0) : 0.0f;}
    SLfloat         k2                      () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(1,0) : 0.0f;}
    SLfloat         p1                      () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(2,0) : 0.0f;}
    SLfloat         p2                      () {return _distortion.rows>=4 ? (SLfloat)_distortion.at<double>(3,0) : 0.0f;}
    SLCVCalibState  state                   () {return _state;}
    SLint           numImgsToCapture        () {return _numOfImgsToCapture;}
    SLint           numCapturedImgs         () {return _numCaptured;}
    SLfloat         reprojectionError       () {return _reprojectionError;}
    SLbool          showUndistorted         () {return _showUndistorted;}
    SLCVSize        boardSize               () {return _boardSize;}
    SLfloat         boardSquareMM           () {return _boardSquareMM;}
    SLfloat         boardSquareM            () {return _boardSquareMM * 0.001f;}
    SLstring        calibrationTime         () {return _calibrationTime;}
    SLstring        calibFileName           () {return _calibFileName;}
    SLstring        stateStr                ()
    {   switch(_state)
        {   case CS_uncalibrated:       return "CS_uncalibrated";
            case CS_calibrated:         return "CS_calibrated";
            case CS_guessed:            return "CS_guessed";
            case CS_calibrateStream:    return "CS_calibrateStream";
            case CS_calibrateGrab:      return "CS_calibrateGrab";
            case CS_startCalculating:   return "CS_startCalculating";
            default:                    return "unknown";
        }
    }

private:
    SLfloat         calcCameraFOV       ();

    ///////////////////////////////////////////////////////////////////////////////////
    SLCVMat         _cameraMat;             //!< 3x3 Matrix for intrinsic camera matrix
    SLCVMat         _distortion;            //!< 4x1 Matrix for intrinsic distortion
    ///////////////////////////////////////////////////////////////////////////////////

    SLCVCalibState  _state;                 //!< calibration state enumeration
    SLfloat         _cameraFovDeg;          //!< Vertical field of view in degrees
    SLstring        _calibFileName;         //!< name for calibration file
    SLstring        _calibParamsFileName;   //!< name of calibration paramters file
    SLint           _calibFlags;            //!< OpenCV calibration flags
    SLbool          _calibFixPrincipalPoint;//!< Calib. flag for fix principal point
    SLbool          _calibFixAspectRatio;   //!< Calib. flag for fix aspect ratio
    SLbool          _calibZeroTangentDist;  //!< Calib. flag for zero tangent distortion
    SLbool          _isMirroredH;           //!< Flag if image must be horizontally mirrored
    SLbool          _isMirroredV;           //!< Flag if image must be vertically mirrored
    SLCVSize        _boardSize;             //!< NO. of inner chessboard corners.
    SLfloat         _boardSquareMM;         //!< Size of chessboard square in mm
    SLint           _numOfImgsToCapture;    //!< NO. of images to capture
    SLint           _numCaptured;           //!< NO. of images captured
    SLfloat         _reprojectionError;     //!< Reprojection error after calibration
    SLCVVVPoint2f   _imagePoints;           //!< 2D vector of corner points in chessboard
    SLCVSize        _imageSize;             //!< Input image size in pixels
    SLbool          _showUndistorted;       //!< Flag if image should be undistorted
    SLCVMat         _undistortMapX;         //!< Undistortion float map in x-direction 
    SLCVMat         _undistortMapY;         //!< Undistortion float map in y-direction
    SLCVMat         _cameraMatUndistorted;  //!< Camera matrix for undistorted image
    SLstring        _calibrationTime;       //!< Time stamp string of calibration

    static const SLint _CALIBFILEVERSION;   //!< Global const file format version
};
//-----------------------------------------------------------------------------
#endif // SLCVCalibration_H
