//#############################################################################
//  File:      SLCVCalibration.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCALIBRATION_H
#define SLCVCALIBRATION_H

#include <stdafx.h>
#include <SLCV.h>
using namespace std;

//-----------------------------------------------------------------------------
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
    bool            findChessboard      (SLCVMat image,
                                         bool drawCorners = true);

    static SLstring defaultPath;        //!< Default path for calibration files

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
    bool            showUndistorted     () {return _showUndistorted;}
    SLCVSize        boardSize           () {return _boardSize;}
    SLfloat         boardSquareMM       () {return _boardSquareMM;}
    SLfloat         boardSquareM        () {return _boardSquareMM * 0.001f;}

private:
    void            calcCameraFOV  ();

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
    bool            _showUndistorted;       //!< Flag if image should be undistorted
};
//-----------------------------------------------------------------------------
#endif // SLCVCalibration_H
