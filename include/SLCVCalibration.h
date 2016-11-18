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
#include <opencv2/core.hpp>
using namespace std;

//-----------------------------------------------------------------------------
class SLCVCalibration
{
public:
                SLCVCalibration     ();

    bool        loadCamParams       ();
    bool        loadCalibParams     ();
    void        calibrate           ();
    void        calculate           ();
    void        showUndistorted     (bool su) {_showUndistorted = su;}
    SLMat4f     createGLMatrix      (const cv::Mat& tVec, 
                                     const cv::Mat& rVec);
    bool        findChessboard      (cv::Mat image,
                                     bool drawCorners = true);

    static SLstring defaultPath;    //!< Default path for calibration files

    // Setters
    void            state               (SLCVCalibState s) {_state = s;}

    // Getters
    cv::Mat&        intrinsics          () {return _intrinsics;}
    cv::Mat&        distortion          () {return _distortion;}
    float           cameraFovDeg        () {return _cameraFovDeg;}
    SLCVCalibState  state               () {return _state;}
    int             numImgsToCapture    () {return _numOfImgsToCapture;}
    int             numCapturedImgs     () {return _numCaptured;}
    float           reprojectionError   () {return _reprojectionError;}
    bool            showUndistorted     () {return _showUndistorted;}
    cv::Size        boardSize           () {return _boardSize;}
    float           boardSquareMM       () {return _boardSquareMM;}
    float           boardSquareM        () {return _boardSquareMM * 0.001f;}

private:
    void            calculateCameraFOV  ();
    void            clear               ();

    cv::Mat         _intrinsics;            //!< Matrix with intrisic camera paramters           
    cv::Mat         _distortion;            //!< Matrix with distortion parameters
    float           _cameraFovDeg;          //!< Field of view in degrees

    SLCVCalibState  _state;
    string          _calibFileName;         //!< name for calibration file
    string          _calibParamsFileName;   //!< name of calibration paramters file
    cv::Size        _boardSize;             //!< NO. of inner chessboard corners.
    float           _boardSquareMM;         //!< Size of chessboard square in mm
    int             _numOfImgsToCapture;    //!< NO. of images to capture
    int             _numCaptured;           //!< NO. of images captured
    float           _reprojectionError;     //!< Reprojection error after calibration

    vector<vector<cv::Point2f>> _imagePoints;

    cv::Size        _imageSize;
    bool            _showUndistorted;
};
//-----------------------------------------------------------------------------
#endif // SLCVCalibration_H
