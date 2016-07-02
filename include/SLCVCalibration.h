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
    enum SLCVCalibState {IDLE, CAPTURING, CALCULATING, CALIBRATED};

                SLCVCalibration       ();

    bool        loadCamParams       ();
    bool        loadCalibParams     ();
    void        calibrate           ();
    void        addImage            (cv::Mat image);
    void        calculate           ();
    void        showUndistorted     (bool su) {_showUndistorted = su;}
    SLMat4f     createGLMatrix      (const cv::Mat& tVec, 
                                     const cv::Mat& rVec);
    static bool findChessboard      (cv::Mat& frame,
                                     cv::Size& size,
                                     vector<cv::Point2f>& corners,
                                     int flags);

    static SLstring defaultPath;    //!< Default path for calibration files

    // Getters
    cv::Mat&    intrinsics          () {return _intrinsics;}
    cv::Mat&    distortion          () {return _distortion;}
    float       cameraFovDeg        () {return _cameraFovDeg;}
    bool        stateIsCapturing    () {return _state == CAPTURING;}
    bool        stateIsCalibrated   () {return _state == CALIBRATED;}
    int         numImgsToCapture    () {return _numOfImgsToCapture;}
    int         numCapturedImgs     () {return _numCaptured;}
    float       reprojectionError   () {return _reprojectionError;}
    bool        showUndistorted     () {return _showUndistorted;}

private:
    void        calculateCameraFOV  ();
    void        clear               ();

    cv::Mat         _intrinsics;            
    cv::Mat         _distortion;
    float           _cameraFovDeg;

    SLCVCalibState  _state;
    string          _calibFileName;         //!< name for calibration file
    string          _calibParamsFileName;
    int             _numInnerCornersWidth;
    int             _numInnerCornersHeight;
    float           _squareSizeMM;
    int             _captureDelayMS;
    int             _numOfImgsToCapture;
    int             _numCaptured;
    float           _reprojectionError;

    clock_t         _prevTimestamp;
    vector<vector<cv::Point2f>> _imagePoints;

    cv::Size        _imageSize;
    bool            _showUndistorted;
};
//-----------------------------------------------------------------------------
#endif // SLCVCalibration_H
