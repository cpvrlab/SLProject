//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef ARCALIBRATION_H
#define ARCALIBRATION_H

#include <opencv2/core.hpp>
using namespace std;

class ARCalibration
{
public:
    enum CalibState {IDLE, CAPTURING, CALCULATING, CALIBRATED};

                ARCalibration       ();

    bool        loadCamParams       (string dir);
    bool        loadCalibParams     (string calibFilesDir);
    void        calibrate           ();
    void        addImage            (cv::Mat image);
    void        calculate           (string saveDir);

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
    void        clear();

    cv::Mat     _intrinsics;
    cv::Mat     _distortion;
    float       _cameraFovDeg;

    CalibState  _state;
    string      _calibFileName;         //!< name for calibration file
    string      _calibParamsFileName;

    int         _numInnerCornersWidth;
    int         _numInnerCornersHeight;
    float       _squareSizeMM;
    int         _captureDelayMS;
    int         _numOfImgsToCapture;

    int         _numCaptured;
    float       _reprojectionError;

    clock_t     _prevTimestamp;
    vector<vector<cv::Point2f>> _imagePoints;

    cv::Size    _imageSize;

    bool        _showUndistorted;
};

#endif // ARCALIBRATION_H
