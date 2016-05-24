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

class ARCalibration
{
public:
    enum CalibState { IDLE, CAPTURING, CALCULATING, CALIBRATED };
    ARCalibration();
    bool loadCamParams(std::string dir);
    void calibrate();

    cv::Mat&    intrinsics() { return _intrinsics; }
    cv::Mat&    distortion() { return _distortion; }
    float       getCameraFov() { return _cameraFovDeg; }
    void        addImage(cv::Mat image);

    bool        capturing() { return _state == CAPTURING; }
    bool        calibrated() { return _state == CALIBRATED; }
    bool        uncalibrated() { return _state != CALIBRATED; }

    bool        loadCalibrationParams(std::string calibFilesDir);

    int         getNumImgsToCapture() { return _numOfImgsToCapture; }
    int         getNumCapturedImgs()  { return _numCaptured; }
    float       getReprojectionError() { return _reprojectionError; }
    void        calculate( std::string saveDir );
    bool        getShowUndistorted() { return _showUndistorted; }

private:
    void calculateCameraFieldOfView();
    void clear();

    cv::Mat _intrinsics;
    cv::Mat _distortion;
    float   _cameraFovDeg;

    //cv::Mat     _image;
    CalibState  _state;

    //name for calibration file
    std::string          _calibFileName;
    std::string          _calibParamsFileName;

    int _numInnerCornersWidth;
    int _numInnerCornersHeight;
    float _squareSizeMM;
    int _captureDelayMS;
    int _numOfImgsToCapture;

    int _numCaptured;
    float _reprojectionError;

    clock_t _prevTimestamp;
    std::vector<std::vector<cv::Point2f> > _imagePoints;

    cv::Size _imageSize;

    bool _showUndistorted;
};

#endif // ARCALIBRATION_H
