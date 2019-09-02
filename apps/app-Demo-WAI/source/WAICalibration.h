#ifndef WAICALIBRATION
#define WAICALIBRATION
using namespace std;

#define _USE_MATH_DEFINES
#include <math.h>
#include <WAISensorCamera.h>

enum CalibrationState
{
    CalibrationState_None,
    CalibrationState_Guess,
    CalibrationState_Calibrated
};

class WAICalibration
{
    public:
    WAICalibration();
    virtual bool loadFromFile(std::string path);
    virtual bool saveToFile(std::string path);
    virtual void reset();
    virtual void changeImageSize(int width, int height);

    float calcCameraVerticalFOV();
    float calcCameraHorizontalFOV();
    float calcCameraVerticalFOV(cv::Mat& cameraMat);
    float calcCameraHorizontalFOV(cv::Mat& cameraMat);

    void                   computeMatrix(cv::Mat& mat, float fov);
    WAI::CameraCalibration getCameraCalibration();
    cv::Mat&               cameraMat() { return _cameraMat; }
    cv::Mat&               distortion() { return _distortion; }
    CalibrationState       getState() { return _state; }
    std::string            getCalibrationPath() { return _calibrationPath; }
    std::string            stateStr();

    float fx() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(0, 0) : 0.0f; }
    float fy() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(1, 1) : 0.0f; }
    float cx() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(0, 2) : 0.0f; }
    float cy() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(1, 2) : 0.0f; }
    float k1() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(0, 0) : 0.0f; }
    float k2() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(1, 0) : 0.0f; }
    float p1() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(2, 0) : 0.0f; }
    float p2() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(3, 0) : 0.0f; }

    protected:
    CalibrationState _state;
    cv::Mat          _cameraMat;
    cv::Mat          _distortion;
    cv::Size         _imgSize;
    float            _cameraFovDeg;
    std::string      _calibrationPath;
    int              _numCaptured;
    bool             _isMirroredH;
    bool             _isMirroredV;
    bool             _calibFixAspectRatio;
    bool             _calibFixPrincipalPoint;
    bool             _calibZeroTangentDist;
    float            _reprojectionError;
    float            _calibrationTime;
    int              _camSizeIndex;
    std::string      _computerModel;
    std::string      _creationDate;
};
#endif
