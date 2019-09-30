/*!
* \file  CamCalibration.h
* \brief Camera calibration for perspective camera model
*/

#include "CamCalibration.h"
#include "Utils.h"

#ifdef _WINDOWS
#    define _USE_MATH_DEFINES
#    include <math.h>
#endif // _WINDOWS

#define DEGTORAD 180 / M_PI
#define RADTODEG M_PI / 180

CamCalibration::CamCalibration(std::string filePath)
{
    load(filePath);
}

CamCalibration::CamCalibration(cv::Size imgSize, double horizFOV)
  : _imgSize(imgSize)
{
    double cx = imgSize.width / 2;
    double cy = imgSize.height / 2;
    double f  = cx / tanf(horizFOV * 0.5 * DEGTORAD);

    _cameraMat                  = cv::Mat::eye(3, 3, CV_64F);
    _cameraMat.at<double>(0, 0) = f;
    _cameraMat.at<double>(1, 1) = f;
    _cameraMat.at<double>(0, 2) = cx;
    _cameraMat.at<double>(1, 2) = cy;
}

CamCalibration::CamCalibration(cv::Size imgSize, float camFocalLengthMM, cv::Size2f camSensorSizeMM)
  : _imgSize(imgSize)
{
    double cx = imgSize.width / 2;
    double cy = imgSize.height / 2;
    double f  = imgSize.width * camFocalLengthMM / camSensorSizeMM.width;

    _cameraMat                  = cv::Mat::eye(3, 3, CV_64F);
    _cameraMat.at<double>(0, 0) = f;
    _cameraMat.at<double>(1, 1) = f;
    _cameraMat.at<double>(0, 2) = cx;
    _cameraMat.at<double>(1, 2) = cy;
}

CamCalibration::CamCalibration(const CamCalibration& toCopy)
  : _cameraMat(toCopy._cameraMat.clone()),
    _distortion(toCopy._distortion.clone()),
    _calibrationTime(toCopy._calibrationTime),
    _reprojectionError(toCopy._reprojectionError),
    _imgSize(toCopy._imgSize),
    _fixAspectRatio(toCopy._fixAspectRatio),
    _zeroTangentDistortion(toCopy._zeroTangentDistortion),
    _fixPrincipalPoint(toCopy._fixPrincipalPoint),
    _scaleFactor(toCopy._scaleFactor)
{
}

//! load camera calibration from given path
void CamCalibration::load(std::string filePath)
{
    //todo: check if file exists and throw exception
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Could not open calibration file storage!");

    fs["cameraMat"] >> _calibrationTime;
    fs["cameraMat"] >> _cameraMat;
    fs["distortion"] >> _distortion;
    fs["imgSize"] >> _imgSize;
    fs["reprojectionError"] >> _reprojectionError;
    fs["fixAspectRatio"] >> _fixAspectRatio;
    fs["zeroTangentDistortion"] >> _zeroTangentDistortion;
    fs["fixPrincipalPoint"] >> _fixPrincipalPoint;
    if (!fs["zeroRadialDistortion"].empty())
        fs["zeroRadialDistortion"] >> _zeroRadialDistortion;

    fs.release();
}

void CamCalibration::save(const std::string& filePath)
{
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::stringstream ss;
        ss << "Unable to store camera calibration to file: " << filePath;
        throw std::runtime_error(ss.str().c_str());
    }

    fs << "calibrationTime" << _calibrationTime;
    fs << "cameraMat" << _cameraMat;
    fs << "distortion" << _distortion;
    fs << "imgSize" << _imgSize;
    fs << "reprojectionError" << _reprojectionError;
    fs << "fixAspectRatio" << _fixAspectRatio;
    fs << "zeroTangentDistortion" << _zeroTangentDistortion;
    fs << "zeroRadialDistortion" << _zeroRadialDistortion;
    fs << "fixPrincipalPoint" << _fixPrincipalPoint;
    fs.release();
}

void CamCalibration::scale(double scaleFactor)
{
    _scaleFactor = scaleFactor;
    _cameraMat *= scaleFactor;
    _cameraMat.at<double>(2, 2) = 1.0;
}

const float CamCalibration::getHorizontalFOV() const
{
    //FOV = 2*atan(width*0.5/distance) *180 / PI()
    float f   = 0.5 * (_cameraMat.at<double>(0, 0) + _cameraMat.at<double>(1, 1));
    float fov = 2 * atanf(0.5 * _imgSize.width / f) * RADTODEG;
    return fov;
}
