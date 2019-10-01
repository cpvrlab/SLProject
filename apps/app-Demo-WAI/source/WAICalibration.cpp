#include <opencv2/core/core.hpp>

#include <iostream>

#include <WAICalibration.h>
#include <SLApplication.h>
#include <Utils.h>

using namespace std;
using namespace cv;

WAICalibration::WAICalibration()
{
    _imgSize.width  = 640;
    _imgSize.height = 480;
    reset();
}

void WAICalibration::changeImageSize(int width, int height)
{
    _imgSize.width  = width;
    _imgSize.height = height;
    reset();
}

void WAICalibration::reset()
{
    float fov = 65;
    computeMatrix(_cameraMat, fov);
    // No distortion
    _distortion = (Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

    _cameraFovDeg    = fov;
    _calibrationPath = std::string("");
    _state           = CalibrationState_Guess;
    _computerInfo    = SLApplication::getComputerInfos();
}

void WAICalibration::computeMatrix(cv::Mat& mat, float fov)
{
    float cx = (float)_imgSize.width * 0.5f;
    float cy = (float)_imgSize.height * 0.5f;
    float fx = cx / tanf(fov * 0.5f * M_PI / 180.0);
    float fy = fx;
    mat      = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

std::string WAICalibration::stateStr()
{
    if (_state == CalibrationState_Guess)
        return std::string("Guess");
    else
        return std::string("Calibrated");
}

bool WAICalibration::saveToFile(std::string path)
{
    FileStorage fs(path, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        return false;
    }

    fs << "imageSizeWidth" << _imgSize.width;
    fs << "imageSizeHeight" << _imgSize.height;
    fs << "cameraMat" << _cameraMat;
    fs << "distortion" << _distortion;
    fs << "numCaptured" << _numCaptured;
    fs << "isMirroredH" << _isMirroredH;
    fs << "isMirroredV" << _isMirroredV;
    fs << "calibFixAspectRatio" << _calibFixAspectRatio;
    fs << "calibFixPrincipalPoint" << _calibFixPrincipalPoint;
    fs << "calibZeroTangentDist" << _calibZeroTangentDist;
    fs << "reprojectionError" << _reprojectionError;
    fs << "calibrationTime" << _calibrationTime;
    fs << "camSizeIndex" << _camSizeIndex;
    fs << "FOV" << calcCameraHorizontalFOV();
    fs << "ComputerModel" << SLApplication::computerModel;
    fs << "CreationDate" << Utils::getDateTime2String();

    fs.release();

    return true;
}

bool WAICalibration::loadFromFile(std::string path)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Can t open calibration " << path << std::endl;
        return false;
    }

    _filename = Utils::getFileName(path);

    fs["imageSizeWidth"] >> _imgSize.width;
    fs["imageSizeHeight"] >> _imgSize.height;
    fs["cameraMat"] >> _cameraMat;
    fs["distortion"] >> _distortion;
    fs["numCaptured"] >> _numCaptured;
    fs["isMirroredH"] >> _isMirroredH;
    fs["isMirroredV"] >> _isMirroredV;
    fs["calibFixAspectRatio"] >> _calibFixAspectRatio;
    fs["calibFixPrincipalPoint"] >> _calibFixPrincipalPoint;
    fs["calibZeroTangentDist"] >> _calibZeroTangentDist;
    fs["reprojectionError"] >> _reprojectionError;
    fs["calibrationTime"] >> _calibrationTime;
    fs["camSizeIndex"] >> _camSizeIndex;

    fs["ComputerModel"] >> _computerInfo;
    if (_computerInfo.empty())
    {
        std::vector<std::string> stringParts;
        Utils::splitString(Utils::getFileNameWOExt(_filename), '_', stringParts);
        if (stringParts.size() >= 3)
            _computerInfo = stringParts[1];
        else
        {
            _computerInfo = SLApplication::getComputerInfos();
            std::cout << "Assuming calibration is for current device" << std::endl;
        }
    }

    fs["CreationDate"] >> _creationDate;
    fs.release();

    _state    = CalibrationState_Calibrated;
    float fov = calcCameraHorizontalFOV();

    _calibrationPath = path;
    //std::cout << "calibration file " << path << " loaded.    FOV = " << fov << std::endl;
    return true;
}

float WAICalibration::calcCameraVerticalFOV()
{
    float fy = (float)_cameraMat.at<double>(1, 1);
    float cy = (float)_cameraMat.at<double>(1, 2);
    return 2.0 * atan2(cy, fy) * 180.0 / M_PI;
}

float WAICalibration::calcCameraHorizontalFOV()
{
    float fx = (float)_cameraMat.at<double>(0, 0);
    float cx = (float)_cameraMat.at<double>(0, 2);
    return 2.0 * atan2(cx, fx) * 180.0 / M_PI;
}

float WAICalibration::calcCameraVerticalFOV(cv::Mat& cameraMat)
{
    float fy = (float)cameraMat.at<double>(1, 1);
    float cy = (float)cameraMat.at<double>(1, 2);
    return 2.0 * atan2(cy, fy) * 180.0 / M_PI;
}

float WAICalibration::calcCameraHorizontalFOV(cv::Mat& cameraMat)
{
    float fx = (float)cameraMat.at<double>(0, 0);
    float cx = (float)cameraMat.at<double>(0, 2);
    return 2.0 * atan2(cx, fx) * 180.0 / M_PI;
}
