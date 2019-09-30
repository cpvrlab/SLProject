/*!
 * \file  CamCalibrationManager.cpp
 * \brief Camera calibration manager to calculate CamCalibration
 */

#include "CamCalibrationManager.h"
#include "Utils.h"

CamCalibrationManager::CamCalibrationManager(cv::Size boardSize, cv::Size imgSize, float squareSize, int numOfImgs)
  : _boardSize(boardSize), _imageSize(imgSize), _squareSize(squareSize), _minNumImgs(numOfImgs)
{
}

void CamCalibrationManager::addCorners(const std::vector<cv::Point2f>& corners)
{
    _calibCorners.push_back(corners);
}

CamCalibration CamCalibrationManager::calculateCalibration(
  bool fixAspectRatio,
  bool zeroTangentDistortion,
  bool fixPrincipalPoint)
{
    CamCalibration calib;

    //if (!readyForCalibration())
    //{
    //    SFV_WARN("Not enough corners collected!");
    //    return calib;
    //}

    // combine calibration flags
    int calibFlags = 0;
    if (fixAspectRatio)
        calibFlags |= cv::CALIB_FIX_ASPECT_RATIO;
    if (zeroTangentDistortion)
        calibFlags |= cv::CALIB_ZERO_TANGENT_DIST;
    if (fixPrincipalPoint)
        calibFlags |= cv::CALIB_FIX_PRINCIPAL_POINT;

    // calculate
    cv::Mat rvecs, tvecs;
    auto    boardCorners     = calcBoardCorners3D();
    calib._reprojectionError = cv::calibrateCamera(
      boardCorners, _calibCorners, _imageSize, calib._cameraMat, calib._distortion, rvecs, tvecs, calibFlags);

    // set additional parameters
    calib._imgSize               = _imageSize;
    calib._calibrationTime       = Utils::getLocalTimeString();
    calib._fixAspectRatio        = fixAspectRatio;
    calib._zeroTangentDistortion = zeroTangentDistortion;
    calib._fixPrincipalPoint     = fixPrincipalPoint;
    calib._zeroRadialDistortion  = false;
    return calib;
}

std::string CamCalibrationManager::getHelpMsg()
{
    std::string msg = "Collect " + std::to_string(_minNumImgs) +
                      " images of a chessboard from different angles. All inner corners have to be detected!";
    return msg;
}

std::string CamCalibrationManager::getStatusMsg()
{
    std::string msg = "(" + std::to_string(_calibCorners.size()) + "/" + std::to_string(_minNumImgs) + ")";
    return msg;
}

//! Calculates the 3D positions of the chessboard corners
std::vector<std::vector<cv::Point3f>> CamCalibrationManager::calcBoardCorners3D()
{
    std::vector<std::vector<cv::Point3f>> objectPoints3D;
    // Because OpenCV image coords are top-left we define the according
    // 3D coords also top-left.
    std::vector<cv::Point3f> ptsOfOneBoard;
    for (int y = _boardSize.height - 1; y >= 0; --y)
    {
        for (int x = 0; x < _boardSize.width; ++x)
        {
            ptsOfOneBoard.push_back(cv::Point3f(x * _squareSize, y * _squareSize, 0));
        }
    }

    objectPoints3D.resize(_calibCorners.size(), ptsOfOneBoard);
    return objectPoints3D;
}
