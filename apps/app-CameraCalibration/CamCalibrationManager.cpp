/*!
 * \file  CamCalibrationManager.cpp
 * \brief Camera calibration manager to calculate CamCalibration
 */

#include "CamCalibrationManager.h"
#include <CVCalibrationEstimator.h>
#include "Utils.h"

CamCalibrationManager::CamCalibrationManager(cv::Size boardSize,
                                             cv::Size imgSize,
                                             float    squareSize,
                                             int      numOfImgs,
                                             bool     useReleaseObjectMethod)
  : _boardSize(boardSize),
    _imageSize(imgSize),
    _squareSize(squareSize),
    _minNumImgs(numOfImgs),
    _useReleaseObjectMethod(useReleaseObjectMethod)
{
}

void CamCalibrationManager::addCorners(const std::vector<cv::Point2f>& corners)
{
    _calibCorners.push_back(corners);
}

CVCalibration CamCalibrationManager::calculateCalibration(
  bool fixAspectRatio,
  bool zeroTangentDistortion,
  bool fixPrincipalPoint,
  bool calibRationalModel,
  bool calibTiltedModel,
  bool calibThinPrismModel)
{
    // combine calibration flags
    int flags = 0;
    if (fixPrincipalPoint)
        flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if (fixAspectRatio)
        flags |= cv::CALIB_FIX_ASPECT_RATIO;
    if (zeroTangentDistortion)
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
    if (calibRationalModel)
        flags |= cv::CALIB_RATIONAL_MODEL;
    if (calibTiltedModel)
        flags |= cv::CALIB_TILTED_MODEL;
    if (calibThinPrismModel)
        flags |= cv::CALIB_THIN_PRISM_MODEL;

    cv::Mat cameraMat;
    cv::Mat distortion;

    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float>   reprojErrs;
    float                totalAvgErr = 0;
    vector<cv::Point3f>  newObjPoints;

    bool          ok = CVCalibrationEstimator::calcCalibration(_imageSize,
                                                      cameraMat,
                                                      distortion,
                                                      _calibCorners,
                                                      rvecs,
                                                      tvecs,
                                                      reprojErrs,
                                                      totalAvgErr,
                                                      _boardSize,
                                                      _squareSize,
                                                      flags,
                                                      _useReleaseObjectMethod);
    CVCalibration calibration(cameraMat,
                              distortion,
                              _imageSize,
                              _boardSize,
                              _squareSize,
                              totalAvgErr,
                              _calibCorners.size(),
                              Utils::getLocalTimeString(),
                              false,
                              false,
                              0,
                              CVCameraType::FRONTFACING,
                              "",
                              flags);
    return calibration;
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
