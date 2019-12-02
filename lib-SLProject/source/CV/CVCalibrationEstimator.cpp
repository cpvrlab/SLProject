//#############################################################################
//  File:      CVCalibration.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVCalibrationEstimator.h>
#include <CVCalibration.h>
#include <Utils.h>
#include <algorithm> // std::max
#include <SLApplication.h>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
CVCalibrationEstimator::CVCalibrationEstimator(int calibFlags)
  : _calibFlags(calibFlags),
    _calibParamsFileName("calib_in_params.yml")
{
    if (!loadCalibParams())
    {
        Utils::exitMsg("CVCalibrationEstimator: could not load calibration parameter",
                       __LINE__,
                       __FILE__);
    }

    _state = State::Stream;
}
//-----------------------------------------------------------------------------
//! Initiates the final calculation
bool CVCalibrationEstimator::calculate()
{
    bool calibrationSuccessful = false;
    if (!_calibrationTask.valid())
    {
        _calibrationTask = std::async(std::launch::async, &CVCalibrationEstimator::calibrateAsync, this);
    }
    else if (_calibrationTask.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
    {
        calibrationSuccessful = _calibrationTask.get();
        if (calibrationSuccessful)
        {
            //todo: transfer calibration to mainCalib
            //_state = CVCalibEstimState::Calculating;

            //save();
            Utils::log("Calibration succeeded.");
            Utils::log("Reproj. error: %f\n", _reprojectionError);
        }
        else
        {
            //todo
            //_state = CS_uncalibrated;
            Utils::log("Calibration failed.");
        }
    }

    return calibrationSuccessful;
}
//-----------------------------------------------------------------------------
bool CVCalibrationEstimator::calibrateAsync()
{
    _state = State::Calculating;
    //todo
    //_computerInfos = SLApplication::getComputerInfos();

    _numCaptured = 0;
    //extract corners from captured images
    for (cv::Mat img : _calibrationImgs)
    {
        CVVPoint2f preciseCorners2D;
        int        flags          = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
        bool       foundPrecisely = cv::findChessboardCorners(img,
                                                        _boardSize,
                                                        preciseCorners2D,
                                                        flags);

        if (foundPrecisely)
        {
            cv::cornerSubPix(img,
                             preciseCorners2D,
                             CVSize(11, 11),
                             CVSize(-1, -1),
                             TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,
                                          30000,
                                          0.01));

            //add detected points
            _imagePoints.push_back(preciseCorners2D);
            _numCaptured++;
        }
    }

    CVVMat        rvecs, tvecs;
    vector<float> reprojErrs;
    cv::Mat       cameraMat;
    cv::Mat       distortion;

    bool ok = calcCalibration(_imageSize,
                              cameraMat,
                              distortion,
                              _imagePoints,
                              rvecs,
                              tvecs,
                              reprojErrs,
                              _reprojectionError,
                              _boardSize,
                              _boardSquareMM,
                              _calibFlags);
    //correct number of caputured, extraction may have failed
    if (!rvecs.empty() || !reprojErrs.empty())
        _numCaptured = (int)std::max(rvecs.size(), reprojErrs.size());
    else
        _numCaptured = 0;

    if (ok)
    {
        std::string calibrationTime = Utils::getDateTime2String();
    }

    return ok;
}
//-----------------------------------------------------------------------------
//! Calculates the calibration with the given set of image points
bool CVCalibrationEstimator::calcCalibration(CVSize&            imageSize,
                                             CVMat&             cameraMatrix,
                                             CVMat&             distCoeffs,
                                             const CVVVPoint2f& imagePoints,
                                             CVVMat&            rvecs,
                                             CVVMat&            tvecs,
                                             vector<float>&     reprojErrs,
                                             float&             totalAvgErr,
                                             CVSize&            boardSize,
                                             float              squareSize,
                                             int                flag)
{
    // Init camera matrix with the eye setter
    cameraMatrix = CVMat::eye(3, 3, CV_64F);

    // We need to set eleme at 0,0 to 1 if we want a fix aspect ratio
    if (flag & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = 1.0;

    // init the distortion coeffitients to zero
    distCoeffs = CVMat::zeros(8, 1, CV_64F);

    CVVVPoint3f objectPoints(1);

    CVCalibrationEstimator::calcBoardCorners3D(boardSize,
                                               squareSize,
                                               objectPoints[0]);

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    ////////////////////////////////////////////////
    //Find intrinsic and extrinsic camera parameters
    double rms = cv::calibrateCamera(objectPoints,
                                     imagePoints,
                                     imageSize,
                                     cameraMatrix,
                                     distCoeffs,
                                     rvecs,
                                     tvecs,
                                     flag);
    ////////////////////////////////////////////////

    Utils::log("Re-projection error reported by calibrateCamera: %f\n", rms);

    bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

    totalAvgErr = (float)calcReprojectionErrors(objectPoints,
                                                imagePoints,
                                                rvecs,
                                                tvecs,
                                                cameraMatrix,
                                                distCoeffs,
                                                reprojErrs);
    return ok;
}
//-----------------------------------------------------------------------------
//! Calculates the reprojection error of the calibration
double CVCalibrationEstimator::calcReprojectionErrors(const CVVVPoint3f& objectPoints,
                                                      const CVVVPoint2f& imagePoints,
                                                      const CVVMat&      rvecs,
                                                      const CVVMat&      tvecs,
                                                      const CVMat&       cameraMatrix,
                                                      const CVMat&       distCoeffs,
                                                      vector<float>&     perViewErrors)
{
    CVVPoint2f imagePoints2;
    size_t     totalPoints = 0;
    double     totalErr    = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        cv::projectPoints(objectPoints[i],
                          rvecs[i],
                          tvecs[i],
                          cameraMatrix,
                          distCoeffs,
                          imagePoints2);

        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n         = objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}
//-----------------------------------------------------------------------------
//! Loads the chessboard calibration pattern parameters
bool CVCalibrationEstimator::loadCalibParams()
{
    FileStorage fs;
    string      fullCalibIniFile = CVCalibration::calibIniPath + _calibParamsFileName;

    fs.open(fullCalibIniFile, FileStorage::READ);
    if (!fs.isOpened())
    {
        Utils::log("Could not open the calibration parameter file: %s\n", fullCalibIniFile.c_str());
        return false;
    }

    //assign paramters
    fs["numInnerCornersWidth"] >> _boardSize.width;
    fs["numInnerCornersHeight"] >> _boardSize.height;
    fs["squareSizeMM"] >> _boardSquareMM;
    fs["numOfImgsToCapture"] >> _numOfImgsToCapture;

    return true;
}
//-----------------------------------------------------------------------------
//!< Finds the inner chessboard corners in the given image
bool CVCalibrationEstimator::updateAndDecorate(CVMat        imageColor,
                                               const CVMat& imageGray,
                                               bool         grabFrame,
                                               bool         drawCorners)
{
    assert(!imageGray.empty() &&
           "CVCalibration::findChessboard: imageGray is empty!");
    assert(!imageColor.empty() &&
           "CVCalibration::findChessboard: imageColor is empty!");
    assert(_boardSize.width && _boardSize.height &&
           "CVCalibration::findChessboard: _boardSize is not set!");

    //debug save image
    //stringstream ss;
    //ss << "imageIn_" << _numCaptured << ".png";
    //cv::imwrite(ss.str(), imageColor);

    cv::Size imageSize = imageColor.size();

    cv::Mat imageGrayExtract = imageGray;
    //resize image so that we get fluent caputure workflow for high resolutions
    double scale              = 1.0;
    bool   doScale            = false;
    int    targetExtractWidth = 640;
    if (imageSize.width > targetExtractWidth)
    {
        doScale = true;
        scale   = (double)imageSize.width / (double)targetExtractWidth;
        cv::resize(imageGray, imageGrayExtract, cv::Size(), 1 / scale, 1 / scale);
    }

    CVVPoint2f corners2D;
    bool       found = cv::findChessboardCorners(imageGrayExtract,
                                           _boardSize,
                                           corners2D,
                                           cv::CALIB_CB_FAST_CHECK);

    if (found)
    {
        if (grabFrame)
        {
            //save a copy of this image
            _calibrationImgs.push_back(imageGray.clone());
            //increase number of capturings
            _numCaptured++;

            //simulate a snapshot
            cv::bitwise_not(imageColor, imageColor);
            _state = State::Stream;
        }

        if (drawCorners)
        {
            if (doScale)
            {
                //scale corners into original image size
                for (cv::Point2f& pt : corners2D)
                {
                    pt *= scale;
                }
            }

            cv::drawChessboardCorners(imageColor,
                                      _boardSize,
                                      CVMat(corners2D),
                                      found);
        }
    }
    return found;
}
//-----------------------------------------------------------------------------
//! Calculates the 3D positions of the chessboard corners
void CVCalibrationEstimator::calcBoardCorners3D(const CVSize& boardSize,
                                                float         squareSize,
                                                CVVPoint3f&   objectPoints3D)
{
    // Because OpenCV image coords are top-left we define the according
    // 3D coords also top-left.
    objectPoints3D.clear();
    for (int y = boardSize.height - 1; y >= 0; --y)
        for (int x = 0; x < boardSize.width; ++x)
            objectPoints3D.push_back(CVPoint3f((float)x * squareSize,
                                               (float)y * squareSize,
                                               0));
}
