//#############################################################################
//  File:      SENSCalibrationEstimator.cpp
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SENSCalibrationEstimator.h>
#include <SENSCalibration.h>
#include <Utils.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
SENSCalibrationEstimator::SENSCalibrationEstimator(SENSCalibrationEstimatorParams params,
                                                   int                            camSizeIndex,
                                                   bool                           mirroredH,
                                                   bool                           mirroredV,
                                                   SENSCameraType                 camType,
                                                   std::string                    computerInfos,
                                                   std::string                    calibDataPath,
                                                   std::string                    imageOutputPath,
                                                   std::string                    exePath)
  : _params(params),
    _camSizeIndex(camSizeIndex),
    _mirroredH(mirroredH),
    _mirroredV(mirroredV),
    _camType(camType),
    _calibParamsFileName("calib_in_params.yml"),
    _exception("Undefined error", 0, __FILE__),
    _computerInfos(computerInfos),
    _calibDataPath(calibDataPath),
    _exePath(exePath)
{
    if (!loadCalibParams())
    {
        throw SENSCalibrationEstimatorException("Could not load calibration parameter!",
                                                __LINE__,
                                                __FILE__);
    }

    if (params.mode == SENSCalibrationEstimatorParams::EstimatorMode::OnlyCaptureAndSave)
    {
        if (!Utils::dirExists(imageOutputPath))
        {
            std::stringstream ss;
            ss << "Image output directory does not exist: " << imageOutputPath;
            throw SENSCalibrationEstimatorException(ss.str(),
                                                    __LINE__,
                                                    __FILE__);
        }
        else
        {
            //make subdirectory where images are stored to
            _calibImgOutputDir = Utils::unifySlashes(imageOutputPath) + "calibimages/";
            Utils::makeDir(_calibImgOutputDir);
            if (!Utils::dirExists(_calibImgOutputDir))
            {
                std::stringstream ss;
                ss << "Could not create image output directory: " << _calibImgOutputDir;
                throw SENSCalibrationEstimatorException(ss.str(),
                                                        __LINE__,
                                                        __FILE__);
            }
        }
    }
}
//-----------------------------------------------------------------------------
SENSCalibrationEstimator::~SENSCalibrationEstimator()
{
    //wait for the async task to finish
    if (_calibrationTask.valid())
        _calibrationTask.wait();
}
//-----------------------------------------------------------------------------
//! Initiates the final calculation
bool SENSCalibrationEstimator::calculate()
{
    bool calibrationSuccessful = false;
    if (!_calibrationTask.valid())
    {
        _calibrationTask = std::async(std::launch::async, &SENSCalibrationEstimator::calibrateAsync, this);
    }
    else if (_calibrationTask.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
    {
        calibrationSuccessful = _calibrationTask.get();
        if (calibrationSuccessful)
        {
            Utils::log("SLProject", "Calibration succeeded.");
            Utils::log("SLProject", "Reproj. error: %f", _reprojectionError);
        }
        else
        {
            Utils::log("SLProject", "Calibration failed.");
        }
    }

    return calibrationSuccessful;
}
//-----------------------------------------------------------------------------
bool SENSCalibrationEstimator::extractAsync()
{
    if (_imageSize.width == 0 && _imageSize.height == 0)
        _imageSize = _currentImgToExtract.size();
    else if (_imageSize.width != _currentImgToExtract.size().width || _imageSize.height != _currentImgToExtract.size().height)
    {
        _hasAsyncError = true;
        _exception     = SENSCalibrationEstimatorException("Image size changed during capturing process!",
                                                       __LINE__,
                                                       __FILE__);
        return false;
    }

    bool foundPrecisely = false;
    try
    {
        std::vector<cv::Point2f> preciseCorners2D;
        int                      flags          = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
        bool                     foundPrecisely = cv::findChessboardCorners(_currentImgToExtract,
                                                        _boardSize,
                                                        preciseCorners2D,
                                                        flags);

        if (foundPrecisely)
        {
            cv::cornerSubPix(_currentImgToExtract,
                             preciseCorners2D,
                             cv::Size(11, 11),
                             cv::Size(-1, -1),
                             TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,
                                          30,
                                          0.0001));

            //add detected points
            _imagePoints.push_back(preciseCorners2D);
            _numCaptured++;
        }
    }
    catch (std::exception& e)
    {
        _hasAsyncError = true;
        _exception     = SENSCalibrationEstimatorException(e.what(), __LINE__, __FILE__);
        return false;
    }
    catch (...)
    {
        _hasAsyncError = true;
        _exception     = SENSCalibrationEstimatorException("Unknown exception during calibration!", __LINE__, __FILE__);
        return false;
    }

    return foundPrecisely;
}
//-----------------------------------------------------------------------------
bool SENSCalibrationEstimator::calibrateAsync()
{
    bool ok = false;
    try
    {
        _numCaptured = 0;
        std::vector<cv::Mat> rvecs, tvecs;
        vector<float>        reprojErrs;
        cv::Mat              cameraMat;
        cv::Mat              distortion;

        ok = calcCalibration(_imageSize,
                             cameraMat,
                             distortion,
                             _imagePoints,
                             rvecs,
                             tvecs,
                             reprojErrs,
                             _reprojectionError,
                             _boardSize,
                             _boardSquareMM,
                             _params.calibrationFlags(),
                             _params.useReleaseObjectMethod);
        //correct number of caputured, extraction may have failed
        if (!rvecs.empty() || !reprojErrs.empty())
            _numCaptured = (int)std::max(rvecs.size(), reprojErrs.size());
        else
            _numCaptured = 0;

        if (ok)
        {
            //instantiate calibration
            _calibration = std::make_unique<SENSCalibration>(cameraMat,
                                                             distortion,
                                                             _imageSize,
                                                             _boardSize,
                                                             _boardSquareMM,
                                                             _reprojectionError,
                                                             _numCaptured,
                                                             Utils::getDateTime2String(),
                                                             _camSizeIndex,
                                                             _mirroredH,
                                                             _mirroredV,
                                                             _camType,
                                                             _computerInfos,
                                                             _params.calibrationFlags(),
                                                             true);
        }
    }
    catch (std::exception& e)
    {
        _hasAsyncError = true;
        _exception     = SENSCalibrationEstimatorException(e.what(), __LINE__, __FILE__);
        return false;
    }
    catch (...)
    {
        _hasAsyncError = true;
        _exception     = SENSCalibrationEstimatorException("Unknown exception during calibration!", __LINE__, __FILE__);
        return false;
    }

    return ok;
}
//-----------------------------------------------------------------------------
//! Calculates the calibration with the given set of image points
bool SENSCalibrationEstimator::calcCalibration(cv::Size&                          imageSize,
                                               cv::Mat&                           cameraMatrix,
                                               cv::Mat&                           distCoeffs,
                                               const vector<vector<cv::Point2f>>& imagePoints,
                                               std::vector<cv::Mat>&              rvecs,
                                               std::vector<cv::Mat>&              tvecs,
                                               vector<float>&                     reprojErrs,
                                               float&                             totalAvgErr,
                                               cv::Size&                          boardSize,
                                               float                              squareSize,
                                               int                                flag,
                                               bool                               useReleaseObjectMethod)
{
    // Init camera matrix with the eye setter
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    // We need to set eleme at 0,0 to 1 if we want a fix aspect ratio
    if (flag & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = 1.0;

    // init the distortion coeffitients to zero
    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    vector<vector<cv::Point3f>> objectPoints(1);

    SENSCalibrationEstimator::calcBoardCorners3D(boardSize,
                                                 squareSize,
                                                 objectPoints[0]);

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    ////////////////////////////////////////////////
    //Find intrinsic and extrinsic camera parameters
    int iFixedPoint = -1;
    if (useReleaseObjectMethod)
        iFixedPoint = boardSize.width - 1;
#if 0
    double rms = cv::calibrateCameraRO(objectPoints,
                                       imagePoints,
                                       imageSize,
                                       iFixedPoint,
                                       cameraMatrix,
                                       distCoeffs,
                                       rvecs,
                                       tvecs,
                                       cv::noArray(),
                                       flag);
#else
    double rms = cv::calibrateCamera(objectPoints,
                                     imagePoints,
                                     imageSize,
                                     //iFixedPoint,
                                     cameraMatrix,
                                     distCoeffs,
                                     rvecs,
                                     tvecs,
                                     //cv::noArray(),
                                     flag);
#endif
    ////////////////////////////////////////////////

    Utils::log("SLProject", "Re-projection error reported by calibrateCamera: %f", rms);

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
double SENSCalibrationEstimator::calcReprojectionErrors(const vector<vector<cv::Point3f>>& objectPoints,
                                                        const vector<vector<cv::Point2f>>& imagePoints,
                                                        const std::vector<cv::Mat>&        rvecs,
                                                        const std::vector<cv::Mat>&        tvecs,
                                                        const cv::Mat&                     cameraMatrix,
                                                        const cv::Mat&                     distCoeffs,
                                                        vector<float>&                     perViewErrors)
{
    std::vector<cv::Point2f> imagePoints2;
    size_t                   totalPoints = 0;
    double                   totalErr    = 0, err;
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
bool SENSCalibrationEstimator::loadCalibParams()
{
    FileStorage fs;
    string      fullCalibIniFile = Utils::findFile(_calibParamsFileName,
                                                   {_calibDataPath, _exePath});
    fs.open(fullCalibIniFile, FileStorage::READ);
    if (!fs.isOpened())
    {
        Utils::log("SLProject", "Could not open the calibration parameter file: %s", fullCalibIniFile.c_str());
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
void SENSCalibrationEstimator::saveImage(cv::Mat imageGray)
{
    std::stringstream ss;
    ss << _calibImgOutputDir << "CalibImge_" << Utils::getDateTime2String() << ".jpg";
    cv::imwrite(ss.str(), imageGray);
}
//-----------------------------------------------------------------------------
void SENSCalibrationEstimator::updateExtractAndCalc(bool found, bool grabFrame, cv::Mat imageGray)
{
    switch (_state)
    {
        case State::Streaming:
        {
            if (grabFrame && found)
            {
                _currentImgToExtract = imageGray.clone();
                //start async extraction
                if (!_calibrationTask.valid())
                {
                    _calibrationTask = std::async(std::launch::async, &SENSCalibrationEstimator::extractAsync, this);
                }

                _state = State::BusyExtracting;
            }
            break;
        }
        case State::BusyExtracting:
        {
            //check if async task is ready
            if (_calibrationTask.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
            {
                bool extractionSuccessful = _calibrationTask.get();

                if (_hasAsyncError)
                {
                    _state = State::Error;
                    throw _exception;
                }
                else if (_numCaptured >= _numOfImgsToCapture)
                {
                    //if ready and number of capturings exceed number of required start calculation
                    _calibrationTask = std::async(std::launch::async, &SENSCalibrationEstimator::calibrateAsync, this);
                    _state           = State::Calculating;
                }
                else
                {
                    _state = State::Streaming;
                }
            }
            break;
        }
        case State::Calculating:
        {
            if (_calibrationTask.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
            {
                _calibrationSuccessful = _calibrationTask.get();

                if (_calibrationSuccessful)
                {
                    _state = State::Done;
                    Utils::log("SLProject", "Calibration succeeded.");
                    Utils::log("SLProject", "Reproj. error: %f", _reprojectionError);
                }
                else
                {
                    Utils::log("SLProject", "Calibration failed.");
                    if (_hasAsyncError)
                    {
                        _state = State::Error;
                        throw _exception;
                    }
                    else
                        _state = State::Done;
                }
            }
            break;
        }
        default: break;
    }
}
//-----------------------------------------------------------------------------
void SENSCalibrationEstimator::updateOnlyCapture(bool found, bool grabFrame, cv::Mat imageGray)
{
    switch (_state)
    {
        case State::Streaming:
        {
            if (grabFrame && found)
            {
                saveImage(imageGray);
                _numCaptured++;
            }

            if (_numCaptured >= _numOfImgsToCapture)
            {
                _state                 = State::DoneCaptureAndSave;
                _calibrationSuccessful = true;
            }
            break;
        }
        default: break;
    }
}
//-----------------------------------------------------------------------------
//!< Finds the inner chessboard corners in the given image
bool SENSCalibrationEstimator::updateAndDecorate(cv::Mat        imageColor,
                                                 const cv::Mat& imageGray,
                                                 bool           grabFrame,
                                                 bool           drawCorners)
{
    assert(!imageGray.empty() &&
           "SENSCalibrationEstimator::findChessboard: imageGray is empty!");
    assert(!imageColor.empty() &&
           "SENSCalibrationEstimator::findChessboard: imageColor is empty!");
    assert(_boardSize.width && _boardSize.height &&
           "SENSCalibrationEstimator::findChessboard: _boardSize is not set!");

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

    std::vector<cv::Point2f> corners2D;
    bool                     found = cv::findChessboardCorners(imageGrayExtract,
                                           _boardSize,
                                           corners2D,
                                           cv::CALIB_CB_FAST_CHECK);

    if (found)
    {
        if (grabFrame && _state == State::Streaming)
        {
            //simulate a snapshot
            cv::bitwise_not(imageColor, imageColor);
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
                                      cv::Mat(corners2D),
                                      found);
        }
    }

    if (_params.mode == SENSCalibrationEstimatorParams::EstimatorMode::ExtractAndCalculate)
    {
        //update state machine for extraction and calculation
        updateExtractAndCalc(found, grabFrame, imageGray);
    }
    else // SENSCalibrationEstimatorParams::EstimatorMode::OnlyCaptureAndSave
    {
        updateOnlyCapture(found, grabFrame, imageGray);
    }

    return found;
}
//-----------------------------------------------------------------------------
//! Calculates the 3D positions of the chessboard corners
void SENSCalibrationEstimator::calcBoardCorners3D(const cv::Size&           boardSize,
                                                  float                     squareSize,
                                                  std::vector<cv::Point3f>& objectPoints3D)
{
    // Because OpenCV image coords are top-left we define the according
    // 3D coords also top-left.
    objectPoints3D.clear();
    for (int y = boardSize.height - 1; y >= 0; --y)
        for (int x = 0; x < boardSize.width; ++x)
            objectPoints3D.push_back(cv::Point3f((float)x * squareSize,
                                                 (float)y * squareSize,
                                                 0));
}
