//#############################################################################
//  File:      CVCalibration.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVCalibrationEstimator.h>
#include <CVCalibration.h>
#include <Utils.h>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
CVCalibrationEstimator::CVCalibrationEstimator(CVCalibrationEstimatorParams params,
                                               int                          camSizeIndex,
                                               bool                         mirroredH,
                                               bool                         mirroredV,
                                               CVCameraType                 camType,
                                               std::string                  computerInfos,
                                               std::string                  calibDataPath,
                                               std::string                  imageOutputPath)
  : _params(params),
    _camSizeIndex(camSizeIndex),
    _mirroredH(mirroredH),
    _mirroredV(mirroredV),
    _camType(camType),
    _calibration(_camType, ""),
    _calibParamsFileName("calib_in_params.yml"),
    _exception("Undefined error", 0, __FILE__),
    _computerInfos(computerInfos),
    _calibDataPath(calibDataPath)
{
    if (!loadCalibParams())
    {
        throw CVCalibrationEstimatorException("Could not load calibration parameter!",
                                              __LINE__,
                                              __FILE__);
    }

    if (params.mode == CVCalibrationEstimatorParams::EstimatorMode::OnlyCaptureAndSave)
    {
        if (!Utils::dirExists(imageOutputPath))
        {
            std::stringstream ss;
            ss << "Image output directory does not exist: " << imageOutputPath;
            throw CVCalibrationEstimatorException(ss.str(),
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
                throw CVCalibrationEstimatorException(ss.str(),
                                                      __LINE__,
                                                      __FILE__);
            }
        }
    }
}
//-----------------------------------------------------------------------------
CVCalibrationEstimator::~CVCalibrationEstimator()
{
    //wait for the async task to finish
    if (_calibrationTask.valid())
        _calibrationTask.wait();
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
            Utils::log("Calibration succeeded.");
            Utils::log("Reproj. error: %f\n", _reprojectionError);
        }
        else
        {
            Utils::log("Calibration failed.");
        }
    }

    return calibrationSuccessful;
}
//-----------------------------------------------------------------------------
bool CVCalibrationEstimator::extractAsync()
{
    if (_imageSize.width == 0 && _imageSize.height == 0)
        _imageSize = _currentImgToExtract.size();
    else if (_imageSize.width != _currentImgToExtract.size().width || _imageSize.height != _currentImgToExtract.size().height)
    {
        _hasAsyncError = true;
        _exception     = CVCalibrationEstimatorException("Image size changed during capturing process!",
                                                     __LINE__,
                                                     __FILE__);
        return false;
    }

    bool foundPrecisely = false;
    try
    {
        CVVPoint2f preciseCorners2D;
        int        flags          = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
        bool       foundPrecisely = cv::findChessboardCorners(_currentImgToExtract,
                                                        _boardSize,
                                                        preciseCorners2D,
                                                        flags);

        if (foundPrecisely)
        {
            cv::cornerSubPix(_currentImgToExtract,
                             preciseCorners2D,
                             CVSize(11, 11),
                             CVSize(-1, -1),
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
        _exception     = CVCalibrationEstimatorException(e.what(), __LINE__, __FILE__);
        return false;
    }
    catch (...)
    {
        _hasAsyncError = true;
        _exception     = CVCalibrationEstimatorException("Unknown exception during calibration!", __LINE__, __FILE__);
        return false;
    }

    return foundPrecisely;
}
//-----------------------------------------------------------------------------
bool CVCalibrationEstimator::calibrateAsync()
{
    bool ok = false;
    try
    {
        _numCaptured = 0;
        CVVMat        rvecs, tvecs;
        vector<float> reprojErrs;
        cv::Mat       cameraMat;
        cv::Mat       distortion;

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
            _calibration = CVCalibration(cameraMat,
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
                                         _params.calibrationFlags());
        }
    }
    catch (std::exception& e)
    {
        _hasAsyncError = true;
        _exception     = CVCalibrationEstimatorException(e.what(), __LINE__, __FILE__);
        return false;
    }
    catch (...)
    {
        _hasAsyncError = true;
        _exception     = CVCalibrationEstimatorException("Unknown exception during calibration!", __LINE__, __FILE__);
        return false;
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
                                             int                flag,
                                             bool               useReleaseObjectMethod)
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
    int iFixedPoint = -1;
    if (useReleaseObjectMethod)
        iFixedPoint = boardSize.width - 1;
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
    string      fullCalibIniFile = _calibDataPath + _calibParamsFileName;

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
void CVCalibrationEstimator::saveImage(cv::Mat imageGray)
{
    std::stringstream ss;
    ss << _calibImgOutputDir << "CalibImge_" << Utils::getDateTime2String() << ".jpg";
    cv::imwrite(ss.str(), imageGray);
}
//-----------------------------------------------------------------------------
void CVCalibrationEstimator::updateExtractAndCalc(bool found, bool grabFrame, cv::Mat imageGray)
{
    switch (_state)
    {
        case State::Streaming: {
            if (grabFrame && found)
            {
                _currentImgToExtract = imageGray.clone();
                //start async extraction
                if (!_calibrationTask.valid())
                {
                    _calibrationTask = std::async(std::launch::async, &CVCalibrationEstimator::extractAsync, this);
                }

                _state = State::BusyExtracting;
            }
            break;
        }
        case State::BusyExtracting: {
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
                    _calibrationTask = std::async(std::launch::async, &CVCalibrationEstimator::calibrateAsync, this);
                    _state           = State::Calculating;
                }
                else
                {
                    _state = State::Streaming;
                }
            }
            break;
        }
        case State::Calculating: {
            if (_calibrationTask.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
            {
                _calibrationSuccessful = _calibrationTask.get();

                if (_calibrationSuccessful)
                {
                    _state = State::Done;
                    Utils::log("Calibration succeeded.");
                    Utils::log("Reproj. error: %f\n", _reprojectionError);
                }
                else
                {
                    Utils::log("Calibration failed.");
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
    }
}
//-----------------------------------------------------------------------------
void CVCalibrationEstimator::updateOnlyCapture(bool found, bool grabFrame, cv::Mat imageGray)
{
    switch (_state)
    {
        case State::Streaming: {
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
    }
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
                                      CVMat(corners2D),
                                      found);
        }
    }

    if (_params.mode == CVCalibrationEstimatorParams::EstimatorMode::ExtractAndCalculate)
    {
        //update state machine for extraction and calculation
        updateExtractAndCalc(found, grabFrame, imageGray);
    }
    else // CVCalibrationEstimatorParams::EstimatorMode::OnlyCaptureAndSave
    {
        updateOnlyCapture(found, grabFrame, imageGray);
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
