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

#include <CVCalibration.h>
#include <Utils.h>
#include <ftplib.h>
#include <algorithm> // std::max
#include <SLApplication.h>

using namespace cv;
using namespace std;
//-----------------------------------------------------------------------------
off64_t ftpUploadSizeMax = 0;
//-----------------------------------------------------------------------------
//! Calibration Upload callback for progress feedback
int ftpCallbackUpload(off64_t xfered, void* arg)
{
    if (ftpUploadSizeMax)
    {
        int xferedPC = (int)((float)xfered / (float)ftpUploadSizeMax * 100.0f);
        cout << "Bytes saved: " << xfered << " (" << xferedPC << ")" << endl;
        //SLApplication::jobProgressNum(xferedPC);
    }
    else
    {
        cout << "Bytes saved: " << xfered << endl;
    }
    return xfered ? 1 : 0;
}

//-----------------------------------------------------------------------------
//! Default path for calibration files
//! Is overwritten in slCreateAppAndScene.
string CVCalibration::calibIniPath;
//-----------------------------------------------------------------------------
//! FTP credentials for calibration up- and download
const string CVCalibration::_FTP_HOST = "pallas.bfh.ch:21";
const string CVCalibration::_FTP_USER = "upload";
const string CVCalibration::_FTP_PWD  = "FaAdbD3F2a";
const string CVCalibration::_FTP_DIR  = "calibrations";
//-----------------------------------------------------------------------------
//! Increase the _CALIBFILEVERSION each time you change the file format
// Version 6, Date: 6.JUL.2019: Added device parameter from Android
const int CVCalibration::_CALIBFILEVERSION = 6;
//-----------------------------------------------------------------------------
CVCalibration::CVCalibration()
  : _state(CS_uncalibrated),
    _cameraFovHDeg(0.0f),
    _cameraFovVDeg(0.0f),
    _calibFileName(""), // is set in load
    _calibParamsFileName("calib_in_params.yml"),
    _calibFixPrincipalPoint(false),
    _calibFixAspectRatio(false),
    _calibZeroTangentDist(false),
    _calibRationalModel(false),
    _calibTiltedModel(false),
    _calibThinPrismModel(false),
    _boardSize(0, 0),
    _boardSquareMM(0.0f),
    _numOfImgsToCapture(0),
    _numCaptured(0),
    _reprojectionError(-1.0f),
    _camSizeIndex(-1),
    _showUndistorted(false),
    _calibrationTime("-"),
    _devFocalLength(0.0f),
    _devSensorSizeW(0.0f),
    _devSensorSizeH(0.0f),
    _isMirroredH(false),
    _isMirroredV(false)
{
}
//-----------------------------------------------------------------------------
//! Resets the calibration to the uncalibrated state
void CVCalibration::clear()
{
    _numCaptured       = 0;
    _reprojectionError = -1.0f;
    _imagePoints.clear();
    _cameraFovHDeg   = 0.0f;
    _cameraFovVDeg   = 0.0f;
    _calibrationTime = "-";
    _undistortMapX.release();
    _undistortMapY.release();
    _state         = CS_uncalibrated;
    _computerInfos = SLApplication::getComputerInfos();
    _calibrationImgs.clear();
}
//-----------------------------------------------------------------------------
//! Loads the calibration information from the config file
bool CVCalibration::load(const string& calibDir,
                         const string& calibFileName,
                         bool          mirrorHorizontally,
                         bool          mirrorVertically)
{
    _calibDir      = Utils::unifySlashes(calibDir);
    _calibFileName = calibFileName;
    _isMirroredH   = mirrorHorizontally;
    _isMirroredV   = mirrorVertically;

    //load camera parameter
    string fullPathAndFilename = _calibDir + _calibFileName;

    // try to download from ftp if no calibration exists locally
    if (!Utils::fileExists(fullPathAndFilename))
        downloadCalibration(fullPathAndFilename);

    // try to open the local calibration file
    FileStorage fs(fullPathAndFilename, FileStorage::READ);
    if (!fs.isOpened())
    {
        Utils::log("Calibration     : %s\n", calibFileName.c_str());
        Utils::log("Calib. created  : No. Calib. will be estimated\n");
        _numCaptured       = 0;
        _isMirroredH       = mirrorHorizontally;
        _isMirroredV       = mirrorVertically;
        _reprojectionError = 0;
        _calibrationTime   = "-";
        _state             = CS_uncalibrated;
        _camSizeIndex      = -1;
        return false;
    }

    // Reset if new file format version is available
    int calibFileVersion = 0;
    fs["CALIBFILEVERSION"] >> calibFileVersion;
    if (calibFileVersion < _CALIBFILEVERSION)
    {
        _numCaptured            = 0;
        _isMirroredH            = mirrorHorizontally;
        _isMirroredV            = mirrorVertically;
        _calibFixAspectRatio    = true;
        _calibFixPrincipalPoint = true;
        _calibZeroTangentDist   = true;
        _calibRationalModel     = false;
        _calibTiltedModel       = false;
        _calibThinPrismModel    = false;
        _reprojectionError      = -1;
        _calibrationTime        = "-";
        _state                  = CS_uncalibrated;
        _camSizeIndex           = -1;
    }
    else
    {
        fs["imageSizeWidth"] >> _imageSize.width;
        fs["imageSizeHeight"] >> _imageSize.height;
        fs["numCaptured"] >> _numCaptured;
        fs["isMirroredH"] >> _isMirroredH;
        fs["isMirroredV"] >> _isMirroredV;
        fs["calibFixAspectRatio"] >> _calibFixAspectRatio;
        fs["calibFixPrincipalPoint"] >> _calibFixPrincipalPoint;
        fs["calibZeroTangentDist"] >> _calibZeroTangentDist;
        fs["calibRationalModel"] >> _calibRationalModel;
        fs["calibTiltedModel"] >> _calibTiltedModel;
        fs["calibThinPrismModel"] >> _calibThinPrismModel;
        fs["cameraMat"] >> _cameraMat;
        fs["distortion"] >> _distortion;
        fs["reprojectionError"] >> _reprojectionError;
        fs["calibrationTime"] >> _calibrationTime;
        fs["camSizeIndex"] >> _camSizeIndex;
        _state = _numCaptured ? CS_calibrated : CS_uncalibrated;
    }

    //estimate computer infos
    if (!fs["computerInfos"].empty())
        fs["computerInfos"] >> _computerInfos;
    else
    {
        std::vector<std::string> stringParts;
        Utils::splitString(Utils::getFileNameWOExt(_calibFileName), '_', stringParts);
        if (stringParts.size() >= 3)
            _computerInfos = stringParts[1];
    }

    // close the input file
    fs.release();

    //calculate FOV and undistortion maps
    if (_state == CS_calibrated)
    {
        //calcCameraFov();
        calculateUndistortedCameraMat();
        calcCameraFovFromUndistortedCameraMat();
        buildUndistortionMaps();
    }

    Utils::log("Calib. loaded  : %s\n", fullPathAndFilename.c_str());
    Utils::log("Calib. created : %s\n", _calibrationTime.c_str());
    Utils::log("Camera FOV H/V : %3.1f/%3.1f\n", _cameraFovVDeg, _cameraFovHDeg);

    return true;
}
//-----------------------------------------------------------------------------
//! Saves the camera calibration parameters to the config file
void CVCalibration::save(std::string forceSavePath)
{
    string fullPathAndFilename;
    if (forceSavePath.empty())
        fullPathAndFilename = _calibDir + _calibFileName;
    else
        fullPathAndFilename = forceSavePath;

    cv::FileStorage fs(fullPathAndFilename, FileStorage::WRITE);

    if (!fs.isOpened())
    {
        Utils::log("Failed to write calib. %s\n", fullPathAndFilename.c_str());
        return;
    }

    char buf[1024];
    sprintf(buf,
            "flags:%s%s%s%s%s%s%s",
            _calibFlags & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
            _calibFlags & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
            _calibFlags & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
            _calibFlags & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "",
            _calibFlags & CALIB_RATIONAL_MODEL ? " +rational_model" : "",
            _calibFlags & CALIB_THIN_PRISM_MODEL ? " +thin_prism_model" : "",
            _calibFlags & CALIB_TILTED_MODEL ? " +tilted_model" : "");
    fs.writeComment(buf, 0);

    fs << "CALIBFILEVERSION" << _CALIBFILEVERSION;
    fs << "calibrationTime" << _calibrationTime;
    fs << "imageSizeWidth" << _imageSize.width;
    fs << "imageSizeHeight" << _imageSize.height;
    fs << "boardSizeWidth" << _boardSize.width;   // do not reload
    fs << "boardSizeHeight" << _boardSize.height; // do not reload
    fs << "boardSquareMM" << _boardSquareMM;      // do not reload
    fs << "numCaptured" << _numCaptured;
    fs << "calibFlags" << _calibFlags;
    fs << "isMirroredH" << _isMirroredH;
    fs << "isMirroredV" << _isMirroredV;
    fs << "calibFixAspectRatio" << _calibFixAspectRatio;
    fs << "calibFixPrincipalPoint" << _calibFixPrincipalPoint;
    fs << "calibZeroTangentDist" << _calibZeroTangentDist;
    fs << "calibRationalModel" << _calibRationalModel;
    fs << "calibTiltedModel" << _calibTiltedModel;
    fs << "calibThinPrismModel" << _calibThinPrismModel;
    fs << "cameraMat" << _cameraMat;
    fs << "distortion" << _distortion;
    fs << "reprojectionError" << _reprojectionError;
    fs << "cameraFovVDeg" << _cameraFovVDeg;
    fs << "cameraFovHDeg" << _cameraFovHDeg;
    fs << "camSizeIndex" << _camSizeIndex;
    fs << "DeviceLensFocalLength" << _devFocalLength;
    fs << "DeviceSensorPhysicalSizeW" << _devSensorSizeW;
    fs << "DeviceSensorPhysicalSizeH" << _devSensorSizeW;
    fs << "computerInfos" << _computerInfos;
    /*
    SLGLState* stateGL = SLGLState::instance();
    fs << "computerUser" << SLApplication::computerUser;
    fs << "computerName" << SLApplication::computerName;
    fs << "computerBrand" << SLApplication::computerBrand;
    fs << "computerArch" << SLApplication::computerArch;
    fs << "computerOS" << SLApplication::computerOS;
    fs << "computerOSVer" << SLApplication::computerOSVer;
    fs << "OpenGLVersion" << stateGL->glVersionNO();
    fs << "OpenGLVendor" << stateGL->glVendor();
    fs << "OpenGLRenderer" << stateGL->glRenderer();
    fs << "GLSLVersion" << stateGL->glSLVersionNO();
    fs << "SLProjectVersion" << SLApplication::version;
    fs << "DeviceLensAperture" << SLApplication::deviceParameter["DeviceLensAperture"];
    fs << "DeviceLensFocusDistanceCalibration" << SLApplication::deviceParameter["DeviceLensFocusDistanceCalibration"];
    fs << "DeviceLensMinimumFocusDistance" << SLApplication::deviceParameter["DeviceLensMinimumFocusDistance"];
    fs << "DeviceSensorActiveArraySizeW" << SLApplication::deviceParameter["DeviceSensorActiveArraySizeW"];
    fs << "DeviceSensorActiveArraySizeH" << SLApplication::deviceParameter["DeviceSensorActiveArraySizeH"];
    */

    // close file
    fs.release();
    Utils::log("Calib. saved    : %s\n", fullPathAndFilename.c_str());
    uploadCalibration(fullPathAndFilename);
}
//-----------------------------------------------------------------------------
//! Loads the chessboard calibration pattern parameters
bool CVCalibration::loadCalibParams()
{
    FileStorage fs;
    string      fullCalibIniFile = calibIniPath + _calibParamsFileName;

    fs.open(fullCalibIniFile, FileStorage::READ);
    if (!fs.isOpened())
    {
        Utils::log("Could not open the calibration parameter file: %s\n", fullCalibIniFile.c_str());
        _state = CS_uncalibrated;
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
//! Calculates the vertical field of view angle in degrees
void CVCalibration::calcCameraFovFromUndistortedCameraMat()
{
    if (_cameraMatUndistorted.rows != 3 || _cameraMatUndistorted.cols != 3)
        Utils::exitMsg("CVCalibration::calcCameraFovFromSceneCameraMat: No _cameraMatUndistorted available",
                       __LINE__,
                       __FILE__);

    //calculate vertical field of view
    float fx       = (float)_cameraMatUndistorted.at<double>(0, 0);
    float fy       = (float)_cameraMatUndistorted.at<double>(1, 1);
    float cx       = (float)_cameraMatUndistorted.at<double>(0, 2);
    float cy       = (float)_cameraMatUndistorted.at<double>(1, 2);
    _cameraFovHDeg = 2.0f * (float)atan2(cx, fx) * Utils::RAD2DEG;
    _cameraFovVDeg = 2.0f * (float)atan2(cy, fy) * Utils::RAD2DEG;
}

//-----------------------------------------------------------------------------
//! Calculates the 3D positions of the chessboard corners
void CVCalibration::calcBoardCorners3D(const CVSize& boardSize,
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
//-----------------------------------------------------------------------------
//! Calculates the reprojection error of the calibration
float CVCalibration::calcReprojectionErr(const CVVVPoint3f& objectPoints,
                                         const CVVMat&      rvecs,
                                         const CVVMat&      tvecs,
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
                          _cameraMat,
                          _distortion,
                          imagePoints2);

        err = norm(_imagePoints[i], imagePoints2, NORM_L2);

        size_t n         = objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return (float)std::sqrt(totalErr / totalPoints);
}
//-----------------------------------------------------------------------------
//!< Finds the inner chessboard corners in the given image
bool CVCalibration::findChessboard(CVMat        imageColor,
                                   const CVMat& imageGray,
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

    _imageSize = imageColor.size();

    cv::Mat imageGrayExtract = imageGray;
    //resize image so that we get fluent caputure workflow for high resolutions
    double scale              = 1.0;
    bool   doScale            = false;
    int    targetExtractWidth = 640;
    if (_imageSize.width > targetExtractWidth)
    {
        doScale = true;
        scale   = (double)_imageSize.width / (double)targetExtractWidth;
        cv::resize(imageGray, imageGrayExtract, cv::Size(), 1 / scale, 1 / scale);
    }

    CVVPoint2f corners2D;
    bool       found = cv::findChessboardCorners(imageGrayExtract,
                                           _boardSize,
                                           corners2D,
                                           cv::CALIB_CB_FAST_CHECK);

    if (found)
    {
        if (_state == CS_calibrateGrab)
        {
            //save a copy of this image
            _calibrationImgs.push_back(imageGray.clone());
            //increase number of capturings
            _numCaptured++;

            //simulate a snapshot
            cv::bitwise_not(imageColor, imageColor);
            _state = CS_calibrateStream;
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
//! Calculates the reprojection error of the calibration
static double calcReprojectionErrors(const CVVVPoint3f& objectPoints,
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
//! Calculates the calibration with the given set of image points
static bool calcCalibration(CVSize&            imageSize,
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

    CVCalibration::calcBoardCorners3D(boardSize,
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
bool CVCalibration::calibrateAsync()
{
    _state         = CS_startCalculating;
    _computerInfos = SLApplication::getComputerInfos();

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

    _calibFlags = 0;
    if (_calibFixPrincipalPoint) _calibFlags |= CALIB_FIX_PRINCIPAL_POINT;
    if (_calibZeroTangentDist) _calibFlags |= CALIB_ZERO_TANGENT_DIST;
    if (_calibFixAspectRatio) _calibFlags |= CALIB_FIX_ASPECT_RATIO;
    if (_calibRationalModel) _calibFlags |= CALIB_RATIONAL_MODEL;
    if (_calibTiltedModel) _calibFlags |= CALIB_TILTED_MODEL;
    if (_calibThinPrismModel) _calibFlags |= CALIB_THIN_PRISM_MODEL;
    bool ok = calcCalibration(_imageSize,
                              _cameraMat,
                              _distortion,
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
        calculateUndistortedCameraMat();
        calcCameraFovFromUndistortedCameraMat();
        buildUndistortionMaps();
        _calibrationTime = Utils::getDateTime2String();
    }
    else
    {
        _cameraFovVDeg   = 0.0f;
        _cameraFovHDeg   = 0.0f;
        _calibrationTime = "-";
        _undistortMapX.release();
        _undistortMapY.release();
    }

    _calibrationImgs.clear();

    return ok;
}
//-----------------------------------------------------------------------------
//! Initiates the final calculation
bool CVCalibration::calculate()
{
    bool calibrationSuccessful = false;
    //if (!_calibrationTask.valid())
    //{
    //    _calibrationTask = std::async(std::launch::async, &CVCalibration::calibrateAsync, this);
    //}
    //else if (_calibrationTask.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
    //{
    //    calibrationSuccessful = _calibrationTask.get();
    //    if (calibrationSuccessful)
    //    {
    //        _state = CS_calibrated;
    //        save();
    //        Utils::log("Calibration succeeded.");
    //        Utils::log("Reproj. error: %f\n", _reprojectionError);
    //    }
    //    else
    //    {
    //        _state = CS_uncalibrated;
    //        Utils::log("Calibration failed.");
    //    }
    //}

    return calibrationSuccessful;
}
//-----------------------------------------------------------------------------
//! get inscribed and circumscribed rectangle
void getInnerAndOuterRectangles(const cv::Mat&    cameraMatrix,
                                const cv::Mat&    distCoeffs,
                                const cv::Mat&    R,
                                const cv::Mat&    newCameraMatrix,
                                cv::Size          imgSize,
                                cv::Rect_<float>& inner,
                                cv::Rect_<float>& outer)
{
    const int N = 9;
    // Fill matrix with N * N sampling points
    cv::Mat pts(N * N, 2, CV_32F);
    for (int y = 0, k = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            pts.at<float>(k, 0) = (float)x * imgSize.width / (N - 1);
            pts.at<float>(k, 1) = (float)y * imgSize.height / (N - 1);
            k++;
        }
    }

    pts = pts.reshape(2);
    cv::undistortPoints(pts, pts, cameraMatrix, distCoeffs, R, newCameraMatrix);
    pts = pts.reshape(1);

    float iX0 = -FLT_MAX, iX1 = FLT_MAX, iY0 = -FLT_MAX, iY1 = FLT_MAX;
    float oX0 = FLT_MAX, oX1 = -FLT_MAX, oY0 = FLT_MAX, oY1 = -FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for (int y = 0, k = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            cv::Point2f p = {pts.at<float>(k, 0), pts.at<float>(k, 1)};
            oX0           = MIN(oX0, p.x);
            oX1           = MAX(oX1, p.x);
            oY0           = MIN(oY0, p.y);
            oY1           = MAX(oY1, p.y);

            if (x == 0)
                iX0 = MAX(iX0, p.x);
            if (x == N - 1)
                iX1 = MIN(iX1, p.x);
            if (y == 0)
                iY0 = MAX(iY0, p.y);
            if (y == N - 1)
                iY1 = MIN(iY1, p.y);
            k++;
        }
    inner = cv::Rect_<float>(iX0, iY0, iX1 - iX0, iY1 - iY0);
    outer = cv::Rect_<float>(oX0, oY0, oX1 - oX0, oY1 - oY0);
}
//! Calculate a camera matrix that we use for the scene graph and for the reprojection of the undistored image
//! (This is a manipulated version of cv::getOptimalNewCameraMatrix but with equal focal lengths in x and y)
void CVCalibration::calculateUndistortedCameraMat()
{
    if (_cameraMat.rows != 3 || _cameraMat.cols != 3)
        Utils::exitMsg("CVCalibration::calculateUndistortedCameraMat: No intrinsic parameter available",
                       __LINE__,
                       __FILE__);

    // An alpha of 0 leads to no black borders
    // An alpha of 1 leads to black borders
    // (with alpha equaly zero the augmentation fits best)
    double alpha = 1.0;

    bool centerPrinciplePoint = true;
    if (centerPrinciplePoint)
    {
        //Attention: the principle point has to be centered because for the projection matrix we assume that image plane is "symmetrically arranged wrt the focal plane"
        //(see http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix)
        //bool centerPrinciplePoint = true;
        //_cameraMatUndistorted     = cv::getOptimalNewCameraMatrix(_cameraMat, _distortion, _imageSize, alpha, _imageSize, nullptr, centerPrinciplePoint);

        double cx0 = _cameraMat.at<double>(0, 2);
        double cy0 = _cameraMat.at<double>(1, 2);
        double cx  = (_imageSize.width) * 0.5;
        double cy  = (_imageSize.height) * 0.5;

        cv::Rect_<float> inner, outer;
        getInnerAndOuterRectangles(_cameraMat, _distortion, cv::Mat(), _cameraMat, _imageSize, inner, outer);
        double s0 = std::max(std::max(std::max((double)cx / (cx0 - inner.x), (double)cy / (cy0 - inner.y)),
                                      (double)cx / (inner.x + inner.width - cx0)),
                             (double)cy / (inner.y + inner.height - cy0));
        double s1 = std::min(std::min(std::min((double)cx / (cx0 - outer.x), (double)cy / (cy0 - outer.y)),
                                      (double)cx / (outer.x + outer.width - cx0)),
                             (double)cy / (outer.y + outer.height - cy0));
        double s  = s0 * (1 - alpha) + s1 * alpha;

        _cameraMatUndistorted = _cameraMat.clone();
        _cameraMatUndistorted.at<double>(0, 0) *= s;
        _cameraMatUndistorted.at<double>(1, 1) *= s;
        _cameraMatUndistorted.at<double>(0, 2) = cx;
        _cameraMatUndistorted.at<double>(1, 2) = cy;
    }
    else
    {
        _cameraMatUndistorted = cv::getOptimalNewCameraMatrix(_cameraMat, _distortion, _imageSize, alpha, _imageSize, nullptr, centerPrinciplePoint);
    }

    std::cout << "_cameraMatUndistorted: " << _cameraMatUndistorted << std::endl;
    std::cout << "_cameraMat: " << _cameraMat << std::endl;
}
//-----------------------------------------------------------------------------
//! Builds undistortion maps after calibration or loading
void CVCalibration::buildUndistortionMaps()
{
    if (_cameraMatUndistorted.rows != 3 || _cameraMatUndistorted.cols != 3)
        Utils::exitMsg("CVCalibration::buildUndistortionMaps: No _cameraMatUndistorted available",
                       __LINE__,
                       __FILE__);

    // Create undistortion maps
    _undistortMapX.release();
    _undistortMapY.release();

    cv::initUndistortRectifyMap(_cameraMat,
                                _distortion,
                                cv::Mat(), // Identity matrix R
                                _cameraMatUndistorted,
                                _imageSize,
                                CV_16SC2, //before we had CV_32FC1 but in all tutorials they use CV_16SC2.. is there a reason?
                                _undistortMapX,
                                _undistortMapY);

    if (_undistortMapX.empty() || _undistortMapY.empty())
        Utils::exitMsg("CVCalibration::buildUndistortionMaps failed.",
                       __LINE__,
                       __FILE__);
}
//-----------------------------------------------------------------------------
//! Undistorts the inDistorted image into the outUndistorted
void CVCalibration::remap(CVMat& inDistorted,
                          CVMat& outUndistorted)
{
    assert(!inDistorted.empty() &&
           "Input image is empty!");

    assert(!_undistortMapX.empty() &&
           !_undistortMapY.empty() &&
           "Undistortion Maps are empty!");

    cv::remap(inDistorted,
              outUndistorted,
              _undistortMapX,
              _undistortMapY,
              cv::INTER_LINEAR);
}
//-----------------------------------------------------------------------------
//! Calculates camera intrinsics from a guessed FOV angle
/*! Most laptop-, webcam- or mobile camera have a horizontal view angle or
so called field of view (FOV) of around 65 degrees. From this parameter we
can calculate the most important intrinsic parameter the focal length. All
other parameters are set as if the lens would be perfect: No lens distortion
and the view axis goes through the center of the image.
If the focal length and sensor size is provided by the device we deduce the
the fov from it.
*/
void CVCalibration::createFromGuessedFOV(int imageWidthPX,
                                         int imageHeightPX)
{
    // aspect ratio
    float withOverHeight = (float)imageWidthPX / (float)imageHeightPX;

    // average horizontal view angle in degrees
    float fovH = 65.0f;

    // the vertical fov is derived from the width because it could be cropped
    float fovV = fovH / withOverHeight;

    // overwrite if device lens and sensor information exist and are reasonable
    if (_devFocalLength > 0.0f && _devSensorSizeW > 0.0f && _devSensorSizeH > 0.0f)
    {
        float devFovH = 2.0f * atan(_devSensorSizeW / (2.0f * _devFocalLength)) * Utils::RAD2DEG;
        float devFovV = devFovH / withOverHeight;
        if (devFovH > 60.0f && devFovH < 70.0f)
        {
            fovH = devFovH;
            fovV = devFovV;
        }
    }

    // Create standard camera matrix
    // fx, fx, cx, cy are all in pixel values not mm
    // We asume that we have an ideal image sensor with square pixels
    // so that the focal length fx and fy are identical
    // See the OpenCV documentation for more details:
    // http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html

    float cx = (float)imageWidthPX * 0.5f;
    float cy = (float)imageHeightPX * 0.5f;
    float fx = cx / tanf(fovH * 0.5f * Utils::DEG2RAD);
    float fy = fx;

    _imageSize.width  = imageWidthPX;
    _imageSize.height = imageHeightPX;
    _cameraMat        = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    _distortion       = (Mat_<double>(5, 1) << 0, 0, 0, 0, 0); // No distortion
    _cameraFovHDeg    = fovH;
    _cameraFovVDeg    = fovV;
    _calibrationTime  = Utils::getDateTime2String();
    _state            = CS_guessed;
}
//-----------------------------------------------------------------------------
//! Adapts an allready calibrated camera to a new resolution
void CVCalibration::adaptForNewResolution(const CVSize& newSize)
{
    // allow adaptation only for calibrated cameras
    if (_state != CS_calibrated) return;

    // new center and focal length in pixels not mm
    float fx, fy, cy, cx;
    if (((float)newSize.width / (float)newSize.height) > ((float)_imageSize.width / (float)_imageSize.height))
    {
        float scaleFactor = (float)newSize.width / (float)_imageSize.width;

        fx                    = this->fx() * scaleFactor;
        fy                    = this->fy() * scaleFactor;
        float oldHeightScaled = _imageSize.height * scaleFactor;
        float heightDiff      = (oldHeightScaled - newSize.height) * 0.5f;

        cx = this->cx() * scaleFactor;
        cy = this->cy() * scaleFactor - heightDiff;
    }
    else
    {
        float scaleFactor    = (float)newSize.height / (float)_imageSize.height;
        fx                   = this->fx() * scaleFactor;
        fy                   = this->fy() * scaleFactor;
        float oldWidthScaled = _imageSize.width * scaleFactor;
        float widthDiff      = (oldWidthScaled - newSize.width) * 0.5f;

        cx = this->cx() * scaleFactor - widthDiff;
        cy = this->cy() * scaleFactor;
    }

    std::cout << "adaptForNewResolution: _cameraMat before: " << _cameraMat << std::endl;
    _cameraMat = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    //_distortion remains unchanged
    _calibrationTime = Utils::getDateTime2String();

    _imageSize.width  = newSize.width;
    _imageSize.height = newSize.height;

    std::cout << "adaptForNewResolution: _cameraMat after: " << _cameraMat << std::endl;

    calculateUndistortedCameraMat();
    calcCameraFovFromUndistortedCameraMat();
    buildUndistortionMaps();
    //save();
}
//-----------------------------------------------------------------------------
//! Uploads the active calibration to the ftp server
void CVCalibration::uploadCalibration(const string& fullPathAndFilename)
{
    if (!Utils::fileExists(fullPathAndFilename))
    {
        Utils::log("Calib. file doesn't exist: %s\n", fullPathAndFilename.c_str());
        return;
    }

    if (state() != CS_calibrated)
    {
        Utils::log("Camera is not calibrated.");
        return;
    }

    ftplib ftp;

    if (ftp.Connect(_FTP_HOST.c_str()))
    {
        if (ftp.Login(_FTP_USER.c_str(), _FTP_PWD.c_str()))
        {
            if (ftp.Chdir(_FTP_DIR.c_str()))
            {
                // Get the latest calibration filename on the ftp
                string latestCalibFile = getLatestCalibFilename(ftp, fullPathAndFilename);

                // Set the calibfile version
                int versionNO = 0;
                if (!latestCalibFile.empty())
                {
                    versionNO = getVersionInCalibFilename(latestCalibFile);
                }

                // Increase the version
                versionNO++;
                stringstream versionSS;
                versionSS << "(" << versionNO << ")";
                versionSS.str();

                // Build new filename on ftp with version number
                string fileWOExt          = Utils::getFileNameWOExt(fullPathAndFilename);
                string newVersionFilename = fileWOExt + versionSS.str() + ".xml";

                // Upload
                if (!ftp.Put(fullPathAndFilename.c_str(),
                             newVersionFilename.c_str(),
                             ftplib::transfermode::image))
                    Utils::log("*** ERROR: ftp.Put failed. ***\n");
            }
            else
                Utils::log("*** ERROR: ftp.Chdir failed. ***\n");
        }
        else
            Utils::log("*** ERROR: ftp.Login failed. ***\n");
    }
    else
        Utils::log("*** ERROR: ftp.Connect failed. ***\n");

    ftp.Quit();
}
//-----------------------------------------------------------------------------
//! Uploads the active calibration to the ftp server
void CVCalibration::downloadCalibration(const string& fullPathAndFilename)
{
    ftplib ftp;

    if (ftp.Connect(_FTP_HOST.c_str()))
    {
        if (ftp.Login(_FTP_USER.c_str(), _FTP_PWD.c_str()))
        {
            if (ftp.Chdir(_FTP_DIR.c_str()))
            {
                // Get the latest calibration filename on the ftp
                string latestCalibFile = getLatestCalibFilename(ftp, fullPathAndFilename);
                int    remoteSize      = 0;
                ftp.Size(latestCalibFile.c_str(),
                         &remoteSize,
                         ftplib::transfermode::image);

                if (remoteSize > 0)
                {
                    string targetFilename = Utils::getFileName(fullPathAndFilename);
                    if (!ftp.Get(fullPathAndFilename.c_str(),
                                 latestCalibFile.c_str(),
                                 ftplib::transfermode::image))
                        Utils::log("*** ERROR: ftp.Get failed. ***\n");
                }
                else
                    Utils::log("*** No calibration to download ***\n");
            }
            else
                Utils::log("*** ERROR: ftp.Chdir failed. ***\n");
        }
        else
            Utils::log("*** ERROR: ftp.Login failed. ***\n");
    }
    else
        Utils::log("*** ERROR: ftp.Connect failed. ***\n");

    ftp.Quit();
}
//-----------------------------------------------------------------------------
//! Returns the latest calibration filename of the same fullPathAndFilename
string CVCalibration::getLatestCalibFilename(ftplib&       ftp,
                                             const string& fullPathAndFilename)
{
    // Get a list of calibrations of the same device
    string dirResult         = _calibDir + "dirResult.txt";
    string filenameWOExt     = Utils::getFileNameWOExt(fullPathAndFilename);
    string filenameWOExtStar = filenameWOExt + "*";

    // Get result of ftp.Dir into the textfile dirResult
    if (ftp.Dir(dirResult.c_str(), filenameWOExtStar.c_str()))
    {
        vector<string> vecFilesInDir;
        vector<string> strippedFiles;

        if (Utils::getFileContent(dirResult, vecFilesInDir))
        {
            for (string& fileInfoLine : vecFilesInDir)
            {
                size_t foundAt = fileInfoLine.find(filenameWOExt);
                if (foundAt != string::npos)
                {
                    string fileWExt  = fileInfoLine.substr(foundAt);
                    string fileWOExt = Utils::getFileNameWOExt(fileWExt);
                    strippedFiles.push_back(fileWOExt);
                }
            }
        }

        if (!strippedFiles.empty())
        {
            // sort filename naturally as many file systems do.
            std::sort(strippedFiles.begin(), strippedFiles.end(), Utils::compareNatural);
            string latest = strippedFiles.back() + ".xml";
            return latest;
        }
        else
            return "";
    }

    // Return empty for not found
    return "";
}
//-----------------------------------------------------------------------------
//! Returns the version number at the end of the calibration filename
int CVCalibration::getVersionInCalibFilename(const string& calibFilename)
{
    string calibFilenameWOExt = Utils::getFileNameWOExt(calibFilename);

    int versionNO = 0;
    if (!calibFilenameWOExt.empty())
    {
        size_t len = calibFilenameWOExt.length();
        if (calibFilenameWOExt.at(len - 1) == ')')
        {
            size_t leftPos = calibFilenameWOExt.rfind('(');
            string verStr  = calibFilenameWOExt.substr(leftPos + 1, len - leftPos - 2);
            versionNO      = stoi(verStr);
        }
    }
    return versionNO;
}
//-----------------------------------------------------------------------------
