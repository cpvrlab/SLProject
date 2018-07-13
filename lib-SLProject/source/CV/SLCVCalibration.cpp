//#############################################################################
//  File:      SLCVCalibration.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLApplication.h>
#include <SLCVCalibration.h>
#include <SLCVCapture.h>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
//! Default path for calibration files
SLstring    SLCVCalibration::calibIniPath       = "../_data/calibrations/";

//! Increase the _CALIBFILEVERSION each time you change the file format
const SLint SLCVCalibration::_CALIBFILEVERSION = 3;    // Date: 26.Fev.2017
//-----------------------------------------------------------------------------
SLCVCalibration::SLCVCalibration() :
    _cameraFovDeg(1.0f),
    _state(CS_uncalibrated),
    _calibFileName(""), // is set in load
    _calibParamsFileName("calib_in_params.yml"),
    _calibFixPrincipalPoint(true),
    _calibFixAspectRatio(true),
    _calibZeroTangentDist(true),
    _boardSize(0, 0),
    _boardSquareMM(0.0f),
    _numOfImgsToCapture(0),
    _numCaptured(0),
    _reprojectionError(-1.0f),
    _showUndistorted(false),
    _calibrationTime("-")
{
}
//-----------------------------------------------------------------------------
//! Resets the calibration to the uncalibrated state
void SLCVCalibration::clear()
{
    _numCaptured = 0;
    _reprojectionError = -1.0f;
    _imagePoints.clear();
    _cameraFovDeg = 1.0f;
    _calibrationTime = "-";
    _undistortMapX.release();
    _undistortMapY.release();
    _state = CS_uncalibrated;
}
//-----------------------------------------------------------------------------
//! Loads the calibration information from the config file
bool SLCVCalibration::load(SLstring calibFileName,
                           SLbool mirrorHorizontally,
                           SLbool mirrorVertically)
{
    _calibFileName = calibFileName;
    _isMirroredH = mirrorHorizontally;
    _isMirroredV = mirrorVertically;

    //load camera parameter
    SLstring fullPathAndFilename = SLApplication::configPath + _calibFileName;
    FileStorage fs(fullPathAndFilename, FileStorage::READ);

    if (!fs.isOpened())
    {   SL_LOG("Calibration     : %s\n", calibFileName.c_str());
        SL_LOG("Calib. created  : No. Calib. will be estimated\n");
        _numCaptured = 0;
        _isMirroredH = mirrorHorizontally;
        _isMirroredV = mirrorVertically;
        _reprojectionError = 0;
        _calibrationTime = "-";
        _state = CS_uncalibrated;
        SLCVCapture::requestedSizeIndex = 0;
        return false;
    }

    SLint calibFileVersion = 0;
    fs["CALIBFILEVERSION"]  >> calibFileVersion;

    // Reset if new file format version is available
    if (calibFileVersion < _CALIBFILEVERSION)
    {
        _numCaptured            = 0;
        _isMirroredH            = mirrorHorizontally;
        _isMirroredV            = mirrorVertically;
        _calibFixAspectRatio    = true;
        _calibFixPrincipalPoint = true;
        _calibZeroTangentDist   = true;
        _reprojectionError      = -1;
        _calibrationTime        = "-";
        _state                  = CS_uncalibrated;
        SLCVCapture::requestedSizeIndex = 0;

    } else
    {
        fs["imageSizeWidth"]            >> _imageSize.width;
        fs["imageSizeHeight"]           >> _imageSize.height;
        fs["numCaptured"]               >> _numCaptured;
        fs["isMirroredH"]               >> _isMirroredH;
        fs["isMirroredV"]               >> _isMirroredV;
        fs["calibFixAspectRatio"]       >> _calibFixAspectRatio;
        fs["calibFixPrincipalPoint"]    >> _calibFixPrincipalPoint;
        fs["calibZeroTangentDist"]      >> _calibZeroTangentDist;
        fs["cameraMat"]                 >> _cameraMat;
        fs["distortion"]                >> _distortion;
        fs["reprojectionError"]         >> _reprojectionError;
        fs["calibrationTime"]           >> _calibrationTime;
        fs["requestedVideoSizeIndex"]   >> SLCVCapture::requestedSizeIndex;
        _state = _numCaptured ? CS_calibrated : CS_uncalibrated;
    }

    // close the input file
    fs.release();

    //calculate FOV and undistortion maps
    if (_state == CS_calibrated)
    {   _cameraFovDeg = calcCameraFOV();
        buildUndistortionMaps();
    }

    SL_LOG("Calib. loaded   : %s\n", fullPathAndFilename.c_str());
    SL_LOG("Calib. created  : %s\n", _calibrationTime.c_str());
    SL_LOG("Camera FOV      : %f\n", _cameraFovDeg);

    return true;
}
//-----------------------------------------------------------------------------
//! Saves the camera calibration parameters to the config file
void SLCVCalibration::save()
{
    SLstring fullPathAndFilename = SLApplication::configPath + _calibFileName;

    cv::FileStorage fs(fullPathAndFilename, FileStorage::WRITE);

    if (!fs.isOpened())
    {   SL_LOG("Failed to write calib. %s\n", fullPathAndFilename.c_str());
        return;
    }

    SLchar buf[1024];
    if (_calibFlags)
    {   sprintf(buf, "flags:%s%s%s%s",
                _calibFlags & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
                _calibFlags & CALIB_FIX_ASPECT_RATIO ?    " +fix_aspectRatio"     : "",
                _calibFlags & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
                _calibFlags & CALIB_ZERO_TANGENT_DIST ?   " +zero_tangent_dist"   : "");
        cvWriteComment(*fs, buf, 0);
    }

    fs << "CALIBFILEVERSION"        << _CALIBFILEVERSION;
    fs << "imageSizeWidth"          << _imageSize.width;
    fs << "imageSizeHeight"         << _imageSize.height;
    fs << "boardSizeWidth"          << _boardSize.width;    // do not reload
    fs << "boardSizeHeight"         << _boardSize.height;   // do not reload
    fs << "boardSquareMM"           << _boardSquareMM;      // do not reload
    fs << "numCaptured"             << _numCaptured;
    fs << "calibFlags"              << _calibFlags;
    fs << "isMirroredH"             << _isMirroredH;
    fs << "isMirroredV"             << _isMirroredV;
    fs << "calibFixAspectRatio"     << _calibFixAspectRatio;
    fs << "calibFixPrincipalPoint"  << _calibFixPrincipalPoint;
    fs << "calibZeroTangentDist"    << _calibZeroTangentDist;
    fs << "cameraMat"               << _cameraMat;
    fs << "distortion"              << _distortion;
    fs << "reprojectionError"       << _reprojectionError;
    fs << "cameraFovDeg"            << _cameraFovDeg;
    fs << "calibrationTime"         << _calibrationTime;
    fs << "requestedVideoSizeIndex" << SLCVCapture::requestedSizeIndex;

    // close file
    fs.release();
    SL_LOG("Calib. saved    : %s\n", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
//! Loads the chessboard calibration pattern parameters
bool SLCVCalibration::loadCalibParams()
{
    FileStorage fs;
    fs.open(calibIniPath + _calibParamsFileName, FileStorage::READ);
    if (!fs.isOpened())
    {   cout << "Could not open the calibration parameter file: "
             << (calibIniPath + _calibParamsFileName) << endl;
        _state = CS_uncalibrated;
        return false;
    }

    //assign paramters
    fs["numInnerCornersWidth"]  >> _boardSize.width;
    fs["numInnerCornersHeight"] >> _boardSize.height;
    fs["squareSizeMM"]          >> _boardSquareMM;
    fs["numOfImgsToCapture"]    >> _numOfImgsToCapture;

    return true;
}
//-----------------------------------------------------------------------------
//! Calculates the vertical field of view angle in degrees
SLfloat SLCVCalibration::calcCameraFOV()
{
    if (_cameraMat.rows!=3||_cameraMat.cols!=3)
        SL_EXIT_MSG("SLCVCalibration::calcCameraFOV: No intrinsic parameter available");

    //calculate vertical field of view
    SLfloat fy = (SLfloat)_cameraMat.at<double>(1,1);
    SLfloat cy = (SLfloat)_cameraMat.at<double>(1,2);
    SLfloat fovRad = 2 * (SLfloat)atan2(cy, fy);
    return fovRad * SL_RAD2DEG;
}
//-----------------------------------------------------------------------------
//! Calculates the 3D positions of the chessboard corners
void SLCVCalibration::calcBoardCorners3D(SLCVSize boardSize, 
                                         SLfloat squareSize, 
                                         SLCVVPoint3f& objectPoints3D)
{
    // Because OpenCV image coords are top-left we define the according
    // 3D coords also top-left.
    objectPoints3D.clear();
    for(SLint y = boardSize.height-1; y >= 0 ; --y)
        for(SLint x = 0; x < boardSize.width; ++x)
            objectPoints3D.push_back(SLCVPoint3f(x*squareSize, 
                                                 y*squareSize, 
                                                 0));
}
//-----------------------------------------------------------------------------
//! Calculates the reprojection error of the calibration
SLfloat SLCVCalibration::calcReprojectionErr(const SLCVVVPoint3f& objectPoints,
                                             const SLCVVMat& rvecs,
                                             const SLCVVMat& tvecs,
                                             SLVfloat& perViewErrors)
{
    SLCVVPoint2f imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i)
    {
        cv::projectPoints(objectPoints[i], 
                          rvecs[i], 
                          tvecs[i], 
                          _cameraMat,
                          _distortion,
                          imagePoints2);

        err = norm(_imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (SLfloat) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return (SLfloat)std::sqrt(totalErr/totalPoints);
}
//-----------------------------------------------------------------------------
//!< Finds the inner chessboard corners in the given image
bool SLCVCalibration::findChessboard(SLCVMat imageColor,
                                     SLCVMat imageGray,
                                     bool drawCorners)
{   
    assert(!imageGray.empty() && 
           "SLCVCalibration::findChessboard: imageGray is empty!");
    assert(!imageColor.empty() && 
           "SLCVCalibration::findChessboard: imageColor is empty!");
    assert(_boardSize.width && _boardSize.height && 
           "SLCVCalibration::findChessboard: _boardSize is not set!");

    //debug save image
    //stringstream ss;
    //ss << "imageIn_" << _numCaptured << ".png";
    //cv::imwrite(ss.str(), imageColor);

    _imageSize = imageColor.size();

    SLCVVPoint2f corners2D;
    SLint flags = CALIB_CB_ADAPTIVE_THRESH | 
                  CALIB_CB_NORMALIZE_IMAGE | 
                  CALIB_CB_FAST_CHECK;

    bool found = cv::findChessboardCorners(imageGray,
                                           _boardSize,
                                           corners2D,
                                           flags);

    if(found && drawCorners)
        cv::drawChessboardCorners(imageColor,
                                  _boardSize,
                                  SLCVMat(corners2D),
                                  found);

    if (found && _state == CS_calibrateGrab)
    {
        cv::cornerSubPix(imageGray,
                         corners2D, 
                         SLCVSize(11,11),
                         SLCVSize(-1,-1), 
                         TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 
                         30, 
                         0.1));

        //add detected points
        _imagePoints.push_back(corners2D);
        _numCaptured++;

        //simulate a snapshot
        cv::bitwise_not(imageColor, imageColor);

        _state = CS_calibrateStream;
    }
    return found;
}
//-----------------------------------------------------------------------------
//! Calculates the reprojection error of the calibration
static double calcReprojectionErrors(const SLCVVVPoint3f& objectPoints,
                                     const SLCVVVPoint2f& imagePoints,
                                     const SLCVVMat& rvecs,
                                     const SLCVVMat& tvecs,
                                     const SLCVMat& cameraMatrix ,
                                     const SLCVMat& distCoeffs,
                                     SLVfloat& perViewErrors)
{
    SLCVVPoint2f imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i)
    {
        cv::projectPoints(objectPoints[i], 
                          rvecs[i], 
                          tvecs[i], 
                          cameraMatrix, 
                          distCoeffs, 
                          imagePoints2);

        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (SLfloat) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}
//-----------------------------------------------------------------------------
//! Calculates the calibration with the given set of image points
static bool calcCalibration(SLCVSize& imageSize,
                            SLCVMat& cameraMatrix,
                            SLCVMat& distCoeffs,
                            SLCVVVPoint2f imagePoints,
                            SLCVVMat& rvecs, 
                            SLCVVMat& tvecs,
                            SLVfloat& reprojErrs,
                            SLfloat& totalAvgErr,
                            SLCVSize& boardSize,
                            SLfloat squareSize,
                            SLint flag)
{
    // Init camera matrix with the eye setter
    cameraMatrix = SLCVMat::eye(3, 3, CV_64F);

    // We need to set eleme at 0,0 to 1 if we want a fix aspect ratio
    if(flag & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0,0) = 1.0;

    // init the distortion coeffitients to zero
    distCoeffs = SLCVMat::zeros(8, 1, CV_64F);

    SLCVVVPoint3f objectPoints(1);

    SLCVCalibration::calcBoardCorners3D(boardSize, 
                                        squareSize, 
                                        objectPoints[0]);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

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

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

    totalAvgErr = (SLfloat)calcReprojectionErrors(objectPoints,
                                                  imagePoints,
                                                  rvecs,
                                                  tvecs,
                                                  cameraMatrix,
                                                  distCoeffs,
                                                  reprojErrs);
    return ok;
}
//-----------------------------------------------------------------------------
//! Initiates the final calculation
bool SLCVCalibration::calculate()
{
    _state = CS_startCalculating;

    SLCVVMat rvecs, tvecs;
    SLVfloat reprojErrs;

    _calibFlags = 0;
    if (_calibFixPrincipalPoint) _calibFlags |= CALIB_FIX_PRINCIPAL_POINT;
    if (_calibZeroTangentDist)   _calibFlags |= CALIB_ZERO_TANGENT_DIST;
    if (_calibFixAspectRatio)    _calibFlags |= CALIB_FIX_ASPECT_RATIO;

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

    if(!rvecs.empty() || !reprojErrs.empty())
         _numCaptured = (int)std::max(rvecs.size(), reprojErrs.size());
    else _numCaptured = 0;

    if (ok)
    {
        buildUndistortionMaps();

        _cameraFovDeg = calcCameraFOV();
        _calibrationTime = SLUtils::getLocalTimeString();
        _state = CS_calibrated;
        save();

        cout << "Calibration succeeded. Reprojection error = " << _reprojectionError << endl;
        cout << "cameraMat:" << _cameraMat << endl;
        cout << "distortion:"<< _distortion << endl;
    } else
    {   _cameraFovDeg = 1.0f;
        _calibrationTime = "-";
        _undistortMapX.release();
        _undistortMapY.release();
        _state = CS_uncalibrated;
        cout << "Calibration failed." << endl;
    }
    return ok;
}
//-----------------------------------------------------------------------------
//! Builds undistortion maps after calibration or loading
void SLCVCalibration::buildUndistortionMaps()
{
    // An alpha of 0 leads to no black borders
    // An alpha of 1 leads to black borders
    double alpha = 0.5; 
    
    // Create optimal camera matrix for undistorted image
    _cameraMatUndistorted = cv::getOptimalNewCameraMatrix(_cameraMat,
                                                          _distortion,
                                                          _imageSize,
                                                          alpha,
                                                          _imageSize,
                                                          0,
                                                          true);
    // Create undistortion maps
    _undistortMapX.release();
    _undistortMapY.release();

    cv::initUndistortRectifyMap(_cameraMat,
                                _distortion,
                                cv::Mat(),   // Identity matrix R
                                _cameraMatUndistorted,
                                _imageSize,
                                CV_32FC1,
                                _undistortMapX,
                                _undistortMapY);

    if (_undistortMapX.empty() || _undistortMapY.empty())
        SL_EXIT_MSG("SLCVCalibration::buildUndistortionMaps failed.");
}
//-----------------------------------------------------------------------------
//! Undistorts the inDistorted image into the outUndistorted
void SLCVCalibration::remap(SLCVMat &inDistorted, SLCVMat &outUndistorted)
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
              CV_INTER_LINEAR);
}
//-----------------------------------------------------------------------------
//! Calculates camera intrinsics from a guessed FOV angle
/* Most laptop-, webcam- or mobile camera have a vertical view angle or
socalled field of view (FOV) of around 40-44 degrees. From this parameter we
can calculate the most important intrinsic parameter the focal length. All
other parameters are set as if the lens would be perfect: No lens distortion
and the view axis goes through the center of the image.
*/
void SLCVCalibration::createFromGuessedFOV(SLint imageWidthPX,
                                           SLint imageHeightPX)
{
    // vertical view angle in degrees
    SLfloat fov = 42.0f;

    // Create standard camera matrix
    // fx, fx, cx, cy are all in pixel values not mm
    // We asume that we have an ideal image sensor with square pixels
    // so that the focal length fx and fy are identical
    // See the OpenCV documentation for more details:
    // http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html

    SLfloat cx = (float)imageWidthPX * 0.5f;
    SLfloat cy = (float)imageHeightPX * 0.5f;
    SLfloat fy = cy / tanf(fov*0.5f*SL_DEG2RAD);
    SLfloat fx = fy;

    _imageSize.width = imageWidthPX;
    _imageSize.height = imageHeightPX;
    _cameraMat = (Mat_<double>(3,3) << fx,  0,  cx,
                                        0, fy,  cy,
                                        0,  0,   1);
    // No distortion
    _distortion = (Mat_<double>(5,1) << 0,0,0,0,0);

    _cameraFovDeg = fov;
    _calibrationTime = SLUtils::getLocalTimeString();
    _state = CS_guessed;
}
//-----------------------------------------------------------------------------
