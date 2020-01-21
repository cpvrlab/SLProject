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

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
//! Increase the _CALIBFILEVERSION each time you change the file format
// Version 6, Date: 6.JUL.2019: Added device parameter from Android
const int CVCalibration::_CALIBFILEVERSION = 6;
//-----------------------------------------------------------------------------
CVCalibration::CVCalibration(CVCameraType type, std::string computerInfos)
  : _state(CS_uncalibrated),
    _cameraFovHDeg(0.0f),
    _cameraFovVDeg(0.0f),
    _calibFileName(""), // is set in load
    _boardSize(0, 0),
    _boardSquareMM(0.0f),
    _numCaptured(0),
    _reprojectionError(-1.0f),
    _camSizeIndex(-1),
    _calibrationTime("-"),
    _isMirroredH(false),
    _isMirroredV(false),
    _camType(type),
    _computerInfos(computerInfos)
{
}
//-----------------------------------------------------------------------------
//creates a fully defined calibration
CVCalibration::CVCalibration(cv::Mat            cameraMat,
                             cv::Mat            distortion,
                             cv::Size           imageSize,
                             cv::Size           boardSize,
                             float              boardSquareMM,
                             float              reprojectionError,
                             int                numCaptured,
                             const std::string& calibrationTime,
                             int                camSizeIndex,
                             bool               mirroredH,
                             bool               mirroredV,
                             CVCameraType       camType,
                             std::string        computerInfos,
                             int                calibFlags)
  : _cameraMat(cameraMat.clone()),
    _distortion(distortion.clone()),
    _imageSize(imageSize),
    _boardSize(boardSize),
    _boardSquareMM(boardSquareMM),
    _reprojectionError(reprojectionError),
    _numCaptured(numCaptured),
    _calibrationTime(calibrationTime),
    _camSizeIndex(camSizeIndex),
    _isMirroredH(mirroredH),
    _isMirroredV(mirroredV),
    _camType(camType),
    _computerInfos(computerInfos),
    _calibFlags(calibFlags)
{
    _cameraMatOrig = _cameraMat.clone();
    _imageSizeOrig = _imageSize;

    calculateUndistortedCameraMat();
    calcCameraFovFromUndistortedCameraMat();
    buildUndistortionMaps();
    _state = CS_calibrated;
}
//-----------------------------------------------------------------------------
//create a guessed calibration using image size and horizontal fov angle
CVCalibration::CVCalibration(cv::Size     imageSize,
                             float        fovH,
                             bool         mirroredH,
                             bool         mirroredV,
                             CVCameraType camType,
                             std::string  computerInfos)
  : _isMirroredH(mirroredH),
    _isMirroredV(mirroredV),
    _camType(camType),
    _computerInfos(computerInfos)
{
    createFromGuessedFOV(imageSize.width, imageSize.height, fovH);
    _cameraMatOrig = _cameraMat.clone();
    _imageSizeOrig = _imageSize;
}
//-----------------------------------------------------------------------------
//create a guessed calibration using sensor size, camera focal length and captured image size
CVCalibration::CVCalibration(float        sensorWMM,
                             float        sensorHMM,
                             float        focalLengthMM,
                             cv::Size     imageSize,
                             bool         mirroredH,
                             bool         mirroredV,
                             CVCameraType camType,
                             std::string  computerInfos)
  : _isMirroredH(mirroredH),
    _isMirroredV(mirroredV),
    _camType(camType),
    _computerInfos(computerInfos)
{
    // aspect ratio
    float devFovH = 2.0f * atan(sensorWMM / (2.0f * focalLengthMM)) * Utils::RAD2DEG;
    if (devFovH > 60.0f && devFovH < 70.0f)
    {
        createFromGuessedFOV(imageSize.width, imageSize.height, devFovH);
    }
    else
    {
        //if not between
        createFromGuessedFOV(imageSize.width, imageSize.height, 65.0);
    }
    _cameraMatOrig = _cameraMat.clone();
    _imageSizeOrig = _imageSize;
}
//-----------------------------------------------------------------------------
//! Loads the calibration information from the config file
bool CVCalibration::load(const string& calibDir,
                         const string& calibFileName)
{
    //load camera parameter
    string fullPathAndFilename = Utils::unifySlashes(calibDir) + calibFileName;

    // try to open the local calibration file
    FileStorage fs(fullPathAndFilename, FileStorage::READ);
    if (!fs.isOpened())
    {
        Utils::log("Calibration     : %s\n", calibFileName.c_str());
        Utils::log("Calib. created  : No. Calib. will be estimated\n");
        _numCaptured       = 0;
        _isMirroredH       = false;
        _isMirroredV       = false;
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
        _numCaptured       = 0;
        _reprojectionError = -1;
        _calibrationTime   = "-";
        _state             = CS_uncalibrated;
        _camSizeIndex      = -1;
    }
    else
    {
        fs["imageSizeWidth"] >> _imageSize.width;
        fs["imageSizeHeight"] >> _imageSize.height;
        fs["numCaptured"] >> _numCaptured;
        fs["isMirroredH"] >> _isMirroredH;
        fs["isMirroredV"] >> _isMirroredV;
        fs["cameraMat"] >> _cameraMat;
        fs["distortion"] >> _distortion;
        fs["reprojectionError"] >> _reprojectionError;
        fs["calibrationTime"] >> _calibrationTime;
        fs["camSizeIndex"] >> _camSizeIndex;
        fs["boardSizeWidth"] >> _boardSize.width;
        fs["boardSizeHeight"] >> _boardSize.height;
        fs["boardSquareMM"] >> _boardSquareMM;
        _state = _numCaptured ? CS_calibrated : CS_uncalibrated;
    }

    //estimate computer infos
    if (!fs["computerInfos"].empty())
        fs["computerInfos"] >> _computerInfos;
    else
    {
        std::vector<std::string> stringParts;
        Utils::splitString(Utils::getFileNameWOExt(calibFileName), '_', stringParts);
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

    _cameraMatOrig = _cameraMat.clone();
    _imageSizeOrig = _imageSize;

    return true;
}
//-----------------------------------------------------------------------------
//! Saves the camera calibration parameters to the config file
bool CVCalibration::save(const string& calibDir,
                         const string& calibFileName)
{
    std::string fullPathAndFilename = Utils::unifySlashes(calibDir) + calibFileName;

    cv::FileStorage fs(fullPathAndFilename, FileStorage::WRITE);

    if (!fs.isOpened())
    {
        Utils::log("Failed to write calib. %s\n", fullPathAndFilename.c_str());
        return false;
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
    fs << "calibFixAspectRatio" << (_calibFlags & CALIB_FIX_ASPECT_RATIO);
    fs << "calibFixPrincipalPoint" << (_calibFlags & CALIB_FIX_PRINCIPAL_POINT);
    fs << "calibZeroTangentDist" << (_calibFlags & CALIB_ZERO_TANGENT_DIST);
    fs << "calibRationalModel" << (_calibFlags & CALIB_RATIONAL_MODEL);
    fs << "calibTiltedModel" << (_calibFlags & CALIB_TILTED_MODEL);
    fs << "calibThinPrismModel" << (_calibFlags & CALIB_THIN_PRISM_MODEL);
    fs << "cameraMat" << _cameraMat;
    fs << "distortion" << _distortion;
    fs << "reprojectionError" << _reprojectionError;
    fs << "cameraFovVDeg" << _cameraFovVDeg;
    fs << "cameraFovHDeg" << _cameraFovHDeg;
    fs << "camSizeIndex" << _camSizeIndex;
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
    return true;
    //uploadCalibration(fullPathAndFilename);
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
 @param fovH average horizontal view angle in degrees
*/
void CVCalibration::createFromGuessedFOV(int   imageWidthPX,
                                         int   imageHeightPX,
                                         float fovH)
{
    // aspect ratio
    float withOverHeight = (float)imageWidthPX / (float)imageHeightPX;

    // the vertical fov is derived from the width because it could be cropped
    float fovV = fovH / withOverHeight;

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
//! Adapts an already calibrated camera to a new resolution (cropping and scaling)
void CVCalibration::adaptForNewResolution(const CVSize& newSize)
{
    if (_state == CS_uncalibrated)
        return;

    // new center and focal length in pixels not mm
    float fx, fy, cy, cx;

    // use original camera matrix for adaptions.
    // Otherwise we get rounding errors after too many adaptions.
    float fxOrig = (float)_cameraMatOrig.at<double>(0, 0);
    float fyOrig = (float)_cameraMatOrig.at<double>(1, 1);
    float cxOrig = (float)_cameraMatOrig.at<double>(0, 2);
    float cyOrig = (float)_cameraMatOrig.at<double>(1, 2);

    if (((float)newSize.width / (float)newSize.height) >
        ((float)_imageSizeOrig.width / (float)_imageSizeOrig.height))
    {
        float scaleFactor = (float)newSize.width / (float)_imageSizeOrig.width;

        fx                    = fxOrig * scaleFactor;
        fy                    = fyOrig * scaleFactor;
        float oldHeightScaled = _imageSizeOrig.height * scaleFactor;
        float heightDiff      = (oldHeightScaled - newSize.height) * 0.5f;

        cx = cxOrig * scaleFactor;
        cy = cyOrig * scaleFactor - heightDiff;
    }
    else
    {
        float scaleFactor    = (float)newSize.height / (float)_imageSizeOrig.height;
        fx                   = fxOrig * scaleFactor;
        fy                   = fyOrig * scaleFactor;
        float oldWidthScaled = _imageSizeOrig.width * scaleFactor;
        float widthDiff      = (oldWidthScaled - newSize.width) * 0.5f;

        cx = cxOrig * scaleFactor - widthDiff;
        cy = cyOrig * scaleFactor;
    }

    //std::cout << "adaptForNewResolution: _cameraMat before: " << _cameraMat << std::endl;
    _cameraMat = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    //std::cout << "adaptForNewResolution: _cameraMat after: " << _cameraMat << std::endl;
    //_distortion remains unchanged
    _calibrationTime = Utils::getDateTime2String();

    //std::cout << "adaptForNewResolution: _imageSize before: " << _imageSize << std::endl;
    _imageSize.width  = newSize.width;
    _imageSize.height = newSize.height;
    //std::cout << "adaptForNewResolution: _imageSize after: " << _imageSize << std::endl;

    calculateUndistortedCameraMat();
    calcCameraFovFromUndistortedCameraMat();
    buildUndistortionMaps();
}
//-----------------------------------------------------------------------------
//! Calculate a camera matrix that we use for the scene graph and for the reprojection of the undistored image
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
        //_cameraMatUndistorted     = cv::getOptimalNewCameraMatrix(_cameraMat, _distortion, _imageSize, alpha, _imageSize, nullptr, centerPrinciplePoint);
        //! (The following is the algorithm from cv::getOptimalNewCameraMatrix and the code is here for understanding (it does the same))

        double cx0 = _cameraMat.at<double>(0, 2);
        double cy0 = _cameraMat.at<double>(1, 2);
        double cx  = (_imageSize.width) * 0.5;
        double cy  = (_imageSize.height) * 0.5;

        cv::Rect_<float> inner, outer;
        getInnerAndOuterRectangles(_cameraMat,
                                   _distortion,
                                   cv::Mat(),
                                   _cameraMat,
                                   _imageSize,
                                   inner,
                                   outer);

        double s0 = std::max(std::max(std::max((double)cx / (cx0 - inner.x),
                                               (double)cy / (cy0 - inner.y)),
                                      (double)cx / (inner.x + inner.width - cx0)),
                             (double)cy / (inner.y + inner.height - cy0));

        double s1 = std::min(std::min(std::min((double)cx / (cx0 - outer.x),
                                               (double)cy / (cy0 - outer.y)),
                                      (double)cx / (outer.x + outer.width - cx0)),
                             (double)cy / (outer.y + outer.height - cy0));

        double s = s0 * (1 - alpha) + s1 * alpha;

        _cameraMatUndistorted = _cameraMat.clone();
        _cameraMatUndistorted.at<double>(0, 0) *= s;
        _cameraMatUndistorted.at<double>(1, 1) *= s;
        _cameraMatUndistorted.at<double>(0, 2) = cx;
        _cameraMatUndistorted.at<double>(1, 2) = cy;
    }
    else
    {
        _cameraMatUndistorted = cv::getOptimalNewCameraMatrix(_cameraMat,
                                                              _distortion,
                                                              _imageSize,
                                                              alpha,
                                                              _imageSize,
                                                              nullptr,
                                                              centerPrinciplePoint);
    }

    //std::cout << "_cameraMatUndistorted: " << _cameraMatUndistorted << std::endl;
    //std::cout << "_cameraMat: " << _cameraMat << std::endl;
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
