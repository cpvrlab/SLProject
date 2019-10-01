//#############################################################################
//  File:      CVCalibration.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCALIBRATION_H
#define CVCALIBRATION_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVTypedefs.h>
#include <ftplib.h>

using namespace std;
//-----------------------------------------------------------------------------
//! OpenCV Calibration state
enum CVCalibState
{
    CS_uncalibrated,     //!< The camera is not calibrated (no calibration found)
    CS_calibrateStream,  //!< The calibration is running with live video stream
    CS_calibrateGrab,    //!< The calibration is running and an image should be grabbed
    CS_startCalculating, //!< The calibration starts during the next frame
    CS_calibrated,       //!< The camera is calibrated
    CS_guessed           //!< The camera intrinsics where estimated from FOV
};

//-----------------------------------------------------------------------------
//! Live video camera calibration class with OpenCV an OpenCV calibration.
/*! The camera calibration can determine the inner or intrinsic parameters such
as the focal length and the lens distortion and external or extrinsic parameter
such as the camera pose towards a known geometry.
\n
For a good calibration we have to make 15-20 images from a chessboard pattern.
The chessboard pattern can be printed from the CalibrationChessboard_8x5_A4.pdf
in the folder data/calibration. It is important that one side has an odd number
of inner corners. Like this it is unambiguous and can be rotated in any direction.
\n
The different calibration states are handled within AppDemoTracking::onUpdateTracking:
\n
- CS_uncalibrated:     The camera is not calibrated (no calibration found found)
- CS_calibrateStream:  The calibration is running with live video stream
- CS_calibrateGrab:    The calibration is running and an image should be grabbed
- CS_startCalculating: The calibration starts during the next frame
- CS_calibrated:       The camera is calibrated
- CS_estimate:         The camera intrinsics are set from an estimated FOV angle
\n
The core of the intrinsic calibration is stored in the members _cameraMat and
_distortion. For the calibration internals see the OpenCV documentation:
http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
After a successfully calibration the parameters are stored in a config file on
the SL::configPath. If it exists, it is loaded from there at startup.
If doesn't exist a simple calibration from a default field of view angle is
estimated.
\n
The CVCapture instance has two video camera calibrations, one for a main camera
(CVCapture::calibMainCam) and one for the selfie camera on mobile devices
(CVCapture::calibScndCam). The member CVCapture::activeCalib points to the active
one and is set by the CVCapture::videoType (VT_NONE, VT_MAIN, VT_SCND) during the
scene assembly in AppDemoLoad. On mobile devices the front camera is the
selfie camera (our secondary) and the back camera is the our main camera.
*/
class CVCalibration
{
    public:
    CVCalibration();

    bool   load(const string& calibDir,
                const string& calibFileName,
                bool          mirrorHorizontally,
                bool          mirrorVertically);
    void   save();
    bool   loadCalibParams();
    bool   calculate();
    void   clear();
    void   uploadCalibration(const string& fullPathAndFilename);
    void   downloadCalibration(const string& fullPathAndFilename);
    string getLatestCalibFilename(ftplib& ftp, const string& calibFileWOExt);
    int    getVersionInCalibFilename(const string& calibFilename);
    float  calcReprojectionErr(const CVVVPoint3f& objectPoints,
                               const CVVMat&      rvecs,
                               const CVVMat&      tvecs,
                               vector<float>&     perViewErrors);
    bool   findChessboard(CVMat        imageColor,
                          const CVMat& imageGray,
                          bool         drawCorners = true);
    void   buildUndistortionMaps();
    void   remap(CVMat& inDistorted,
                 CVMat& outUndistorted);
    void   createFromGuessedFOV(int imageWidthPX,
                                int imageHeightPX);
    void   adaptForNewResolution(const CVSize& newSize);

    static string calibIniPath; //!< calibration init parameters file path
    static void   calcBoardCorners3D(const CVSize& boardSize,
                                     float         squareSize,
                                     CVVPoint3f&   objectPoints3D);

    // Setters
    void state(CVCalibState s) { _state = s; }
    void imageSize(const CVSize& newSize)
    {
        if (newSize != _imageSize)
        {
            //if (_state == CS_calibrated)
            //    adaptForNewResolution(newSize);
            //save();
            //else
            {
                clear();
                _imageSize = newSize;
            }
        }
    }
    void camSizeIndex(int index)
    {
        _camSizeIndex = index;
    }
    void toggleMirrorH()
    {
        clear();
        _isMirroredH = !_isMirroredH;
        save();
    }
    void toggleMirrorV()
    {
        clear();
        _isMirroredV = !_isMirroredV;
        save();
    }
    void toggleFixPrincipalPoint()
    {
        clear();
        _calibFixPrincipalPoint = !_calibFixPrincipalPoint;
    }
    void toggleFixAspectRatio()
    {
        clear();
        _calibFixAspectRatio = !_calibFixAspectRatio;
    }
    void toggleZeroTangentDist()
    {
        clear();
        _calibZeroTangentDist = !_calibZeroTangentDist;
    }
    void showUndistorted(bool su) { _showUndistorted = su; }
    void devFocalLength(float f) { _devFocalLength = f; }
    void devSensorSizeW(float w) { _devSensorSizeW = w; }
    void devSensorSizeH(float h) { _devSensorSizeH = h; }

    // Getters
    CVSize       imageSize() { return _imageSize; }
    int          camSizeIndex() { return _camSizeIndex; }
    float        imageAspectRatio() { return (float)_imageSize.width / (float)_imageSize.height; }
    CVMat&       cameraMat() { return _cameraMat; }
    CVMat&       distortion() { return _distortion; }
    float        cameraFovVDeg() { return _cameraFovVDeg; }
    float        cameraFovHDeg() { return _cameraFovHDeg; }
    bool         calibFixPrincipalPoint() { return _calibFixPrincipalPoint; }
    bool         calibFixAspectRatio() { return _calibFixAspectRatio; }
    bool         calibZeroTangentDist() { return _calibZeroTangentDist; }
    bool         isMirroredH() { return _isMirroredH; }
    bool         isMirroredV() { return _isMirroredV; }
    float        fx() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(0, 0) : 0.0f; }
    float        fy() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(1, 1) : 0.0f; }
    float        cx() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(0, 2) : 0.0f; }
    float        cy() { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(1, 2) : 0.0f; }
    float        k1() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(0, 0) : 0.0f; }
    float        k2() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(1, 0) : 0.0f; }
    float        p1() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(2, 0) : 0.0f; }
    float        p2() { return _distortion.rows >= 4 ? (float)_distortion.at<double>(3, 0) : 0.0f; }
    CVCalibState state() { return _state; }
    int          numImgsToCapture() { return _numOfImgsToCapture; }
    int          numCapturedImgs() { return _numCaptured; }
    float        reprojectionError() { return _reprojectionError; }
    bool         showUndistorted() { return _showUndistorted; }
    CVSize       boardSize() { return _boardSize; }
    float        boardSquareMM() { return _boardSquareMM; }
    float        boardSquareM() { return _boardSquareMM * 0.001f; }
    string       calibrationTime() { return _calibrationTime; }
    string       calibDir() { return _calibDir; }
    string       calibFileName() { return _calibFileName; }
    string       stateStr()
    {
        switch (_state)
        {
            case CS_uncalibrated: return "CS_uncalibrated";
            case CS_calibrated: return "CS_calibrated";
            case CS_guessed: return "CS_guessed";
            case CS_calibrateStream: return "CS_calibrateStream";
            case CS_calibrateGrab: return "CS_calibrateGrab";
            case CS_startCalculating: return "CS_startCalculating";
            default: return "unknown";
        }
    }

    private:
    void calcCameraFov();

    ///////////////////////////////////////////////////////////////////////////////////
    CVMat _cameraMat;  //!< 3x3 Matrix for intrinsic camera matrix
    CVMat _distortion; //!< 4x1 Matrix for intrinsic distortion
    ///////////////////////////////////////////////////////////////////////////////////

    CVCalibState _state;                  //!< calibration state enumeration
    float        _cameraFovVDeg;          //!< Vertical field of view in degrees
    float        _cameraFovHDeg;          //!< Horizontal field of view in degrees
    string       _calibDir;               //!< directory of calibration file
    string       _calibFileName;          //!< name for calibration file
    string       _calibParamsFileName;    //!< name of calibration paramters file
    int          _calibFlags;             //!< OpenCV calibration flags
    bool         _calibFixPrincipalPoint; //!< Calib. flag for fix principal point
    bool         _calibFixAspectRatio;    //!< Calib. flag for fix aspect ratio
    bool         _calibZeroTangentDist;   //!< Calib. flag for zero tangent distortion
    bool         _isMirroredH = false;    //!< Flag if image must be horizontally mirrored
    bool         _isMirroredV = false;    //!< Flag if image must be vertically mirrored
    CVSize       _boardSize;              //!< NO. of inner chessboard corners.
    float        _boardSquareMM;          //!< Size of chessboard square in mm
    int          _numOfImgsToCapture;     //!< NO. of images to capture
    int          _numCaptured;            //!< NO. of images captured
    float        _reprojectionError;      //!< Reprojection error after calibration
    CVVVPoint2f  _imagePoints;            //!< 2D vector of corner points in chessboard
    CVSize       _imageSize;              //!< Input image size in pixels (after cropping)
    int          _camSizeIndex;           //!< The requested camera size index
    bool         _showUndistorted;        //!< Flag if image should be undistorted
    CVMat        _undistortMapX;          //!< Undistortion float map in x-direction
    CVMat        _undistortMapY;          //!< Undistortion float map in y-direction
    CVMat        _cameraMatUndistorted;   //!< Camera matrix for undistorted image
    string       _calibrationTime;        //!< Time stamp string of calibration
    float        _devFocalLength;         //!< Androids DeviceLensFocalLength
    float        _devSensorSizeW;         //!< Androids DeviceSensorPhysicalSizeW
    float        _devSensorSizeH;         //!< Androids DeviceSensorPhysicalSizeH

    static const int    _CALIBFILEVERSION; //!< Global const file format version
    static const string _FTP_HOST;         //!< ftp host for calibration up and download
    static const string _FTP_USER;         //!< ftp login user for calibration up and download
    static const string _FTP_PWD;          //!< ftp login pwd for calibration up and download
    static const string _FTP_DIR;          //!< ftp directory for calibration up and download
};
//-----------------------------------------------------------------------------
#endif // CVCalibration_H
