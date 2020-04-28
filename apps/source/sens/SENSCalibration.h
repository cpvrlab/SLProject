//#############################################################################
//  File:      SENSCalibration.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENSCALIBRATION_H
#define SENSCALIBRATION_H

#include <SENSCalibrationEstimatorParams.h>

using namespace std;

class SENSCalibration
{
public:
    enum class State
    {
        uncalibrated, //!< The camera is not calibrated (no calibration found)
        calibrated,   //!< The camera is calibrated
        guessed,      //!< The camera intrinsics where estimated from FOV
    };

    //creates a fully defined calibration
    SENSCalibration(const cv::Mat&     cameraMat,
                    const cv::Mat&     distortion,
                    cv::Size           imageSize,
                    cv::Size           boardSize,
                    float              boardSquareMM,
                    float              reprojectionError,
                    int                numCaptured,
                    const std::string& calibrationTime,
                    int                camSizeIndex,
                    bool               mirroredH,
                    bool               mirroredV,
                    SENSCameraType     camType,
                    std::string        computerInfos,
                    int                calibFlags,
                    bool               calcUndistortionMaps);

    //creates a guessed calibration using image size and fov angle
    SENSCalibration(const cv::Size& imageSize,
                    float           fovH,
                    bool            mirroredH,
                    bool            mirroredV,
                    SENSCameraType  type,
                    std::string     computerInfos);

    //create a guessed calibration using sensor size, camera focal length and captured image size
    SENSCalibration(float           sensorWMM,
                    float           sensorHMM,
                    float           focalLengthMM,
                    const cv::Size& imageSize,
                    bool            mirroredH,
                    bool            mirroredV,
                    SENSCameraType  camType,
                    std::string     computerInfos);

    bool load(const string& calibDir,
              const string& calibFileName,
              bool          calcUndistortionMaps);
    bool save(const string& calibDir,
              const string& calibFileName);

    void remap(cv::Mat& inDistorted,
               cv::Mat& outUndistorted);

    //! Adapts an already calibrated camera to a new resolution (cropping and scaling)
    void adaptForNewResolution(const cv::Size& newSize, bool calcUndistortionMaps);
    void buildUndistortionMaps();

    static void getInnerAndOuterRectangles(const cv::Mat&    cameraMatrix,
                                           const cv::Mat&    distCoeffs,
                                           const cv::Mat&    R,
                                           const cv::Mat&    newCameraMatrix,
                                           const cv::Size&   imgSize,
                                           cv::Rect_<float>& inner,
                                           cv::Rect_<float>& outer);

    // Getters
    cv::Size imageSize() const { return _imageSize; }
    cv::Size imageSizeOriginal() const { return _imageSizeOrig; }

    float          imageAspectRatio() const { return (float)_imageSize.width / (float)_imageSize.height; }
    const cv::Mat& cameraMat() const { return _cameraMat; }
    const cv::Mat& cameraMatUndistorted() const { return _cameraMatUndistorted; }
    const cv::Mat& distortion() const { return _distortion; }
    float          cameraFovVDeg() const { return _cameraFovVDeg; }
    float          cameraFovHDeg() const { return _cameraFovHDeg; }

    int  calibrationFlags() { return _calibFlags; }
    bool calibFixPrincipalPoint() { return _calibFlags & cv::CALIB_FIX_PRINCIPAL_POINT; }
    bool calibFixAspectRatio() { return _calibFlags & cv::CALIB_FIX_ASPECT_RATIO; }
    bool calibZeroTangentDist() { return _calibFlags & cv::CALIB_ZERO_TANGENT_DIST; }
    bool calibRationalModel() { return _calibFlags & cv::CALIB_RATIONAL_MODEL; }
    bool calibTiltedModel() { return _calibFlags & cv::CALIB_TILTED_MODEL; }
    bool calibThinPrismModel() { return _calibFlags & cv::CALIB_THIN_PRISM_MODEL; }
    bool isMirroredH() { return _isMirroredH; }
    bool isMirroredV() { return _isMirroredV; }

    float fx() const { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(0, 0) : 0.0f; }
    float fy() const { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(1, 1) : 0.0f; }
    float cx() const { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(0, 2) : 0.0f; }
    float cy() const { return _cameraMat.cols == 3 && _cameraMat.rows == 3 ? (float)_cameraMat.at<double>(1, 2) : 0.0f; }
    float k1() const { return _distortion.rows >= 4 ? (float)_distortion.at<double>(0, 0) : 0.0f; }
    float k2() const { return _distortion.rows >= 4 ? (float)_distortion.at<double>(1, 0) : 0.0f; }
    float p1() const { return _distortion.rows >= 4 ? (float)_distortion.at<double>(2, 0) : 0.0f; }
    float p2() const { return _distortion.rows >= 4 ? (float)_distortion.at<double>(3, 0) : 0.0f; }
    float k3() const { return _distortion.rows >= 5 ? (float)_distortion.at<double>(4, 0) : 0.0f; }
    float k4() const { return _distortion.rows >= 6 ? (float)_distortion.at<double>(5, 0) : 0.0f; }
    float k5() const { return _distortion.rows >= 7 ? (float)_distortion.at<double>(6, 0) : 0.0f; }
    float k6() const { return _distortion.rows >= 8 ? (float)_distortion.at<double>(7, 0) : 0.0f; }
    float s1() const { return _distortion.rows >= 9 ? (float)_distortion.at<double>(8, 0) : 0.0f; }
    float s2() const { return _distortion.rows >= 10 ? (float)_distortion.at<double>(9, 0) : 0.0f; }
    float s3() const { return _distortion.rows >= 11 ? (float)_distortion.at<double>(10, 0) : 0.0f; }
    float s4() const { return _distortion.rows >= 12 ? (float)_distortion.at<double>(11, 0) : 0.0f; }
    float tauX() const { return _distortion.rows >= 13 ? (float)_distortion.at<double>(12, 0) : 0.0f; }
    float tauY() const { return _distortion.rows >= 14 ? (float)_distortion.at<double>(13, 0) : 0.0f; }

    SENSCameraType camType() const { return _camType; }
    State          state() const { return _state; }
    int            numCapturedImgs() const { return _numCaptured; }
    float          reprojectionError() const { return _reprojectionError; }
    cv::Size       boardSize() const { return _boardSize; }
    float          boardSquareMM() const { return _boardSquareMM; }
    float          boardSquareM() const { return _boardSquareMM * 0.001f; }
    string         calibrationTime() const { return _calibrationTime; }
    string         calibFileName() const { return _calibFileName; }
    string         computerInfos() const { return _computerInfos; }
    string         stateStr() const
    {
        switch (_state)
        {
            case State::uncalibrated: return "uncalibrated";
            case State::calibrated: return "calibrated";
            case State::guessed: return "guessed";
            default: return "unknown";
        }
    }

private:
    void calcCameraFovFromUndistortedCameraMat();
    void calculateUndistortedCameraMat();
    void createFromGuessedFOV(int imageWidthPX, int imageHeightPX, float fovH);
    ///////////////////////////////////////////////////////////////////////////////////
    cv::Mat _cameraMat;  //!< 3x3 Matrix for intrinsic camera matrix
    cv::Mat _distortion; //!< 4x1 Matrix for intrinsic distortion
    ///////////////////////////////////////////////////////////////////////////////////
    //original data used for adaption:
    cv::Mat  _cameraMatOrig; //!< 3x3 Matrix for intrinsic camera matrix (original from loading or calibration estimation)
    cv::Size _imageSizeOrig; //!< original image size (original from loading or calibration estimation)

    State  _state         = State::uncalibrated; //!< calibration state enumeration
    float  _cameraFovVDeg = 0.0f;                //!< Vertical field of view in degrees
    float  _cameraFovHDeg = 0.0f;                //!< Horizontal field of view in degrees
    string _calibFileName;                       //!< name for calibration file
    int    _calibFlags  = 0;                     //!< OpenCV calibration flags
    bool   _isMirroredH = false;                 //!< Flag if image must be horizontally mirrored
    bool   _isMirroredV = false;                 //!< Flag if image must be vertically mirrored

    int      _numCaptured = 0;           //!< NO. of images captured
    cv::Size _boardSize;                 //!< NO. of inner chessboard corners.
    float    _boardSquareMM     = 20.f;  //!< Size of chessboard square in mm
    float    _reprojectionError = -1.0f; //!< Reprojection error after calibration
    cv::Size _imageSize;                 //!< Input image size in pixels (after cropping)
    int      _camSizeIndex = -1;         //!< The requested camera size index

    cv::Mat        _undistortMapX;         //!< Undistortion float map in x-direction
    cv::Mat        _undistortMapY;         //!< Undistortion float map in y-direction
    cv::Mat        _cameraMatUndistorted;  //!< Camera matrix that defines scene camera and may also be used for reprojection of undistorted image
    string         _calibrationTime = "-"; //!< Time stamp string of calibration
    string         _computerInfos;
    SENSCameraType _camType = SENSCameraType::FRONTFACING;

    static const int _CALIBFILEVERSION; //!< Global const file format version
};
//-----------------------------------------------------------------------------

#endif // SENSCALIBRATION_H
