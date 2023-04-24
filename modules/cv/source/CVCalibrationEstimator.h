//#############################################################################
//  File:      CVCalibrationEstimator.h
//  Date:      Winter 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCALIBRATIONESTIMATOR_H
#define CVCALIBRATIONESTIMATOR_H

#include <string>
#include <future>
#include <CVCalibration.h>
#include <CVTypedefs.h>
#include <CVTypes.h>

using std::string;
using std::vector;

//-----------------------------------------------------------------------------
//! special exception that informs about errors during calibration process
class CVCalibrationEstimatorException : public std::runtime_error
{
public:
    CVCalibrationEstimatorException(const string& msg,
                                    const int     line,
                                    const string& file)
      : std::runtime_error(toMessage(msg, line, file).c_str())
    {
    }

private:
    string toMessage(const string& msg, const int line, const string& file)
    {
        std::stringstream ss;
        ss << msg << ": Exception thrown at line " << line << " in " << file << std::endl;
        return ss.str();
    }
};
//-----------------------------------------------------------------------------
class CVCalibrationEstimator
{
public:
    enum class State
    {
        Streaming = 0,      //!< Estimator waits for new frames
        Calculating,        //!< Estimator is currently calculating the calibration
        BusyExtracting,     //!< Estimator is busy extracting the corners of a frame
        Done,               //!< Estimator finished
        DoneCaptureAndSave, //!< All images are captured in
        Error
    };

    CVCalibrationEstimator(CVCalibrationEstimatorParams params,
                           int                          camSizeIndex,
                           bool                         mirroredH,
                           bool                         mirroredV,
                           CVCameraType                 camType,
                           string                       computerInfos,
                           string                       calibDataPath,
                           string                       imageOutputPath,
                           string                       exePath);
    ~CVCalibrationEstimator();

    bool calculate();
    bool updateAndDecorate(CVMat        imageColor,
                           const CVMat& imageGray,
                           bool         grabFrame,
                           bool         drawCorners = true);

    State state()
    {
        return _state;
    }
    int numImgsToCapture() { return _numOfImgsToCapture; }
    int numCapturedImgs() { return _numCaptured; }

    bool calibrationSuccessful() { return _calibrationSuccessful; }
    //! Get resulting calibration
    CVCalibration getCalibration() { return _calibration; }
    bool          isBusyExtracting() { return _state == State::BusyExtracting; }
    bool          isCalculating() { return _state == State::Calculating; }
    bool          isStreaming() { return _state == State::Streaming; }
    bool          isDone() { return _state == State::Done; }
    bool          isDoneCaptureAndSave() { return _state == State::DoneCaptureAndSave; }

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
                                int                flag,
                                bool               useReleaseObjectMethod);

private:
    bool calibrateAsync();
    bool extractAsync();
    bool loadCalibParams();
    void updateExtractAndCalc(bool found, bool grabFrame, cv::Mat imageGray);
    void updateOnlyCapture(bool found, bool grabFrame, cv::Mat imageGray);
    void saveImage(cv::Mat imageGray);

    static double calcReprojectionErrors(const CVVVPoint3f& objectPoints,
                                         const CVVVPoint2f& imagePoints,
                                         const CVVMat&      rvecs,
                                         const CVVMat&      tvecs,
                                         const CVMat&       cameraMatrix,
                                         const CVMat&       distCoeffs,
                                         vector<float>&     perViewErrors);
    static void   calcBoardCorners3D(const CVSize& boardSize,
                                     float         squareSize,
                                     CVVPoint3f&   objectPoints3D);

    State _state                 = State::Streaming;
    bool  _calibrationSuccessful = false;

    std::future<bool> _calibrationTask; //!< future object for calculation of calibration in async task

    cv::Mat     _currentImgToExtract;
    CVVVPoint2f _imagePoints;               //!< 2D vector of corner points in chessboard
    CVSize      _boardSize;                 //!< NO. of inner chessboard corners.
    float       _boardSquareMM      = 10.f; //!< Size of chessboard square in mm
    int         _numOfImgsToCapture = 20;   //!< NO. of images to capture
    int         _numCaptured        = 0;    //!< NO. of images captured
    CVSize      _imageSize;                 //!< Input image size in pixels (after cropping)
    float       _reprojectionError = -1.f;  //!< Reprojection error after calibration

    // constructor transfer parameter
    CVCalibrationEstimatorParams _params;
    int                          _camSizeIndex = -1;
    bool                         _mirroredH    = false;
    bool                         _mirroredV    = false;
    CVCameraType                 _camType      = CVCameraType::FRONTFACING;
    CVCalibration                _calibration;         //!< estimated calibration
    string                       _calibParamsFileName; //!< name of calibration paramters file
    string                       _computerInfos;
    string                       _calibDataPath;
    string                       _calibImgOutputDir;
    string                       _exePath;

    // exception handling from async thread
    bool                            _hasAsyncError = false;
    CVCalibrationEstimatorException _exception;
};

#endif // CVCALIBRATIONESTIMATOR_H
