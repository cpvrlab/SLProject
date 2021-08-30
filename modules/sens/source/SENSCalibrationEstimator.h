//#############################################################################
//  File:      SENSCalibrationEstimator.h
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENSCALIBRATIONESTIMATOR_H
#define SENSCALIBRATIONESTIMATOR_H

#include <future>
#include <SENSCalibration.h>

using namespace std;

//-----------------------------------------------------------------------------
//!special exception that informs about errors during calibration process
class SENSCalibrationEstimatorException : public std::runtime_error
{
public:
    SENSCalibrationEstimatorException(const std::string& msg, const int line, const std::string& file)
      : std::runtime_error(toMessage(msg, line, file).c_str())
    {
    }

private:
    std::string toMessage(const std::string& msg, const int line, const std::string& file)
    {
        std::stringstream ss;
        ss << msg << ": Exception thrown at line " << line << " in " << file << std::endl;
        return ss.str();
    }
};
//-----------------------------------------------------------------------------
class SENSCalibrationEstimator
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

    SENSCalibrationEstimator(SENSCalibrationEstimatorParams params,
                             int                            camSizeIndex,
                             bool                           mirroredH,
                             bool                           mirroredV,
                             SENSCameraType                 camType,
                             std::string                    computerInfos,
                             std::string                    calibDataPath,
                             std::string                    imageOutputPath,
                             std::string                    exePath);
    ~SENSCalibrationEstimator();

    bool calculate();
    bool updateAndDecorate(cv::Mat        imageColor,
                           const cv::Mat& imageGray,
                           bool           grabFrame,
                           bool           drawCorners = true);

    State state()
    {
        return _state;
    }
    int numImgsToCapture() { return _numOfImgsToCapture; }
    int numCapturedImgs() { return _numCaptured; }

    bool calibrationSuccessful() { return _calibrationSuccessful; }
    //!Get resulting calibration
    SENSCalibration getCalibration() { return *_calibration; }
    bool            isBusyExtracting() { return _state == State::BusyExtracting; }
    bool            isCalculating() { return _state == State::Calculating; }
    bool            isStreaming() { return _state == State::Streaming; }
    bool            isDone() { return _state == State::Done; }
    bool            isDoneCaptureAndSave() { return _state == State::DoneCaptureAndSave; }

    static bool calcCalibration(cv::Size&                          imageSize,
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
                                bool                               useReleaseObjectMethod);

private:
    bool calibrateAsync();
    bool extractAsync();
    bool loadCalibParams();
    void updateExtractAndCalc(bool found, bool grabFrame, cv::Mat imageGray);
    void updateOnlyCapture(bool found, bool grabFrame, cv::Mat imageGray);
    void saveImage(cv::Mat imageGray);

    static double calcReprojectionErrors(const vector<vector<cv::Point3f>>& objectPoints,
                                         const vector<vector<cv::Point2f>>& imagePoints,
                                         const std::vector<cv::Mat>&        rvecs,
                                         const std::vector<cv::Mat>&        tvecs,
                                         const cv::Mat&                     cameraMatrix,
                                         const cv::Mat&                     distCoeffs,
                                         vector<float>&                     perViewErrors);
    static void   calcBoardCorners3D(const cv::Size&           boardSize,
                                     float                     squareSize,
                                     std::vector<cv::Point3f>& objectPoints3D);

    State _state                 = State::Streaming;
    bool  _calibrationSuccessful = false;

    std::future<bool> _calibrationTask; //!< future object for calculation of calibration in async task

    cv::Mat                     _currentImgToExtract;
    vector<vector<cv::Point2f>> _imagePoints;               //!< 2D vector of corner points in chessboard
    cv::Size                    _boardSize;                 //!< NO. of inner chessboard corners.
    float                       _boardSquareMM      = 10.f; //!< Size of chessboard square in mm
    int                         _numOfImgsToCapture = 20;   //!< NO. of images to capture
    int                         _numCaptured        = 0;    //!< NO. of images captured
    cv::Size                    _imageSize;                 //!< Input image size in pixels (after cropping)
    float                       _reprojectionError = -1.f;  //!< Reprojection error after calibration

    //constructor transfer parameter
    SENSCalibrationEstimatorParams   _params;
    int                              _camSizeIndex = -1;
    bool                             _mirroredH    = false;
    bool                             _mirroredV    = false;
    SENSCameraType                   _camType      = SENSCameraType::FRONTFACING;
    std::unique_ptr<SENSCalibration> _calibration;         //!< estimated calibration
    std::string                      _calibParamsFileName; //!< name of calibration paramters file
    std::string                      _computerInfos;
    std::string                      _calibDataPath;
    std::string                      _calibImgOutputDir;
    std::string                      _exePath;

    //exception handling from async thread
    bool                              _hasAsyncError = false;
    SENSCalibrationEstimatorException _exception;
};

#endif // SENSCALIBRATIONESTIMATOR_H
