//#############################################################################
//  File:      CVCalibrationEstimator.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCALIBRATIONESTIMATOR_H
#define CVCALIBRATIONESTIMATOR_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVCalibration.h>
#include <CVTypedefs.h>
#include <future>

using namespace std;

//-----------------------------------------------------------------------------
class CVCalibrationEstimator
{
public:
    enum class State
    {
        Stream = 0,     //!< The calibration is running with live video stream
        Calculating,    //!< The calibration starts during the next frame
        BusyExtracting, //!< The estimator is busy an can not caputure additional images
        Done,
    };

    CVCalibrationEstimator(int calibFlags);
    bool calculate();
    bool updateAndDecorate(CVMat        imageColor,
                           const CVMat& imageGray,
                           bool         grabFrame,
                           bool         drawCorners = true);

    State state() { return _state; }
    int   numImgsToCapture() { return _numOfImgsToCapture; }
    int   numCapturedImgs() { return _numCaptured; }

    //!Get resulting calibration
    CVCalibration getCalibration() { return _calibration; }
    bool          isBusy() { return _state == State::BusyExtracting; }
    bool          isDone() { return _state == State::Done; }

private:
    bool calibrateAsync();
    bool loadCalibParams();

    static bool   calcCalibration(CVSize&            imageSize,
                                  CVMat&             cameraMatrix,
                                  CVMat&             distCoeffs,
                                  const CVVVPoint2f& imagePoints,
                                  CVVMat&            rvecs,
                                  CVVMat&            tvecs,
                                  vector<float>&     reprojErrs,
                                  float&             totalAvgErr,
                                  CVSize&            boardSize,
                                  float              squareSize,
                                  int                flag);
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

    State             _state;
    CVCalibration     _calibration;         //!< estimated calibration
    std::future<bool> _calibrationTask;     //!< future object for calculation of calibration in async task
    int               _calibFlags = 0;      //!< OpenCV calibration flags
    std::string       _calibParamsFileName; //!< name of calibration paramters file

    std::vector<cv::Mat> _calibrationImgs;           //!< Images captured for calibration
    CVVVPoint2f          _imagePoints;               //!< 2D vector of corner points in chessboard
    CVSize               _boardSize;                 //!< NO. of inner chessboard corners.
    float                _boardSquareMM      = 10.f; //!< Size of chessboard square in mm
    int                  _numOfImgsToCapture = 20;   //!< NO. of images to capture
    int                  _numCaptured        = 0;    //!< NO. of images captured
    CVSize               _imageSize;                 //!< Input image size in pixels (after cropping)
    float                _reprojectionError = -1.f;  //!< Reprojection error after calibration
};

#endif // CVCALIBRATIONESTIMATOR_H
