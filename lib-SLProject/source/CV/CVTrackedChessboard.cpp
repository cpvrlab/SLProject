//#############################################################################
//  File:      CVTrackedChessboard.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/
#include <CVCapture.h>
#include <CVTrackedChessboard.h>

using namespace cv;
//-----------------------------------------------------------------------------
CVTrackedChessboard::CVTrackedChessboard()
{
    CVCalibration* calib = CVCapture::instance()->activeCalib;
    CVCalibration::calcBoardCorners3D(calib->boardSize(),
                                        calib->boardSquareM(),
                                        _boardPoints3D);
    _solved = false;
}
//-----------------------------------------------------------------------------
//! Tracks the chessboard image in the given image for the first sceneview
bool CVTrackedChessboard::track(CVMat            imageGray,
                                  CVMat            imageRgb,
                                  CVCalibration* calib)
{
    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");

    ////////////
    // Detect //
    ////////////

    float  startMS = _timer.elapsedTimeInMilliSec();

    //detect chessboard corners
    int flags = //CALIB_CB_ADAPTIVE_THRESH |
      //CALIB_CB_NORMALIZE_IMAGE |
      CALIB_CB_FAST_CHECK;

    CVVPoint2f corners2D;

    _isVisible = cv::findChessboardCorners(imageGray,
                                           calib->boardSize(),
                                           corners2D,
                                           flags);

    CVTracked::detectTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    if (_isVisible)
    {

        if (_drawDetection)
        {
            cv::drawChessboardCorners(imageRgb, calib->boardSize(), corners2D, true);
        }

        /////////////////////
        // Pose Estimation //
        /////////////////////

        startMS = _timer.elapsedTimeInMilliSec();

        //find the camera extrinsic parameters (rVec & tVec)
        _solved = solvePnP(CVMat(_boardPoints3D),
                           CVMat(corners2D),
                           calib->cameraMat(),
                           calib->distortion(),
                           _rVec,
                           _tVec,
                           _solved,
                           cv::SOLVEPNP_ITERATIVE);

        CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

        if (_solved)
        {
            _objectViewMat = createGLMatrix(_tVec, _rVec);
            return true;
        }
    }

    return false;
}
//------------------------------------------------------------------------------
