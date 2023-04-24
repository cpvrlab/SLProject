//#############################################################################
//  File:      CVTrackedChessboard.cpp
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher
//  License:   This software is provided under the GNU General Public License
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

#include <CVTrackedChessboard.h>
#include <SLFileStorage.h>
#include <Utils.h>

//-----------------------------------------------------------------------------
CVTrackedChessboard::CVTrackedChessboard(string calibIniPath)
  : _calibParamsFileName("calib_in_params.yml"),
    _calibIniPath(calibIniPath)
{
    if (!loadCalibParams())
    {
        Utils::exitMsg("SLProject",
                       "CVTrackedChessboard: could not load calibration parameter",
                       __LINE__,
                       __FILE__);
    }

    calcBoardCorners3D(_boardSize,
                       _edgeLengthM,
                       _boardPoints3D);
    _solved = false;
}
//-----------------------------------------------------------------------------
bool CVTrackedChessboard::loadCalibParams()
{
    string        fullCalibIniFile = _calibIniPath + _calibParamsFileName;
    SLstring      configString     = SLFileStorage::readIntoString(fullCalibIniFile, IOK_config);
    CVFileStorage fs(configString, CVFileStorage::READ | CVFileStorage::MEMORY);

    if (!fs.isOpened())
    {
        Utils::log("SLProject", "Could not open the calibration parameter file: %s", fullCalibIniFile.c_str());
        return false;
    }

    // assign paramters
    fs["numInnerCornersWidth"] >> _boardSize.width;
    fs["numInnerCornersHeight"] >> _boardSize.height;
    // load edge length in MM
    fs["squareSizeMM"] >> _edgeLengthM;
    // convert to M
    _edgeLengthM *= 0.001f;

    return true;
}
//-----------------------------------------------------------------------------
void CVTrackedChessboard::calcBoardCorners3D(const CVSize& boardSize,
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
//! Tracks the chessboard image in the given image for the first sceneview
bool CVTrackedChessboard::track(CVMat          imageGray,
                                CVMat          imageRgb,
                                CVCalibration* calib)
{
    assert(!imageGray.empty() && "ImageGray is empty");
    assert(!imageRgb.empty() && "ImageRGB is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");

    ////////////
    // Detect //
    ////////////

    float startMS = _timer.elapsedTimeInMilliSec();

    // detect chessboard corners
    int flags = cv::CALIB_CB_FAST_CHECK;

    CVVPoint2f corners2D;

    _isVisible = cv::findChessboardCorners(imageGray,
                                           _boardSize,
                                           corners2D,
                                           flags);

    CVTracked::detectTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    if (_isVisible)
    {

        if (_drawDetection)
        {
            cv::drawChessboardCorners(imageRgb, _boardSize, corners2D, true);
        }

        /////////////////////
        // Pose Estimation //
        /////////////////////

        startMS = _timer.elapsedTimeInMilliSec();

        // find the camera extrinsic parameters (rVec & tVec)
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
