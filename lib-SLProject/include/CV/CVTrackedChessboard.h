//#############################################################################
//  File:      CVTrackedChessboard.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVCHESSBOARDTRACKER_H
#define CVCHESSBOARDTRACKER_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVTypedefs.h>
#include <CVTracked.h>

//-----------------------------------------------------------------------------
//! OpenCV chessboard tracker class derived from CVTracked
/*! The chessboard tracker uses the same chessboard pattern as the calibration
class defined in CVCalibration. See CVTrackedChessboard::track for the
core tracking implementation.
*/
class CVTrackedChessboard : public CVTracked
{
public:
    explicit CVTrackedChessboard();

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib) final;

private:
    void calcBoardCorners3D(const CVSize& boardSize,
                            float         squareSize,
                            CVVPoint3f&   objectPoints3D);
    bool loadCalibParams();

    std::string _calibParamsFileName;
    float       _edgeLengthM;   //<! Length of chessboard square in meters
    CVVPoint3f  _boardPoints3D; //<! chessboard corners in world coordinate system
    CVSize      _boardSize;     //<! NO. of inner chessboard corners
    bool        _solved;        //<! Flag if last solvePnP was solved
    CVMat       _rVec;          //<! rotation angle vector from solvePnP
    CVMat       _tVec;          //<! translation vector from solvePnP
};
//-----------------------------------------------------------------------------

#endif // CVCHESSBOARDTRACKER_H
