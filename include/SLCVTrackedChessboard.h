//#############################################################################
//  File:      SLCVTrackedChessboard.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVCHESSBOARDTRACKER_H
#define SLCVCHESSBOARDTRACKER_H

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVTracked.h>

//-----------------------------------------------------------------------------
//! OpenCV chessboard tracker class derived from SLCVTracked
/*! The chessboard tracker uses the same chessboard pattern as the calibration
class defined in SLCVCalibration. See SLCVTrackedChessboard::track for the
core tracking implementation.
*/
class SLCVTrackedChessboard : public SLCVTracked
{
    public:
                SLCVTrackedChessboard   (SLNode* node);
               ~SLCVTrackedChessboard   () {;}

        bool    track                   (SLCVMat imageGray,
                                         SLCVMat imageRgb,
                                         SLCVCalibration* calib,
                                         SLbool drawDetection,
                                         SLSceneView* sv);
    private:
        SLfloat         _edgeLengthM;   //<! Length of chessboard square in meters
        SLCVVPoint3f    _boardPoints3D; //<! chessboard corners in world coordinate system
        SLCVSize        _boardSize;     //<! NO. of inner chessboard corners
        SLbool          _solved;        //<! Flag if last solvePnP was solved
        SLCVMat         _rVec;          //<! rotation angle vector from solvePnP
        SLCVMat         _tVec;          //<! translation vector from solvePnP
        
};
//-----------------------------------------------------------------------------

#endif // SLCVCHESSBOARDTRACKER_H
