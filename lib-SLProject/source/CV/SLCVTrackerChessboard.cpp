//#############################################################################
//  File:      SLCVTrackerChessboard.cpp
//  Author:    Michael G�ttlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael G�ttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/* 
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/
#include <SLCVTrackerChessboard.h>

using namespace cv;
//-----------------------------------------------------------------------------
SLCVTrackerChessboard::SLCVTrackerChessboard(SLNode* node) : SLCVTracker(node)
{
    SLCVCalibration* calib = SLScene::current->activeCalib();
    SLCVCalibration::calcBoardCorners3D(calib->boardSize(),
                                        calib->boardSquareM(),
                                        _boardPoints3D);
    _solved = false;
}
//-----------------------------------------------------------------------------
//! Tracks the chessboard image in the given image for the first sceneview
bool SLCVTrackerChessboard::track(SLCVMat imageGray,
                                  SLCVCalibration* calib,
                                  SLSceneView* sv)
{
    assert(!imageGray.empty() && "Image is empty");
    assert(!calib->cameraMat().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    assert(sv && "No sceneview pointer passed");
    assert(sv->camera() && "No active camera in sceneview");

    //detect chessboard corners
    SLint flags = //CALIB_CB_ADAPTIVE_THRESH |
                  CALIB_CB_NORMALIZE_IMAGE |
                  CALIB_CB_FAST_CHECK;

    SLCVVPoint2f corners2D;

    _isVisible = cv::findChessboardCorners(imageGray,
                                           calib->boardSize(),
                                           corners2D,
                                           flags);

    if(_isVisible)
    {
        //find the camera extrinsic parameters (rVec & tVec)
        _solved = solvePnP(SLCVMat(_boardPoints3D),
                           SLCVMat(corners2D),
                           calib->cameraMat(),
                           calib->distortion(),
                           _rVec,
                           _tVec,
                           _solved,
                           cv::SOLVEPNP_ITERATIVE);
        if (_solved)
        {
            _objectViewMat = createGLMatrix(_tVec, _rVec);

            // set the object matrix depending if the
            // tracked node is attached to a camera or not
            if (typeid(*_node)==typeid(SLCamera))
                _node->om(_objectViewMat.inverse());
            else
            {   
				_node->om(calcObjectMatrix(sv->camera()->om(), _objectViewMat));
                _node->setDrawBitsRec(SL_DB_HIDDEN, false);
            }
            return true;
        }
    }
    
    // Hide tracked node if not visible
    if (_node != sv->camera())
        _node->setDrawBitsRec(SL_DB_HIDDEN, true);

    return false;
}
//------------------------------------------------------------------------------
