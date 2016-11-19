//#############################################################################
//  File:      SLCVTrackerChessboard.cpp
//  Author:    Michael Göttlicher, Marcus Hudritsch
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLCVTrackerChessboard.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
//-----------------------------------------------------------------------------
SLCVTrackerChessboard::SLCVTrackerChessboard(SLNode* node) : SLCVTracker(node)
{
    SLCVCalibration& calib = SLScene::current->calibration();

    //generate vectors for the points on the chessboard
    for (int w = 0; w < calib.boardSize().width; w++)
        for (int h = 0; h < calib.boardSize().height; h++)
            _boardPoints.push_back(Point3d(double(w * calib.boardSquareM()), 
                                           double(h * calib.boardSquareM()), 
                                           0.0));
}
//-----------------------------------------------------------------------------
bool SLCVTrackerChessboard::track(cv::Mat image, 
                                  SLCVCalibration& calib,
                                  SLVSceneView& sceneViews)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib.intrinsics().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
    
    SLCVCalibration& c= SLScene::current->calibration();

    //detect chessboard corners
    int flags = CALIB_CB_ADAPTIVE_THRESH | 
                CALIB_CB_NORMALIZE_IMAGE | 
                CALIB_CB_FAST_CHECK;

    vector<cv::Point2f> corners;

    _isVisible = cv::findChessboardCorners(image, c.boardSize(), corners, flags);

    if(_isVisible)
    {
        cv::Mat rVec, tVec;

        //find the camera extrinsic parameters
        bool solved = solvePnP(Mat(_boardPoints), 
                               Mat(corners), 
                               calib.intrinsics(), 
                               calib.distortion(), 
                               rVec, 
                               tVec, 
                               false, 
                               cv::SOLVEPNP_ITERATIVE);
        if (solved)
        {
            _viewMat = calib.createGLMatrix(tVec, rVec);

            for (auto sv : sceneViews)
            {
                if (_node == sv->camera())
                    _node->om(_viewMat.inverse());
                else
                {   //calculate object matrix (see also calcObjectMatrix)
                    _node->om(sv->camera()->om() * _viewMat);
                    _node->setDrawBitsRec(SL_DB_HIDDEN, false);
                }
            }
            return true;
        }
    }
    
    // Hide tracked node if not visible
    for (auto sv : sceneViews)
        if (_node != sv->camera())
            _node->setDrawBitsRec(SL_DB_HIDDEN, true);

    return false;
}
//------------------------------------------------------------------------------
