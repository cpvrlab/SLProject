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
bool SLCVTrackerChessboard::init(string paramsDir)
{
    SLstring filename = "chessboard_detector_params.yml";
    cv::FileStorage fs(paramsDir + filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {   cout << "Could not find parameter file for Chessboard tracking!" << endl;
        cout << "Tried " << paramsDir + filename << endl;
        return false;
    }
    fs["boardWidth"]  >> _boardSize.width;
    fs["boardHeight"] >> _boardSize.height;
    fs["edgeLengthM"] >> _edgeLengthM;

    //generate vectors for the points on the chessboard
    for (int i = 0; i < _boardSize.width; i++)
        for (int j = 0; j < _boardSize.height; j++)
            _boardPoints.push_back(Point3d(double(i * _edgeLengthM), 
                                           double(j * _edgeLengthM), 
                                           0.0));
    return true;
}
//-----------------------------------------------------------------------------
bool SLCVTrackerChessboard::track(cv::Mat image, 
                                  SLCVCalibration& calib,
                                  SLVSceneView& sceneViews)
{
    if(image.empty() || 
       calib.intrinsics().empty() ||
       _node == nullptr)
       return false;


    //make a gray copy of the webcam image
    //cvtColor(_image, _grayImg, CV_RGB2GRAY);

    //detect chessboard corners
    int flags = CALIB_CB_ADAPTIVE_THRESH | 
                CALIB_CB_NORMALIZE_IMAGE | 
                CALIB_CB_FAST_CHECK;

    vector<cv::Point2f> corners;

    _isVisible = cv::findChessboardCorners(image, _boardSize, corners, flags);

    if(_isVisible)
    {
        cv::Mat rVec, tVec;

        //find the camera extrinsic parameters
        bool result = solvePnP(Mat(_boardPoints), 
                                Mat(corners), 
                                calib.intrinsics(), 
                                calib.distortion(), 
                                rVec, 
                                tVec, 
                                false, 
                                cv::SOLVEPNP_ITERATIVE);

        // Convert cv translation & rotation vector to OpenGL matrix
        _viewMat = calib.createGLMatrix(tVec, rVec);

        ////invert view matrix because we want to set the camera object matrix
        //SLMat4f camOm = _viewMat.inverse();

        ////update camera with calculated view matrix:
        //sv->camera()->om(camOm);

        _node->setDrawBitsRec(SL_DB_HIDDEN, false);
    }
    else
        _node->setDrawBitsRec(SL_DB_HIDDEN, true);
 
    return true;
}
//------------------------------------------------------------------------------
