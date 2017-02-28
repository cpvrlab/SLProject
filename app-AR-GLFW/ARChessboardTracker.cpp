//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Goettlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "ARChessboardTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <ARSceneView.h>
#include <SLAssimpImporter.h>
#include <SLBox.h>

using namespace cv;

//-----------------------------------------------------------------------------
bool ARChessboardTracker::init()
{
    SLstring filename = "chessboard_detector_params.yml";
    cv::FileStorage fs(SLCVCalibration::calibIniPath + filename, 
                       cv::FileStorage::READ);
    if(!fs.isOpened())
    {   cout << "Could not find parameter file for Chessboard tracking!" << endl;
        cout << "Tried " << SLCVCalibration::calibIniPath + filename << endl;
        return false;
    }
    fs["boardWidth"]  >> _boardSize.width;
    fs["boardHeight"] >> _boardSize.height;
    fs["edgeLengthM"] >> _edgeLengthM;

    SLCVCalibration::calcBoardCorners3D(_boardSize, _edgeLengthM, _boardPoints3D);
    return true;
}
//-----------------------------------------------------------------------------
bool ARChessboardTracker::track(cv::Mat image, 
                                SLCVCalibration* calib)
{
    bool found = false;

    if(!image.empty() && !calib->cameraMat().empty())
    {
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
            bool result = solvePnP(Mat(_boardPoints3D), 
                                   Mat(corners), 
                                   calib->cameraMat(),
                                   calib->distortion(),
                                   rVec, 
                                   tVec, 
                                   false, 
                                   cv::SOLVEPNP_ITERATIVE);

            // Convert cv translation & rotation to OpenGL transform matrix
            _viewMat = createGLMatrix(tVec, rVec);
        }
    }

    //todo: return not nessasary
    return _isVisible;
}
//-----------------------------------------------------------------------------
void ARChessboardTracker::updateSceneView(ARSceneView* sv)
{
    if(!_node)
    {
        //create new box
        SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(1.0f,0.7f,0.7f));
        _node = new SLNode("Box");

        // load coordinate axis arrows
        SLAssimpImporter importer;
        SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
        axesNode->scale(0.3f);
        _node->addChild(axesNode);

        float e3 = _edgeLengthM * 3;
        _node->addMesh(new SLBox(0.0f, 0.0f, 0.0f, e3, e3, e3, "Box", rMat));

        SLScene::current->root3D()->addChild(_node);
        _node->updateAABBRec();
    }

    if(_isVisible)
    {
        //invert view matrix because we want to set the camera object matrix
        SLMat4f camOm = _viewMat.inverse();

        //update camera with calculated view matrix:
        sv->camera()->om(camOm);

        //set node visible
        _node->setDrawBitsRec(SL_DB_HIDDEN, false);
    }
    else
        //set invisible
        _node->setDrawBitsRec(SL_DB_HIDDEN, true);
}
//-----------------------------------------------------------------------------
void ARChessboardTracker::unloadSGObjects()
{
    if(_node)
    {
        SLNode* parent = _node->parent();
        parent->deleteChild(_node);
        _node = nullptr;
        parent->updateAABBRec();
    }
}
//------------------------------------------------------------------------------
