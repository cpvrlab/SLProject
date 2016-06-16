//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
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
ARChessboardTracker::ARChessboardTracker(Mat intrinsics, Mat distoriton) :
    ARTracker(intrinsics, distoriton),
    _node(nullptr),
    _cbVisible(false)

{
}
//-----------------------------------------------------------------------------
bool ARChessboardTracker::init(string paramsFileDir)
{
    if(!_p.loadFromFile(paramsFileDir))
        return false;

    //generate vectors for the points on the chessboard
    for (int i = 0; i < _p.boardWidth; i++)
        for (int j = 0; j < _p.boardHeight; j++)
            _boardPoints.push_back(Point3d(double(i * _p.edgeLengthM), 
                                           double(j * _p.edgeLengthM), 
                                           0.0));
    return true;
}
//-----------------------------------------------------------------------------
bool ARChessboardTracker::track()
{
    bool found = false;

    if(!_image.empty() && !_intrinsics.empty())
    {
        //make a gray copy of the webcam image
        //cvtColor(_image, _grayImg, CV_RGB2GRAY);

        //detect chessboard corners
        int flags = CALIB_CB_ADAPTIVE_THRESH | 
                    CALIB_CB_NORMALIZE_IMAGE | 
                    CALIB_CB_FAST_CHECK;
        cv::Size size = Size(_p.boardHeight, _p.boardWidth);

        _cbVisible = cv::findChessboardCorners(_image, size, _imagePoints, flags);

        if(_cbVisible)
        {
            //find the camera extrinsic parameters
            bool result = solvePnP(Mat(_boardPoints), 
                                   Mat(_imagePoints), 
                                   _intrinsics, 
                                   _distortion, 
                                   _rVec, 
                                   _tVec, 
                                   false, 
                                   cv::SOLVEPNP_ITERATIVE);

            //convert vector to rotation matrix
            Rodrigues(_rVec, _rMat);

            // Convert cv translation & rotation to OpenGL transform matrix
            _viewMat = cvMatToGLMat(_tVec, _rMat);
        }
    }

    //todo: return not nessasary
    return _cbVisible;
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

        float edgeLength = _p.edgeLengthM * 3;
        _node->addMesh(new SLBox(0.0f, 0.0f, 0.0f, edgeLength, edgeLength, edgeLength, "Box", rMat));

        SLScene::current->root3D()->addChild(_node);
        _node->updateAABBRec();
    }

    if(_cbVisible)
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
