//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
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
    if( !_p.loadFromFile(paramsFileDir))
        return false;

    //set up matrices for storage of translation and rotation vector
    _rVec = Mat(Size(3, 1), CV_64F);
    _tVec = Mat(Size(3, 1), CV_64F);
    //set up matrix for rotation matrix after rodrigues transformation
    _rMat = Mat(3,3,CV_64F);

    //generate vectors for the points on the chessboard
    for (int i = 0; i < _p.boardWidth; i++)
    {
        for (int j = 0; j < _p.boardHeight; j++)
        {
            _boardPoints.push_back(Point3d(double(i * _p.edgeLengthM), double(j * _p.edgeLengthM), 0.0));
        }
    }

    return true;
}
//-----------------------------------------------------------------------------
bool ARChessboardTracker::track()
{
    bool found = false;

    if(!_image.empty() && !_intrinsics.empty())
    {
        //make a gray copy of the webcam image
        cvtColor(_image, _grayImg, CV_RGB2GRAY);

        //detect chessboard corners
        _cbVisible = findChessboardCorners(_grayImg, Size(_p.boardHeight,_p.boardWidth), _imagePoints );

        if(_cbVisible)
        {
            //find the camera extrinsic parameters
            solvePnP(Mat(_boardPoints), Mat(_imagePoints), _intrinsics, _distortion, _rVec, _tVec, false);

            //Transform calculated position (rotation and translation vector) from openCV to SLProject form
            //as discribed in this post:
            //http://www.morethantechnical.com/2015/02/17/augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode/
            //attention: We dont have to transpose the resulting matrix, because SLProject uses row-major matrices.
            //For direct openGL use you have to transpose the resulting matrix additionally.

            //convert vector to rotation matrix
            Rodrigues(_rVec, _rMat);

            //convert to SLMat4f:
            //y- and z- axis have to be inverted
            /*
                  |  r00   r01   r02   t0 |
                  | -r10  -r11  -r12  -t1 |
              m = | -r20  -r21  -r22  -t2 |
                  |    0     0     0    1 |
            */
            //1st row
            _viewMat(0,0) = _rMat.at<double>(0,0);
            _viewMat(0,1) = _rMat.at<double>(0,1);
            _viewMat(0,2) = _rMat.at<double>(0,2);
            _viewMat(0,3) = _tVec.at<double>(0,0);
            //2nd row
            _viewMat(1,0) = -_rMat.at<double>(1,0);
            _viewMat(1,1) = -_rMat.at<double>(1,1);
            _viewMat(1,2) = -_rMat.at<double>(1,2);
            _viewMat(1,3) = -_tVec.at<double>(1,0);
            //3rd row
            _viewMat(2,0) = -_rMat.at<double>(2,0);
            _viewMat(2,1) = -_rMat.at<double>(2,1);
            _viewMat(2,2) = -_rMat.at<double>(2,2);
            _viewMat(2,3) = -_tVec.at<double>(2,0);
            //4th row
            _viewMat(3,0) = 0.0f;
            _viewMat(3,1) = 0.0f;
            _viewMat(3,2) = 0.0f;
            _viewMat(3,3) = 1.0f;

            //_viewMat.print();
        }
    }

    //todo: return not nessasary
    return _cbVisible;
}
//-----------------------------------------------------------------------------
void ARChessboardTracker::updateSceneView( ARSceneView* sv )
{
    if(!_node)
    {
        //create new box
        SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(1.0f,0.7f,0.7f));
        _node = new SLNode("Box");

        // load coordinate axis arrows
        SLAssimpImporter importer;
        SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
        axesNode->scale(0.3);
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
        sv->camera()->om( camOm );
        //set node visible
        _node->setDrawBitsRec( SL_DB_HIDDEN, false );
    }
    else
        //set invisible
        _node->setDrawBitsRec( SL_DB_HIDDEN, true );
}
//-----------------------------------------------------------------------------
void ARChessboardTracker::unloadSGObjects()
{
    if( _node )
    {
        SLNode* parent = _node->parent();
        parent->deleteChild(_node);
        _node = nullptr;
        parent->updateAABBRec();
    }
}
