//#############################################################################
//  File:      SLTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLScene.h>
#include <SLTracker.h>

// OpenCV
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
//! destructor
SLTracker::~SLTracker()
{
}

//-----------------------------------------------------------------------------
void SLTracker::initChessboard(int boardWidth, int boardHeight, float edgeLengthM)
{
    _cbSize = Size(boardHeight, boardWidth);

    //set up matrices for storage of translation and rotation vector
    _rVec = Mat(Size(3, 1), CV_64F);
    _tVec = Mat(Size(3, 1), CV_64F);
    //set up matrix for rotation matrix after rodrigues transformation
    _rMat = Mat(3,3,CV_64F);

    //generate vectors for the points on the chessboard
    for (int i = 0; i < boardWidth; i++)
    {
        for (int j = 0; j < boardHeight; j++)
        {
            _boardPoints.push_back(Point3d(double(i * edgeLengthM), double(j * edgeLengthM), 0.0));
        }
    }
}

//-----------------------------------------------------------------------------
void SLTracker::loadCamParams(string filename)
{
    //load camera parameter
    //set up a FileStorage object to read camera params from file
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    // read camera matrix and distortion coefficients from file

    if (!fs.isOpened())
    {
        cout << "Could not open the calibration file" << endl;
        return;
    }

    fs["camera_matrix"] >> _intrinsics;
    fs["distortion_coefficients"] >> _distortion;
    // close the input file
    fs.release();

    //calculate projection matrix
    calculateCameraFieldOfView();
}

//-----------------------------------------------------------------------------
void SLTracker::calculateCameraFieldOfView()
{
    //calculate vertical field of view
    float fy = _intrinsics.at<double>(1,1);
    float cy = _intrinsics.at<double>(1,2);
    float fovRad = 2 * atan2( cy, fy );
    _cameraFovDeg = fovRad * SL_RAD2DEG;
}

//-----------------------------------------------------------------------------
bool SLTracker::trackChessboard()
{
    bool found = false;

    if(!_image.empty())
    {
        //make a gray copy of the webcam image
        cvtColor(_image, _grayImg, CV_RGB2GRAY);

        //detect chessboard corners
        found = findChessboardCorners(_grayImg, _cbSize, _imagePoints );

        if(found)
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

    return found;
}


//-----------------------------------------------------------------------------
