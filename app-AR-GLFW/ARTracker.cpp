//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLScene.h>
#include <ARTracker.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
//! constructor
ARTracker::ARTracker(Mat intrinsics, Mat distoriton) :
    _intrinsics(intrinsics),
    _distortion(distoriton)
{
    //set up matrices for storage of translation and rotation vector
    _rVec = Mat(Size(3, 1), CV_64F);
    _tVec = Mat(Size(3, 1), CV_64F);
    //set up matrix for rotation matrix after rodrigues transformation
    _rMat = Mat(3,3,CV_64F);
}
//-----------------------------------------------------------------------------
//! destructor
ARTracker::~ARTracker()
{
}
//-----------------------------------------------------------------------------
SLMat4f ARTracker::cvMatToGLMat(Mat& tVec, Mat& rMat)
{
    // Transform calculated position (rotation and translation vector) 
    // from openCV to SLProject form as discribed in this post:
    // www.morethantechnical.com/2015/02/17/
    // augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode
    // convert to SLMat4f:
    // y- and z- axis have to be inverted
    /*
    tVec = |  t0,  t1,  t2 |
                                        |  r00   r01   r02   t0 |
           | r00, r10, r20 |            | -r10  -r11  -r12  -t1 |
    rMat = | r01, r11, r21 |    slMat = | -r20  -r21  -r22  -t2 |
           | r02, r12, r22 |            |    0     0     0    1 |
    */

    SLMat4f slMat( rMat.at<double>(0, 0),  rMat.at<double>(0, 1),  rMat.at<double>(0, 2),  tVec.at<double>(0, 0),
                  -rMat.at<double>(1, 0), -rMat.at<double>(1, 1), -rMat.at<double>(1, 2), -tVec.at<double>(1, 0),
                  -rMat.at<double>(2, 0), -rMat.at<double>(2, 1), -rMat.at<double>(2, 2), -tVec.at<double>(2, 0),
                                    0.0f,                   0.0f,                   0.0f,                   1.0f);
    return slMat;
}
//-----------------------------------------------------------------------------
