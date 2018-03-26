//#############################################################################
//  File:      SLCVTracked.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLCVTracked.h>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
//! Create an OpenGL 4x4 matrix from an OpenCV translation & rotation vector
SLMat4f SLCVTracked::createGLMatrix(const SLCVMat& tVec, const SLCVMat& rVec)
{
    // 1) convert the passed rotation vector to a rotation matrix
    SLCVMat rMat;
    Rodrigues(rVec, rMat);

    // 2) Create an OpenGL 4x4 column major matrix from the rotation matrix and 
    // translation vector from openCV as discribed in this post:
    // www.morethantechnical.com/2015/02/17/
    // augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode
      
    // The y- and z- axis have to be inverted:
    /*
    tVec = |  t0,  t1,  t2 |
                                        |  r00   r01   r02   t0 |
           | r00, r10, r20 |            | -r10  -r11  -r12  -t1 |
    rMat = | r01, r11, r21 |    glMat = | -r20  -r21  -r22  -t2 |
           | r02, r12, r22 |            |    0     0     0    1 |
    */

    SLMat4f slMat((SLfloat) rMat.at<double>(0, 0), (SLfloat) rMat.at<double>(0, 1), (SLfloat) rMat.at<double>(0, 2), (SLfloat) tVec.at<double>(0, 0),
                  (SLfloat)-rMat.at<double>(1, 0), (SLfloat)-rMat.at<double>(1, 1), (SLfloat)-rMat.at<double>(1, 2), (SLfloat)-tVec.at<double>(1, 0),
                  (SLfloat)-rMat.at<double>(2, 0), (SLfloat)-rMat.at<double>(2, 1), (SLfloat)-rMat.at<double>(2, 2), (SLfloat)-tVec.at<double>(2, 0),
                                             0.0f,                            0.0f,                            0.0f,                           1.0f);
    return slMat;
}
//-----------------------------------------------------------------------------
//! Creates the OpenCV rvec & tvec vectors from an column major OpenGL 4x4 matrix
void SLCVTracked::createRvecTvec(const SLMat4f glMat, SLCVMat& tVec, SLCVMat& rVec)
{
    // The y- and z- axis have to be inverted:
    /*
    tVec = |  t0,  t1,  t2 |
                                        |  r00   r01   r02   t0 |
           | r00, r10, r20 |            | -r10  -r11  -r12  -t1 |
    rMat = | r01, r11, r21 |    glMat = | -r20  -r21  -r22  -t2 |
           | r02, r12, r22 |            |    0     0     0    1 |
    */
    
    // 1) Create cv rotation matrix from OpenGL rotation matrix
    cv::Matx33d rMat( glMat.m(0),-glMat.m(1),-glMat.m(2),
                      glMat.m(4),-glMat.m(5),-glMat.m(6),
                      glMat.m(8),-glMat.m(9),-glMat.m(10));
    
    // 2) Convert rotation matrix to Rodrigues rotation vector
    Rodrigues(rMat, rVec);
    
    // 3) Create tvec vector from translation components
    tVec.at<double>(0, 0) =  glMat.m(12);
    tVec.at<double>(1, 0) = -glMat.m(13);
    tVec.at<double>(2, 0) = -glMat.m(14);
}
//-----------------------------------------------------------------------------
/*! Calculates the object matrix from the cameraObject and the object view matrix.

Nomenclature:
T = homogenious transformation matrix

a
 T = homogenious transformation matrix with subscript b and superscript a
  b

Subscrips and superscripts:  w = world  o = object  c = camera

c
 T  = Transformation of object with respect to camera coordinate system.
  o   It discribes the position of an object in the camera coordinate system.
We get this Transformation from openCVs solvePNP function.

w       c    -1
 T  = ( T )    = Transformation of camera with respect to world coord.-system.
  c       w      Inversion exchanges sub- and superscript.
This is also called the view matrix.

We can combine two or more homogenious transformations to a new one if the
inner sub- and superscript fit together. The resulting transformation
inherits the superscrip from the left and the subscript from the right
transformation. The following tranformation is what we want to do:

w    w     c
 T  = T  *  T   = Transformation of object with respect to world
  o    c     o    coordinate system (object matrix)
*/
SLMat4f SLCVTracked::calcObjectMatrix(const SLMat4f& cameraObjectMat, 
                                      const SLMat4f& objectViewMat)
{   
    // new object matrix = camera object matrix * object-view matrix
    return cameraObjectMat * objectViewMat;
}
//-----------------------------------------------------------------------------

