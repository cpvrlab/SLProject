//#############################################################################
//  File:      CVTracked.cpp
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*
The OpenCV library version 3.4 or above with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant APP_USES_CVCAPTURE.
All classes that use OpenCV begin with CV.
See also the class docs for CVCapture, CVCalibration and CVTracked
for a good top down information.
*/

#include <CVTracked.h>

//-----------------------------------------------------------------------------
// Static declarations
AvgFloat CVTracked::trackingTimesMS;
AvgFloat CVTracked::detectTimesMS;
AvgFloat CVTracked::detect1TimesMS;
AvgFloat CVTracked::detect2TimesMS;
AvgFloat CVTracked::matchTimesMS;
AvgFloat CVTracked::optFlowTimesMS;
AvgFloat CVTracked::poseTimesMS;
//-----------------------------------------------------------------------------
void CVTracked::resetTimes()
{
    // Reset all timing variables
    CVTracked::trackingTimesMS.init(60, 0.0f);
    CVTracked::detectTimesMS.init(60, 0.0f);
    CVTracked::detect1TimesMS.init(60, 0.0f);
    CVTracked::detect2TimesMS.init(60, 0.0f);
    CVTracked::matchTimesMS.init(60, 0.0f);
    CVTracked::optFlowTimesMS.init(60, 0.0f);
    CVTracked::poseTimesMS.init(60, 0.0f);
}
//-----------------------------------------------------------------------------
// clang-format off
//-----------------------------------------------------------------------------
//! Create an OpenGL 4x4 matrix from an OpenCV translation & rotation vector
CVMatx44f CVTracked::createGLMatrix(const CVMat& tVec, const CVMat& rVec)
{
    // 1) convert the passed rotation vector to a rotation matrix
    CVMat rMat;
    Rodrigues(rVec, rMat);

    // 2) Create an OpenGL 4x4 column major matrix from the rotation matrix and 
    // translation vector from openCV as described in this post:
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

    CVMatx44f glM((float) rMat.at<double>(0, 0), (float) rMat.at<double>(0, 1), (float) rMat.at<double>(0, 2), (float) tVec.at<double>(0, 0),
                  (float)-rMat.at<double>(1, 0), (float)-rMat.at<double>(1, 1), (float)-rMat.at<double>(1, 2), (float)-tVec.at<double>(1, 0),
                  (float)-rMat.at<double>(2, 0), (float)-rMat.at<double>(2, 1), (float)-rMat.at<double>(2, 2), (float)-tVec.at<double>(2, 0),
                0.0f,                          0.0f,                          0.0f,                          1.0f);
    return glM;
}
//-----------------------------------------------------------------------------
//! Creates the OpenCV rvec & tvec vectors from an column major OpenGL 4x4 matrix
void CVTracked::createRvecTvec(const CVMatx44f& glMat, CVMat& tVec, CVMat& rVec)
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
    CVMatx33f rMat(glMat.val[0], -glMat.val[1], -glMat.val[2],
                   glMat.val[4], -glMat.val[5], -glMat.val[6],
                   glMat.val[8], -glMat.val[9], -glMat.val[10]);
    
    // 2) Convert rotation matrix to Rodrigues rotation vector
    Rodrigues(rMat, rVec);
    
    // 3) Create tvec vector from translation components
    tVec.at<double>(0, 0) =  glMat.val[3];
    tVec.at<double>(1, 0) = -glMat.val[7];
    tVec.at<double>(2, 0) = -glMat.val[11];
}
//-----------------------------------------------------------------------------
/*! Calculates the object matrix from the cameraObject and the object view matrix.
<br>
Nomenclature:
T = homogenous transformation matrix
<br>
<sup>a</sup>T<sub>b</sub> = homogenous transformation matrix with subscript b and superscript a
<br>
Subscrips and superscripts: w = world; o = object; c = camera
<br>
<sup>c</sup>T<sub>o</sub>  = Transformation of object with respect to camera coordinate system.
It describes the position of an object in the camera coordinate system.
We get this transformation from OpenCVs solvePNP function.
<br>
<sup>w</sup>T<sub>c</sub>  = (<sup>c</sup>T<sub>w</sub>)<sup>-1</sup> = Transformation of camera with respect to world coord.-system.
Inversion exchanges sub- and superscript.
<br>
The inverse of the camera to world matrix is the view matrix or camera matrix.
<br>
We can combine two or more homogenous transformations to a new one if the
inner sub- and superscript fit together. The resulting transformation
inherits the superscript from the left and the subscript from the right
transformation. The following transformation is what we want to do:
<br>
<sup>w</sup>T<sub>o</sub> = <sup>w</sup>T<sub>c</sub> * <sup>c</sup>T<sub>o</sub> =
Transformation of object with respect to world coordinate system (object matrix)
*/
CVMatx44f CVTracked::calcObjectMatrix(const CVMatx44f& cameraObjectMat,
                                      const CVMatx44f& objectViewMat)
{   
    // new object matrix = camera object matrix * object-view matrix
    return cameraObjectMat * objectViewMat;
}
//-----------------------------------------------------------------------------
// clang-format on
//-----------------------------------------------------------------------------
CVVec3f CVTracked::averageVector(vector<CVVec3f> vectors,
                                 vector<float>   weights)
{
    if (vectors.size() == 1)
        return vectors[0];

    CVVec3f total;
    float   totalWeights = 0.0f;

    for (int i = 0; i < vectors.size(); i++)
    {
        float weight = weights[i];
        total += vectors[i] * weight;
        totalWeights += weight;
    }

    return total / totalWeights;
}
//-----------------------------------------------------------------------------
SLQuat4f CVTracked::averageQuaternion(vector<SLQuat4f> quaternions,
                                      vector<float>    weights)
{
    // Based on: https://math.stackexchange.com/questions/61146/averaging-quaternions

    if (quaternions.size() == 1)
        return quaternions[0];

    SLQuat4f total(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < quaternions.size(); i++)
    {
        SLQuat4f quaternion = quaternions[i];
        float    weight     = weights[i];

        if (i > 0 && quaternion.dot(quaternions[0]) < 0.0)
            weight = -weight;

        total.set(total.x() + weight * quaternion.x(),
                  total.y() + weight * quaternion.y(),
                  total.z() + weight * quaternion.z(),
                  total.w() + weight * quaternion.w());
    }

    return total.normalized();
}
//-----------------------------------------------------------------------------
