//#############################################################################
//  File:      CVTrackedArucoCube.h
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVTRACKEDARUCOCUBE_H
#define SLPROJECT_CVTRACKEDARUCOCUBE_H

#include <cv/CVTypedefs.h>
#include <cv/CVTrackedAruco.h>

// TODO: Replace with OpenCV classes, SL not allowed in OpenCV module
#include <SLVec3.h>
#include <SLQuat4.h>

//! OpenCV ArUco cube marker tracker class derived from CVTrackedAruco
/*! Tracks a cube of ArUco markers and averages their values. The origin
 * of the cube is in the center.
 * The markers must be placed in the following manner:
 * ID 0: front
 * ID 1: right
 * ID 2: back
 * ID 3: left
 * ID 4: top
 * ID 5: bottom
 */
class CVTrackedArucoCube : public CVTrackedAruco
{
public:
    CVTrackedArucoCube(string calibIniPath, float edgeLength);

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib);

private:
    CVVec3f  averageVector(vector<CVVec3f> vectors,
                           vector<float>   weights);

    SLQuat4f averageQuaternion(vector<SLQuat4f> quaternions,
                               vector<float>    weights);

private:
    float _edgeLength;

    Averaged<CVVec3f> _averagePosition;
    Averaged<SLQuat4f> _averageRotation;

    SLQuat4f _lastRotation;

};

#endif // SLPROJECT_CVTRACKEDARUCOCUBE_H
