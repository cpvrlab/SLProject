//#############################################################################
//  File:      CVTrackedArucoCube.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <cv/CVTrackedArucoCube.h>
#include <cv/CVTrackedAruco.h>

#include <Utils.h>

#include <SLMat4.h>
#include <SLVec3.h>
#include <GlobalTimer.h>

CVTrackedArucoCube::CVTrackedArucoCube(const int trackedMarkerIDs[6], string calibIniPath)
  : CVTrackedAruco(-1, calibIniPath)
{
    for (int i = 0; i < 6; i++)
    {
        _trackedMarkerIDs[i] = trackedMarkerIDs[i];
    }
}

bool CVTrackedArucoCube::track(CVMat          imageGray,
                               CVMat          imageRgb,
                               CVCalibration* calib)
{
    if (!trackAll(imageGray, imageRgb, calib))
    {
        return false;
    }

    float edgeLength     = 0.05f;
    float halfEdgeLength = edgeLength / 2.0f;
    float penLength      = 0.145f;
    float downwardOffset = penLength - halfEdgeLength;

    float x = 0;
    float y = 0;
    float z = 0;
    int numVisibleFaces = 0;

    for (size_t i = 0; i < arucoIDs.size(); ++i)
    {
        SLMat4f faceViewMatrix(objectViewMats[i].val);
        faceViewMatrix.transpose();

        faceViewMatrix.translate(0.0, 0.0, -0.025);

        SLVec3f translation = faceViewMatrix.translation();
        if (arucoIDs[i] == 1) faceViewMatrix.rotate(-90, SLVec3f::AXISY);        // right
        else if (arucoIDs[i] == 2) faceViewMatrix.rotate(-180, SLVec3f::AXISY);  // back
        else if (arucoIDs[i] == 3) faceViewMatrix.rotate(90, SLVec3f::AXISY);    // left
        else if (arucoIDs[i] == 4) faceViewMatrix.rotate(90, SLVec3f::AXISX);    // top

        // Reset translation
        faceViewMatrix.m(12, translation.x);
        faceViewMatrix.m(13, translation.y);
        faceViewMatrix.m(14, translation.z);

        /*
        _objectViewMat = CVMatx44f(faceViewMatrix.m(0), faceViewMatrix.m(4), faceViewMatrix.m(8), faceViewMatrix.m(12),
                                   faceViewMatrix.m(1), faceViewMatrix.m(5), faceViewMatrix.m(9), faceViewMatrix.m(13),
                                   faceViewMatrix.m(2), faceViewMatrix.m(6), faceViewMatrix.m(10), faceViewMatrix.m(14),
                                   faceViewMatrix.m(3), faceViewMatrix.m(7), faceViewMatrix.m(11), faceViewMatrix.m(15));
        return true;
        */

        faceViewMatrix.translate(0.0, -downwardOffset, 0.0);

        x += faceViewMatrix.m(12);
        y += faceViewMatrix.m(13);
        z += faceViewMatrix.m(14);
        numVisibleFaces++;
    }

    if(numVisibleFaces > 0)
    {
        _objectViewMat = CVMatx44f(1, 0, 0, x / numVisibleFaces,
                                   0, 1, 0, y / numVisibleFaces,
                                   0, 0, 1, z / numVisibleFaces,
                                   0, 0, 0, 1);
        return true;
    }

    return false;
}

CVMatx44f CVTrackedArucoCube::getFaceToCubeRotation(CVTrackedArucoCubeFace face)
{
    CVVec3f rotation;

    switch (face)
    {
        case ACF_left: rotation = CVVec3f(0, Utils::HALFPI, 0); break;
        case ACF_right: rotation = CVVec3f(0, -Utils::HALFPI, 0); break;
        case ACF_bottom: rotation = CVVec3f(Utils::HALFPI, 0, 0); break;
        case ACF_top: rotation = CVVec3f(-Utils::HALFPI, 0, 0); break;
        case ACF_back: rotation = CVVec3f(0, 0, 0); break;
        case ACF_front: rotation = CVVec3f(0, Utils::PI, 0); break;
    }

    CVMatx33f rotMat;
    cv::Rodrigues(rotation, rotMat);

    CVMatx44f result(rotMat.val[0], rotMat.val[1], rotMat.val[2], 0,
                     rotMat.val[3], rotMat.val[4], rotMat.val[5], 0,
                     rotMat.val[6], rotMat.val[7], rotMat.val[8], 0);

    return result;
}