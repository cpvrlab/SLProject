//#############################################################################
//  File:      math/SLAlgo.cpp
//  Author:    Michael Goettlicher
//  Purpose:   Container for general algorithm functions
//  Date:      November
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line
#include <SLAlgo.h>

namespace SLAlgo
{

bool estimateHorizon(const SLMat3f& enuRs, const SLMat3f& sRc, SLVec3f& horizon)
{
    SLMat3f cRenu = (enuRs * sRc).transposed();
    //estimate horizon in camera frame:
    //-normal vector of camera x-y-plane in enu frame definition: this is the camera z-axis epressed in enu frame
    SLVec3f normalCamXYPlane = SLVec3f(0, 0, 1);
    //-normal vector of enu x-y-plane in camera frame: this is the enu z-axis rotated into camera coord. frame
    SLVec3f normalEnuXYPlane = cRenu * SLVec3f(0, 0, 1);
    //-Estimation of intersetion line (horizon):
    //Then the crossproduct of both vectors defines the direction of the intersection line. In our special case we know that the origin is a point that lies on both planes.
    //Then origin together with the direction vector define the horizon.
    horizon.cross(normalEnuXYPlane, normalCamXYPlane);

    //check that vectors are not parallel
    float l = horizon.length();
    if(l < 0.01f)
    {
        horizon = {1.f, 0.f, 0.f};
        return false;
    }
    else
    {
        horizon /= l;
        return true;
    }
}

};
