//#############################################################################
//  File:      SLFrustum.h
//  Authors:   Luc Girod
//  Date:      Summer 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Licesne:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLFRUSTUM
#define SLFRUSTUM

//-----------------------------------------------------------------------------
//! Matrix to 6 frustum plane conversion functions
class SLFrustum
{
public:
    static void viewToFrustumPlanes(SLPlane*       planes,
                                    const SLMat4f& projectionMat,
                                    const SLMat4f& viewMat);
    static void viewToFrustumPlanes(SLPlane*       planes,
                                    const SLMat4f& A);
    static void getPointsInViewSpace(SLVec3f* points,
                                     float    fovV,
                                     float    ratio,
                                     float    clipNear,
                                     float    clipFar);
};
//-----------------------------------------------------------------------------
#endif
