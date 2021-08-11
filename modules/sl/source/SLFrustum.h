//#############################################################################
//  File:      SLFrustum.h
//  Author:    Luc Girod
//  Date:      Summer 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Licesne:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLFRUSTUM
#define SLFRUSTUM

//-----------------------------------------------------------------------------
//! ???
class SLFrustum
{
public:
    static void viewToFrustumPlanes(SLPlane*       planes,
                                    const SLMat4f& P,
                                    const SLMat4f& V);
    static void viewToFrustumPlanes(SLPlane*       planes,
                                    const SLMat4f& A);
    static void getPointsEyeSpace(SLVec3f* points,
                                  float    fovV,
                                  float    ratio,
                                  float    near,
                                  float    far);
};
//-----------------------------------------------------------------------------
#endif
