//#############################################################################
//  File:      SLSphere.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSPHERE_H
#define SLSPHERE_H

#include <SLSpheric.h>

#include <utility>

class SLRay;
class SLMaterial;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! SLSphere creates a sphere mesh based on SLSpheric w. 180 deg polar angle.
class SLSphere : public SLSpheric
{
public:
    explicit SLSphere(SLAssetManager* assetMgr,
                      SLfloat         radius,
                      SLuint          stacks = 32,
                      SLuint          slices = 32,
                      SLstring        name   = "sphere mesh",
                      SLMaterial*     mat    = nullptr) : SLSpheric(assetMgr,
                                                             radius,
                                                             0.0f,
                                                             180.0f,
                                                             stacks,
                                                             slices,
                                                             std::move(name),
                                                             mat)
    {
    }
};
//-----------------------------------------------------------------------------
#endif // SLSPHERE_H
