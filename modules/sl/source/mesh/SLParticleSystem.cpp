//#############################################################################
//  File:      SLParticleSystem.cpp
//  Date:      February 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <climits>
#include <SLParticleSystem.h>

//-----------------------------------------------------------------------------
//! SLParticleSystem ctor with a given vector of points
SLParticleSystem::SLParticleSystem(SLAssetManager* assetMgr,
                   const SLfloat&  amount,
                   const SLVec3f& particleGenPos,
                   const SLVec3f& velocityRandomStart,
                   const SLVec3f& velocityRandomEnd,
                   const SLfloat& timeToLive,
                   const SLstring& name,
                   SLMaterial*     material) : SLMesh(assetMgr, name)
{
    assert(!name.empty());

    _primitive = PT_points;

    if (amount > UINT_MAX) // Need to change for number of floats
        SL_EXIT_MSG("SLParticleSystem supports max. 2^32 vertices.");
    for (std::size_t i = 0; i < P.size(); ++i)
    {
        P[i] = particleGenPos;
    }
    //Need to add the rest

    mat(material);
}   
//-----------------------------------------------------------------------------
