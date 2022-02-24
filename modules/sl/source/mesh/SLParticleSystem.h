//#############################################################################
//  File:      SLParticleSystem.h
//  Date:      February 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLParticleSystem_H
#define SLParticleSystem_H

#include <SLMesh.h>
#include <SLRnd3f.h>

//-----------------------------------------------------------------------------
//! SLParticleSystem creates
/*! The SLParticleSystem mesh object of witch the vertices are drawn as points.
 */
class SLParticleSystem : public SLMesh
{
public:
    //! Ctor for a given vector of points
    SLParticleSystem(SLAssetManager* assetMgr,
                     const SLfloat&  amount,
                     const SLVec3f&  particleGenPos,
                     const SLVec3f&  velocityRandomStart,
                     const SLVec3f&  velocityRandomEnd,
                     const SLfloat&  timeToLive,
                     const SLstring& name     = "Particle system",
             SLMaterial*     material = nullptr);
};
//-----------------------------------------------------------------------------
#endif
