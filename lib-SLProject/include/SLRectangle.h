//#############################################################################
//  File:      SLRectangle.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLRECTANGLE_H
#define SLRECTANGLE_H

#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLRectangle creates a rectangular mesh with a certain resolution
/*! 
The SLRectangle node draws a rectangle with a minimal and a maximal corner in 
the x-y-plane. The normal is [0,0,1].
*/
class SLRectangle : public SLMesh
{
public:
    //! ctor for rectangle w. min & max corner
    SLRectangle(SLAssetManager* assetMgr,
                const SLVec2f&  min,
                const SLVec2f&  max,
                SLuint          resX,
                SLuint          resY,
                SLstring        name = "rectangle mesh",
                SLMaterial*     mat  = nullptr);

    //! ctor for rectangle w. min & max corner & texCoord
    SLRectangle(SLAssetManager* assetMgr,
                const SLVec2f&  min,
                const SLVec2f&  max,
                const SLVec2f&  tmin,
                const SLVec2f&  tmax,
                SLuint          resX,
                SLuint          resY,
                SLstring        name = "rectangle mesh",
                SLMaterial*     mat  = nullptr);

    void buildMesh(SLMaterial* mat);

protected:
    SLVec3f _min;  //!< min corner
    SLVec3f _max;  //!< max corner
    SLVec2f _tmin; //!< min corner texCoord
    SLVec2f _tmax; //!< max corner texCoord
    SLuint  _resX; //!< resolution in x direction
    SLuint  _resY; //!< resolution in y direction
};
//-----------------------------------------------------------------------------
#endif
