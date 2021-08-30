//#############################################################################
//  File:      SLGrid.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGRID_H
#define SLGRID_H

#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLGrid creates a rectangular grid with lines with a certain resolution
/*! The SLGrid mesh draws a grid between a minimal and a maximal corner in the
XZ-plane.
*/
class SLGrid : public SLMesh
{
public:
    //! ctor for rectangle w. min & max corner
    SLGrid(SLAssetManager* assetMgr,
           SLVec3f         minXZ,
           SLVec3f         maxXZ,
           SLuint          resX,
           SLuint          resZ,
           SLstring        name = "grid mesh",
           SLMaterial*     mat  = nullptr);

    void buildMesh(SLMaterial* mat);

protected:
    SLVec3f _min;  //!< min corner
    SLVec3f _max;  //!< max corner
    SLuint  _resX; //!< resolution in x direction
    SLuint  _resZ; //!< resolution in z direction
};
//-----------------------------------------------------------------------------
#endif
