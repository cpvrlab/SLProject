//#############################################################################
//  File:      SLGrid.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGRID_H
#define SLGRID_H

#include <stdafx.h>
#include "SLMesh.h"

//-----------------------------------------------------------------------------
//! SLGrid creates a rectangular grid with lines with a certain resolution
/*! The SLGrid mesh draws a grid between a minimal and a maximal corner in the
XZ-plane.
*/
class SLGrid: public SLMesh
{  public:                 
                        //! ctor for rectangle w. min & max corner
                        SLGrid(SLVec3f minXZ, SLVec3f maxXZ,
                               SLuint resX, SLuint resZ,
                               SLstring name = "Grid",
                               SLMaterial* mat=0);
               
            void        buildMesh(SLMaterial* mat);
               
   protected:
            SLVec3f    _min;     //!< min corner
            SLVec3f    _max;     //!< max corner
            SLuint     _resX;    //!< resolution in x direction
            SLuint     _resZ;    //!< resolution in z direction
};
//-----------------------------------------------------------------------------
#endif
