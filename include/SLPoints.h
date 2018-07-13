//#############################################################################
//  File:      SLPoints.h
//  Author:    Marcus Hudritsch
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPOINTS_H
#define SLPOINTS_H

#include <SLRnd3f.h>
#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLPoints creates
/*! The SLPoints mesh object of witch the vertices are drawn as points.
*/
class SLPoints: public SLMesh
{   public:
                        //! Ctor for a given vector of points
                        SLPoints(SLVVec3f& points,
                                 SLstring name = "point cloud",
                                 SLMaterial* mat=0);

                        //! Ctor for a random point cloud.
                        SLPoints(SLfloat nPoints, SLRnd3f& rnd,
                                 SLstring name = "normal point cloud",
                                 SLMaterial* mat=0);
};
//-----------------------------------------------------------------------------
#endif
