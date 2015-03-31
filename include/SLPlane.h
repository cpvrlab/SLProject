//#############################################################################
//  File:      math/SLPlane.h
//  Author:    Marcus Hudritsch
//  Purpose:   Declaration of a 3D plane class
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPLANE_H
#define SLPLANE_H

#include <SL.h>
#include <SLVec3.h>

//-----------------------------------------------------------------------------
//! Defines a plane in 3D space with the equation ax + by + cy + d = 0
/*!
SLPlane defines a plane in 3D space with the equation ax + by + cy + d = 0
where [a,b,c] is the plane normal and d is the distance from [0,0,0]. The class
is used to define the 6 planes of the view frustum.
*/
class SLPlane
{
    public:
                        SLPlane(const SLVec3f &v1, 
                                const SLVec3f &v2, 
                                const SLVec3f &v3);
                        SLPlane(){N.set(0,0,1); d=0.0f;}
                       ~SLPlane(){;}
                 
            SLVec3f     N; //!< plane normal
            SLfloat     d; //!< d = -(ax+by+cy) = -normal.dot(point)

            void        setPoints         (const SLVec3f &v1, 
                                           const SLVec3f &v2, 
                                           const SLVec3f &v3);
            void        setNormalAndPoint (const SLVec3f &N, 
                                           const SLVec3f &P);
            void        setCoefficients   (const SLfloat A, 
                                           const SLfloat B, 
                                           const SLfloat C, 
                                           const SLfloat D);
                                     
            //! Returns distance between a point P and the plane
    inline   SLfloat     distToPoint (const SLVec3f &p) {return (d+N.dot(p));}
      
            void        print       (const char* name);
};
//-----------------------------------------------------------------------------
#endif


