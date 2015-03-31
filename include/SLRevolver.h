//#############################################################################
//  File:      SLRevolver.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLREVOLVER_H
#define SLREVOLVER_H

#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLRevolver is an SLMesh object built out of revolving points.
/*! 
SLRevolver is an SLMesh object that is built out of points that are revolved 
in slices around and axis. The surface will be outwards if the points in the
array _revPoints increase towards the axis direction.
If all points in the array _revPoints are different the normals will be 
smoothed. If two consecutive points are identical the normals will define a
hard edge. Texture coords. are cylindrically mapped.
*/      
class SLRevolver: public SLMesh 
{  public:                    
                            //! ctor for generic revolver mesh
                            SLRevolver (SLVVec3f revolvePoints,
                                        SLVec3f  revolveAxis,
                                        SLint    slices = 36, 
                                        SLbool   smoothFirst = false,
                                        SLbool   smoothLast = false,
                                        SLstring name = "Revolver",
                                        SLMaterial* mat = 0);

                            //! ctor for derived revolver shapes
                            SLRevolver  (SLstring name) : SLMesh(name) {;}
                           ~SLRevolver  (){;}
                             
            void           buildMesh   (SLMaterial* mat=0);
              
   protected:    
            SLVVec3f       _revPoints;    //!< Array revolving points
            SLVec3f        _revAxis;      //!< axis of revolution
            SLint          _slices;       //!< NO. of slices 
               
            //! flag if the normal of the first point is eqaual to -revAxis
            SLbool         _smoothFirst;
               
            //! flag if the normal of the last point is eqaual to revAxis
            SLbool         _smoothLast;
};
//-----------------------------------------------------------------------------
#endif //SLREVOLVER_H

