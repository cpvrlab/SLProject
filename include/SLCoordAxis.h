//#############################################################################
//  File:      SLCoordAxis.h
//  Author:    Marcus Hudritsch
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCOORDAXIS_H
#define SLCOORDAXIS_H

#include <SLMesh.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
//! Axis aligned coordinate axis mesh
/*!      
The SLAxis mesh draws axis aligned arrows to indicate the coordiante system.
All arrows have unit lenght. The arrow along x,y & z axis have the colors 
red, green & blue.
*/
class SLCoordAxis : public SLMesh
{
    public:
                        SLCoordAxis (SLfloat arrowThickness=0.05f,
                                     SLfloat arrowHeadLenght=0.2f,
                                     SLfloat arrowHeadWidth=0.1f);
               
            void        buildMesh   ();
   
    private:    
            SLfloat     _arrowThickness;    //!< Thickness of the arrow 
            SLfloat     _arrowHeadLength;   //!< Lenght of the arrow head
            SLfloat     _arrowHeadWidth;    //!< Width of the arrow head
};
//-----------------------------------------------------------------------------
#endif
