//#############################################################################
//  File:      SLPolyline.h
//  Author:    Marcus Hudritsch
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPOLYLINE_H
#define SLPOLYLINE_H

#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLPolyline creates a polyline object
/*! 
The SLPolyline node draws a polyline object
*/
class SLPolyline: public SLMesh
{   public:                 
        //! ctor for polyline with a vector of points
        SLPolyline(SLVVec3f points,
                    SLbool closed = false,
                    SLstring name = "Polyline",
                    SLMaterial* mat=0) : SLMesh(name) 
        {
            assert(points.size()>1);
            P = points;
            _primitive = closed ? PT_lineLoop : PT_lines;
            if (P.size() < 65535)
                 for (SLuint i=0; i<P.size(); ++i) I16.push_back(i);
            else for (SLuint i=0; i<P.size(); ++i) I32.push_back(i);
        }
};
//-----------------------------------------------------------------------------
#endif
