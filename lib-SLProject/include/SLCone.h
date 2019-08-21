//#############################################################################
//  File:      SLCone.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCONE_H
#define SLCONE_H

#include <SLRevolver.h>

//-----------------------------------------------------------------------------
//! SLCone creates a cone mesh based on SLRevolver
class SLCone : public SLRevolver
{
    public:
    SLCone(SLfloat     coneRadius,
           SLfloat     coneHeight,
           SLuint      stacks    = 36,
           SLuint      slices    = 36,
           SLbool      hasBottom = true,
           SLstring    name      = "cone mesh",
           SLMaterial* mat       = nullptr);
    ~SLCone() { ; }

    // Getters
    SLfloat radius() { return _radius; }
    SLfloat height() { return _height; }
    SLuint  stacks() { return _stacks; }
    SLbool  hasBottom() { return _hasBottom; }

    protected:
    SLfloat _radius;    //!< radius of cone
    SLfloat _height;    //!< height of cone
    SLuint  _stacks;    //!< No. of stacks of cone
    SLbool  _hasBottom; //!< Flag if cone has a bottom
};
//-----------------------------------------------------------------------------
#endif //SLCONE_H
