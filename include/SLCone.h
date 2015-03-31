//#############################################################################
//  File:      SLCone.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCONE_H
#define SLCONE_H

#include <stdafx.h>
#include <SLRevolver.h>

//-----------------------------------------------------------------------------
//! SLCone creates a cone mesh based on SLRevolver     
class SLCone: public SLRevolver 
{  public:                     
                            SLCone (SLfloat  coneRadius,
                                    SLfloat  coneHeight,
                                    SLint    stacks = 36,
                                    SLint    slices = 36,
                                    SLbool   hasBottom = true,
                                    SLstring name = "Cone",
                                    SLMaterial* mat = 0);
                            ~SLCone (){;}

            // Getters
            SLfloat        radius()    {return _radius;}
            SLfloat        height()    {return _height;}
            SLint          stacks()    {return _stacks;}
            SLbool         hasBottom() {return _hasBottom;}
               
   protected:    
            SLfloat        _radius;    //!< radius of cone
            SLfloat        _height;    //!< height of cone
            SLint          _stacks;    //!< No. of stacks of cone
            SLbool         _hasBottom; //!< Flag if cone has a bottom
};
//-----------------------------------------------------------------------------
#endif //SLCONE_H

