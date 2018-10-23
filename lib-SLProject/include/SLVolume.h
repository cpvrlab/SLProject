//#############################################################################
//  File:      SLVolume.h
//  Author:    Marcus Hudritsch
//  Date:      February 2013
//  Copyright (c): 2002-2013 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLVOLUME_H
#define SLVOLUME_H

#include <stdafx.h>
#include <SLRevolver.h>

class SLRay;
class SLMaterial;

//-----------------------------------------------------------------------------
//! SLSphere creates a sphere mesh based on SLRevolver     
class SLVolume: public SLRevolver 
{  public:                     
                              SLVolume(SLfloat radius,
                                       SLint   stacks = 32,
                                       SLint   slices = 32,
                                       SLstring name = "Volume",
                                       SLMaterial* mat = 0);
                                           
                             ~SLVolume(){;}
                              
               // Getters
               SLfloat        radius() {return _radius;}
               SLint          stacks() {return _stacks;}
               
   protected:    
               SLfloat        _radius; //!< radius of the sphere
               SLint          _stacks; //!< No. of stacks of the sphere
};
//-----------------------------------------------------------------------------
#endif //SLSPHERE_H

