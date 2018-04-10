//#############################################################################
//  File:      SLSphere.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSPHERE_H
#define SLSPHERE_H

#include <SLSpheric.h>

class SLRay;
class SLMaterial;

//-----------------------------------------------------------------------------
//! SLSphere creates a sphere mesh based on SLSpheric w. 180 deg polar angle.     
class SLSphere: public SLSpheric 
{  public:                     
                        SLSphere(SLfloat radius,
                                 SLint stacks = 32,
                                 SLint slices = 32,
                                 SLstring name = "sphere mesh",
                                 SLMaterial* mat = 0) : 
                                 SLSpheric(radius, 
                                           0.0f, 180.0f, 
                                           stacks, slices, 
                                           name, mat){;}
                                                
                       ~SLSphere(){;}
};
//-----------------------------------------------------------------------------
#endif //SLSPHERE_H

