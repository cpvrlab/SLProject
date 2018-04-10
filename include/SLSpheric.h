//#############################################################################
//  File:      SLSpheric.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSPHERIC_H
#define SLSPHERIC_H

#include <SLRevolver.h>

class SLRay;
class SLMaterial;

//-----------------------------------------------------------------------------
//! SLSphere creates a sphere mesh based on SLRevolver     
class SLSpheric: public SLRevolver 
{  public:                     
                        SLSpheric   (SLfloat radius,
                                     SLfloat thetaStartDEG,
                                     SLfloat thetaEndDEG,
                                     SLint stacks = 32,
                                     SLint slices = 32,
                                     SLstring name = "spheric mesh",
                                     SLMaterial* mat = 0);
                                                
                       ~SLSpheric   (){;}
                              
            // Getters
            SLfloat     radius      () {return _radius;}
            SLint       stacks      () {return _stacks;}
               
   protected:    
            SLfloat     _radius;        //!< radius of the sphere
            SLfloat     _thetaStartDEG; //!< Polar start angle 0-180°
            SLfloat     _thetaEndDEG;   //!< Polar end angle 1-180°
            SLint       _stacks;        //!< No. of stacks of the sphere
};
//-----------------------------------------------------------------------------
#endif //SLSPHERE_H

