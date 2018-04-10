//#############################################################################
//  File:      SLDisk.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLDISK_h
#define SLDISK_h

#include <SLRevolver.h>

//-----------------------------------------------------------------------------
//! SLDisk creates a disk mesh based on SLRevolver     
class SLDisk: public SLRevolver 
{  public:                     
                            SLDisk (SLfloat     radius = 1.0f,
                                    SLVec3f     revolveAxis = SLVec3f::AXISY,
                                    SLint       slices = 36,
                                    SLbool      doubleSided = false,
                                    SLstring    name = "disk mesh",
                                    SLMaterial* mat = 0);
                           ~SLDisk (){;}

            // Getters
            SLfloat         radius()    {return _radius;}
               
   protected:    
            SLfloat        _radius;         //!< radius of cone
            SLbool         _doubleSided;    //!< flag if disk has two sides
};
//-----------------------------------------------------------------------------
#endif //SLDISK_h

