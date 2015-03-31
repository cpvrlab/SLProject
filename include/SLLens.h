//#############################################################################
//  File:      SLLens.h
//  Author:    Philipp Jüni
//  Date:      October 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLENS_H
#define SLLENS_H

#include <stdafx.h>
#include "SLRevolver.h"

//-----------------------------------------------------------------------------
//! SLLens creates a lens mesh based on SLRevolver
/*!
SLLens creates a lens mesh based on SLRevolver.
Different constructors allow to either create the lens from the values
written in the eyeglass prescription card or from the radius of the lens.<br>
<br>
<b>Lens types:</b>
\image html LensTypes.png
<b>Myopia</b> ( http://en.wikipedia.org/wiki/Myopia )<br>
The eye is too long for its optical power.<br>
To correct myopic (short-sightedness) a diverging lens is needed.
\image html Myopia.png
<b>Hypermetropia</b> ( http://en.wikipedia.org/wiki/Hyperopia )<br>
The eye is too short for its optical power.<br>
To correct presbyopic (far-sightedness) a converging lens is needed.
\image html Hypermetropia.png
*/
class SLLens: public SLRevolver
{  public:
                //! Create a lense with given sphere, cylinder, diameter and thickness
                SLLens          (double sphere,
                                 double cylinder,
                                 SLfloat diameter,
                                 SLfloat thickness,
                                 SLint stacks = 32,
                                 SLint slices = 32,
                                 SLstring name = "Lens",
                                 SLMaterial* mat = 0);

                //! Create a lense with given radius, diameter and thickness
                SLLens          (SLfloat radiusBot,
                                 SLfloat radiusTop,
                                 SLfloat diameter,
                                 SLfloat thickness,
                                 SLint stacks = 32,
                                 SLint slices = 32,
                                 SLstring name = "Lens",
                                 SLMaterial* mat = 0);

               ~SLLens          () {;}

    private:
        void    init            (SLfloat diopterBot,
                                 SLfloat diopterTop,
                                 SLfloat diameter,
                                 SLfloat thickness,
                                 SLint stacks,
                                 SLint slices,
                                 SLMaterial *mat);
        void    generateLens    (SLfloat radiusBot, 
                                 SLfloat radiusTop, 
                                 SLMaterial* mat);
        SLfloat generateLensBot (SLfloat radius);
        SLfloat generateLensTop (SLfloat radius);
        SLfloat calcSagitta     (SLfloat radius);
        
        SLint   _stacks;        //!< NO. of stacks 
        SLfloat _diameter;      //!< The diameter of the lens
        SLfloat _thickness;     //!< The space between the primary planes of lens sides
        SLfloat _radiusBot;     //!< The radius of the bot (front) side of the lens
        SLfloat _radiusTop;     //!< The radius of the top (back) side of the lens
        SLbool  _pointOutput;   //!< debug helper

};
//-----------------------------------------------------------------------------
#endif //SLLENS_H