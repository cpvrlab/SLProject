//#############################################################################
//  File:      SLCylinder.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCYLINDER_H
#define SLCYLINDER_H

#include <SLRevolver.h>

//-----------------------------------------------------------------------------
//! SLCylinder is creates sphere mesh based on its SLRevolver methods
class SLCylinder : public SLRevolver
{
public:
    SLCylinder(SLAssetManager* assetMgr,
               SLfloat         cylinderRadius,
               SLfloat         cylinderHeight,
               SLuint          stacks    = 1,
               SLuint          slices    = 16,
               SLbool          hasTop    = true,
               SLbool          hasBottom = true,
               SLstring        name      = "cylinder mesh",
               SLMaterial*     mat       = nullptr);

    // Getters
    SLfloat radius() { return _radius; }
    SLfloat height() { return _height; }
    SLbool  hasTop() { return _hasTop; }
    SLbool  hasBottom() { return _hasBottom; }

private:
    SLfloat _radius;    //!< radius of cylinder
    SLfloat _height;    //!< height of cylinder
    SLbool  _hasTop;    //!< Flag if cylinder has a top
    SLbool  _hasBottom; //!< Flag if cylinder has a bottom
};
//-----------------------------------------------------------------------------
#endif // SLCYLINDER_H
