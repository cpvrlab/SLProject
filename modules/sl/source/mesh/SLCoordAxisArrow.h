//#############################################################################
//  File:      SLCoordAxisArrow.h
//  Date:      April 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Jan Dellsperger, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCOORDAXISARROW_H
#define SLCOORDAXISARROW_H

#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! Single arrow for coordinate axis
class SLCoordAxisArrow : public SLMesh
{
public:
    explicit SLCoordAxisArrow(SLAssetManager* assetMgr,
                              SLMaterial*     material        = nullptr,
                              SLfloat         arrowThickness  = 0.05f,
                              SLfloat         arrowHeadLenght = 0.2f,
                              SLfloat         arrowHeadWidth  = 0.1f);

    void buildMesh(SLMaterial* material = nullptr);

private:
    SLfloat _arrowThickness;  //!< Thickness of the arrow
    SLfloat _arrowHeadLength; //!< Lenght of the arrow head
    SLfloat _arrowHeadWidth;  //!< Width of the arrow head
};
//-----------------------------------------------------------------------------

#endif