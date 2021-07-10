//#############################################################################
//  File:      SLNodeLOD.h
//  Author:    Jan Dellsperger, Marcus Hudritsch
//  Date:      July 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLNODELOD_H
#define SLNODELOD_H

#include <SLNode.h>

//-----------------------------------------------------------------------------
class SLNodeLOD : public SLNode
{
public:
    SLNodeLOD();
    void         addChildLOD(SLNode* child, SLfloat minValue, SLfloat maxValue);
    virtual void cullChildren3D(SLSceneView* sv);

private:
    SLint _childIndices[101];    //!< child indices at every percent (0-100)
};
//-----------------------------------------------------------------------------
#endif