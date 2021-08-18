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
//! Level of detail (LOD) group node based on screen space coverage
/*!
 * See the method addChildLOD for more information
 */
class SLNodeLOD : public SLNode
{
public:
    explicit SLNodeLOD(const SLstring& name = "NodeLOD") : SLNode(name) { ; }

    void         addChildLOD(SLNode* child,
                             SLfloat minLodLimit,
                             SLubyte levelForSM = 0);
    virtual void cullChildren3D(SLSceneView* sv);
};
//-----------------------------------------------------------------------------
#endif