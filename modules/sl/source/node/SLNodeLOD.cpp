//#############################################################################
//  File:      SLNodeLOD.cpp
//  Author:    Jan Dellsperger, Marcus Hudritsch
//  Date:      July 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>
#include <SLNodeLOD.h>

//-----------------------------------------------------------------------------
SLNodeLOD::SLNodeLOD()
{
    for (SLint i = 0; i < 101; i++)
    {
        _childIndices[i] = -1;
    }
}
//-----------------------------------------------------------------------------
void SLNodeLOD::addChildLOD(SLNode* child, SLfloat minValue, SLfloat maxValue)
{
    assert(minValue >= 0.0f && minValue <= maxValue);
    assert(maxValue >= minValue && maxValue <= 1.0f);

    SLint childIndex = _children.size();
    for (SLint i = (SLint)(minValue * 100.0f); i <= (SLint)(maxValue * 100.0f); i++)
        _childIndices[i] = childIndex;

    addChild(child);
}
//-----------------------------------------------------------------------------
void SLNodeLOD::cullChildren3D(SLSceneView* sv)
{
    // TODO(dgj1): properly choose LOD to use
    if (!_children.empty())
    {
#if 0 // Distance based LOD selection
        SLint   numChildren  = _children.size();
        SLfloat maxDistance  = 100.0f;
        SLfloat lodThreshold = maxDistance / numChildren;

        SLVec3f cameraPos    = sv->camera()->translationWS();
        SLVec3f nodePos      = translationWS();
        SLfloat distToCamera = nodePos.distance(cameraPos);

        SLint childIndex = (SLint)distToCamera / lodThreshold;
        childIndex       = MAX(0, MIN(childIndex, numChildren - 1));

        _children[childIndex]->cull3DRec(sv);
#else // Area based LOD selection

        SLfloat areaPercentage = _aabb.areaPercentageInSS(sv->scr2fbX(), sv->scr2fbY());
        SLint percentageIndex = std::max(0, std::min((SLint)areaPercentage, 100));
        SLint childIndex = _childIndices[percentageIndex];

        if (childIndex >= 0 && childIndex < _children.size())
        {
            //if (childIndex == 0) SL_LOG("childIndex 0");

            _children[childIndex]->cull3DRec(sv);
        }
#endif
    }
}
//-----------------------------------------------------------------------------