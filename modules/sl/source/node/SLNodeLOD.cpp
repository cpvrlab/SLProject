//#############################################################################
//  File:      SLNodeLOD.cpp
//  Date:      July 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Jan Dellsperger, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>
#include <SLNodeLOD.h>

//-----------------------------------------------------------------------------
//! Adds an LOD node with forced decreasing min LOD coverage
/*!
 * The coverage value is the ratio of the nodes bounding rectangle in screen space
 * to the full viewport. The minLodCoverage value must be > 0.0 and < 1.0 and
 * < than the minLodCoverage of the last node in the LOD group. If the first
 * child node has e.g. a minLodCoverage of 0.1 it means that it will be visible
 * if its bounding rectangle covers more then 10% of the viewport. The first
 * child node in the group must be the one with the highest resolution.
 * @param childToAdd LOD child node pointer to add
 * @param minLodCoverage A value > 0 and < 1 and < than the minLodCoverage
 * of the last node in the LOD group
 * @param levelForSM Level to use for shadow mapping (0 uses the active level)
 */
void SLNodeLOD::addChildLOD(SLNode* childToAdd,
                            SLfloat minLodCoverage,
                            SLubyte levelForSM)
{
    assert(minLodCoverage > 0.0f &&
           minLodCoverage < 1.0f &&
           "SLNodeLOD::addChildLOD min. LOD limit must be > 0 and < 1");

    if (!_children.empty() &&
        _children[_children.size() - 1]->minLodCoverage() <= minLodCoverage)
        SL_EXIT_MSG("SLNodeLOD::addChildLOD: A new child LOD node must have a smaller LOD limit than the last one.");

    childToAdd->minLodCoverage(minLodCoverage);
    childToAdd->levelForSM(levelForSM);
    addChild(childToAdd);

    /* Set new node hidden, so they won't be shadowed if they are outside
     * the view frustum. Only the LOD algorithm in cullChildren3D can make
     * them visible */
    childToAdd->drawBits()->set(SL_DB_HIDDEN, true);
}
//-----------------------------------------------------------------------------
//! Culls the LOD children by evaluating the the screen space coverage
void SLNodeLOD::cullChildren3D(SLSceneView* sv)
{
    if (!_children.empty())
    {
        SLfloat rectCoverage = _aabb.rectCoverageInSS();

        // Set visibility (draw-bit SL_DB_HIDDEN) for each level
        for (SLint i = 0; i < _children.size(); ++i)
        {
            bool isVisible;

            if (i == 0)
                isVisible = rectCoverage >= _children[i]->minLodCoverage();
            else
                isVisible = rectCoverage < _children[i - 1]->minLodCoverage() &&
                            rectCoverage >= _children[i]->minLodCoverage();

            _children[i]->drawBits()->set(SL_DB_HIDDEN, !isVisible);
            _aabb.isVisible(isVisible);

            // cull check only the visible level
            if (isVisible)
                _children[i]->cull3DRec(sv);
        }
    }
}
//-----------------------------------------------------------------------------