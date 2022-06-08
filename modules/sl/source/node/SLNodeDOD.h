//#############################################################################
//  File:      SLNodeDOD.h
//  Date:      June 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLNODEDOD_H
#define SLNODEDOD_H

#include <SLMat4.h>
#include <SLMesh.h>

class SLNodeDOD;

//-----------------------------------------------------------------------------
//! SLVNode typedef for a vector of SLNodes
typedef vector<SLNodeDOD> SLVNodeDOD;
//-----------------------------------------------------------------------------
struct SLNodeDOD
{
    SLNodeDOD(SLMesh* myMesh = nullptr)
      : mesh(myMesh),
        parentID(0),
        childCount(0) {}

    void addChild(SLVNodeDOD& nodes, SLuint parentID, SLNodeDOD node);

    SLuint  parentID;   //!< ID of the parent node
    SLuint  childCount; //!< Number of children
    SLMat4f om;         //!< Object matrix for local transforms
    SLMat4f wm;         //!< World matrix for world transform
    SLMat4f wmI;        //!< Inverse world matrix
    SLMesh* mesh;       //!< Pointer to the mesh if any
};
//-----------------------------------------------------------------------------
/* addChild adds a child node by inserting an SLNodeDOD into a vector in
 * Depth First Search order:
 *
                     | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
                     -------------------------------------
addChild(sg,0,node)  |0 0|
                      ---
addChild(sg,0,node)  |0 1|0 0|
                        * ---
addChild(sg,0,node)  |0 2|0 0|0 0|
                        *     ---
addChild(sg,0,node)  |0 3|0 0|0 0|0 0|
                        *         ---
addChild(sg,2,node)  |0 3|0 0|0 1|2 0|0 0|
                                * ---
addChild(sg,2,node)  |0 3|0 0|0 2|2 0|2 0|0 0|
                                *     ---
addChild(sg,0,node)  |0 4|0 0|0 2|2 0|2 0|0 0|0 0|
                        *                     ---
*/
inline void SLNodeDOD::addChild(SLVNodeDOD& scenegraph,
                                SLuint      parentID,
                                SLNodeDOD   node)
{
    assert(parentID <= scenegraph.size() &&
           "Invalid parentID");

    if (scenegraph.empty())
    {
        // Root node
        scenegraph.push_back(node);
        scenegraph[0].parentID = UINT_MAX;
        scenegraph[0].childCount = 0;
    }
    else
    {
        auto   parentItPos = scenegraph.begin() + parentID;
        SLuint childCount  = scenegraph[parentID].childCount;

        // Find position of last child of parent

        scenegraph.insert(parentItPos + childCount, node);
        scenegraph[parentID].childCount++;
    }
}
//-----------------------------------------------------------------------------
#endif // SLNODEDOD_H
